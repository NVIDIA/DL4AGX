# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import pickle
import os
import json
import types
import torch
import inspect
from copy import deepcopy
from functools import wraps
from contextlib import contextmanager

from .cache import Cache
from .stack import StackManager
from .args import meta
from .traceable_dict import TraceableDict

class Record(object):
    def _walk(self, stack):
        ret = []
        for f in stack[1:]:
            if f.function not in ('_wrapped_call_impl', '_call_impl', '_inner_func'):
                context = [line.strip() for line in f.code_context]
                frm = (str(f.filename), f.lineno, str(f.function), context)
                ret.append(frm)
        return ret

    def __init__(self, call_meta, args, kwargs, ret, stack):
        self.meta = call_meta
        self.meta_args = meta(args)
        self.meta_kwargs = meta(kwargs)
        self.meta_ret = meta(ret)
        self.stack = self._walk(stack)

class Hook(object):
    capturing = False
    capture_mode = None # None, capture, compare
    cache = Cache()
    call_stack = StackManager()
    history = []

    @classmethod
    @contextmanager
    def capture(cls, mode="capture", buf=None):
        try:
            cls.capture_mode = mode
            cls.history = []
            cls.call_stack.clear()
            _tmp = cls.cache._capture
            if buf is not None:
                cls.cache._capture = buf
            yield
        finally:
            cls.capture_mode = None
            cls.cache._capture = _tmp

    def __init__(self, body, attr, prefix) -> None:
        self.prefix = prefix
        self.body = body
        self.attr = attr
        self.funcs = []  # record all functions?        
        fn = getattr(self.body, self.attr)
        self.init(fn)
        self.wrap(fn)  # hijack original function call        
        self.records = []
        
    def init(self, fn):
        if hasattr(fn, "__wrapped__") and fn.__wrapped__ is not None:
            fn = fn.__wrapped__

        self.fname = fn.__code__.co_name
        self.key = f"{self.prefix}.{self.fname}"
        try:
            # TODO: handle partial functions?
            _, self.lineno = inspect.findsource(fn)
            self.pth = inspect.getabsfile(fn)            
            self.sig = inspect.signature(fn)
        except Exception as _:
            print(fn)

    def __str__(self) -> str:
        return f"hooked {self.body.__class__}.{self.attr}, {self.key}"

    def _wrap(self, func):
        me = self

        @wraps(func)
        def _inner_func(*args, **kwargs):
            # TODO: args might have no element
            me.before(args[0])
            
            if self.key in Hook.cache._capture and Hook.capture_mode == "capture":
                args_call = self.handle_args(args)
                # make sure some inplace ops won't affect us
                args_hold = deepcopy(args_call)
                kwargs_hold = deepcopy(kwargs)
            else:
                args_hold = args_call = self.handle_args(args)  # consider if we should skip self / ??
                kwargs_hold = kwargs

            with Hook.call_stack.scope(self.key, f"{self.pth}:{self.lineno} type: {type(self.body)}"):
                ret = me.func(*args_call, **kwargs)  # real forward

            if Hook.capture_mode == "capture":
                stack = inspect.stack()  # this is somehow causing gpu-mem leak
                if inspect.ismodule(self.body):
                    body_ = str(self.body)
                else:
                    body_ = str(type(self.body))
                Hook.history.append(Record(
                    (self.pth, self.lineno, self.fname, self.key, str(self.sig), body_),
                    args_hold, kwargs_hold, ret, stack))

                me.after(args_hold, kwargs_hold, ret)
            return ret
        
        _inner_func.hook = self
        return _inner_func

    def wrap(self, func):
        self.func = func
        fn = self._wrap(self.func)
        setattr(self.body, self.attr, fn)

    def unwrap(self):
        # recover to original fn
        setattr(self.body, self.attr, self.func)

    def before(self, body):
        # for function call, no need to do anything here
        return

    def handle_args(self, args):
        return args

    def after(self, args_, kwargs, ret):
        if Hook.capture_mode:
            # args_ = self.handle_args(args)
            if Hook.capture_mode == "mirror":
                if self.key in Hook.cache._mirror:
                    Hook.cache._mirror[self.key].append(dict(
                        args=deepcopy(args_), 
                        kwargs=deepcopy(kwargs), 
                        ret=deepcopy(ret)))

            if self.key in Hook.cache._capture:
                Hook.cache._capture[self.key].append(dict(
                    args=deepcopy(args_), 
                    kwargs=deepcopy(kwargs), 
                    ret=deepcopy(ret)))

    def _patch(self, func):
        # self.funcs.append(self.func)
        self.wrap(func)

    @classmethod
    def patch(cls, from_func, to_func):
        if hasattr(from_func, "hook"):
            h = from_func.hook
            h._patch(to_func)

class HookMethod(Hook):
    def before(self, body):     
        if Hook.capture_mode:
            if self.key in Hook.cache._capture and Hook.capture_mode == "capture":
                item = Hook.cache._capture[self.key]
                if hasattr(item, "before"):
                    item.before(self.func, body)

    def wrap(self, func):
        self.func = types.MethodType(func, self.body) if inspect.isfunction(func) else func 
        fn = self._wrap(self.func)
        setattr(self.body, self.attr, types.MethodType(fn, self.body))

    def handle_args(self, args):
        return args[1:] # skip self
    
class HookTorchFunc(Hook):
    def handle_args(self, args):
        return args[1:] # skip args

def watch(items):
    for k, v in items:
        Hook.cache._capture[k] = [] if v is None else v

class HookHelper(object):
    _default_should_check_content = set([
        torch.nn.modules.container.Sequential,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.container.ModuleDict,
    ])
    
    _default_should_skip = set([
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.activation.Sigmoid,
        torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.linear.Identity,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.normalization.LayerNorm,
        torch.nn.modules.pooling.AvgPool2d,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.dropout.Dropout,        
    ])

    def __init__(self) -> None:
        self._classes = set()
        self._hooked = dict()
        self._fn_attached = dict()
        self._capturing = False
        self.hooks = {}
        # hook_functionals(self, "torch.nn.functional")

    def should_check_content(self, mod, name):
        return isinstance(mod, 
                          tuple(self._default_should_check_content))

    def should_skip(self, mod, name):
        return isinstance(mod, 
                          tuple(self._default_should_skip))
    
    def attach_hook(self, mod, scope):
        self._classes.add(type(mod))
        self._inner_attach(mod, scope)

    def _inner_attach_callable(self, mod, prefix):
        f1 = isinstance(mod, torch.nn.Module)
        f2 = not isinstance(mod, (torch.nn.modules.container.ModuleList, 
                                  torch.nn.modules.container.Sequential))
        if f1 and f2:
            _fn_names = set()
            _fn_name_lut = dict()
            mod_queue = [mod.__class__]
            while len(mod_queue) > 0:
                # handle function call from base classes
                _curr_mod = mod_queue.pop(0)
                if _curr_mod == torch.nn.modules.module.Module:
                    # reaches the endpoint of inherit chain
                    break
                _curr_fn_names = list(_curr_mod.__dict__.keys())
                for _i in _curr_fn_names:
                    if _i not in _fn_name_lut:
                        _fn_name_lut[_i] = _curr_mod
                _fn_names.update(_curr_fn_names)
                mod_queue.extend(_curr_mod.__bases__)
            _named_mods = set([key for key, _ in mod.named_children()])
            _built_ins = ("forward", 
                          'from_pretrained', 'extra_repr', 'train', 'items',
                          "__module__", "__doc__", "__init__", "__annotations__", 
                          '__constants__', '__repr__', '__setattr__', '__delattr__', '__dir__',
                          "__abstractmethods__", "_abc_impl", 
                          '_freeze_stages', 'init_weights', )

            for _fn_name in _fn_names:
                mid = id(mod)
                if mid not in self._fn_attached:
                    self._fn_attached[mid] = []

                if (_fn_name not in _named_mods) and (_fn_name not in _built_ins):
                    # __dict__ will only have class defined functions
                    # so we don't have to iterate parent classes
                    try:
                        _fn = getattr(mod, _fn_name)
                    except AttributeError as _:
                        # if failed, pass anyway
                        continue

                    if _fn_name not in self._fn_attached[mid]:
                        # setattr(mod, _fn_name, hook)
                        self._fn_attached[mid].append(_fn_name)
                        if isinstance(_fn, types.MethodType):
                            hook = HookMethod(mod, _fn_name, prefix)
                            self.hooks[hook.key] = hook
                        elif isinstance(_fn, types.FunctionType):
                            hook = Hook(mod, _fn_name, prefix)
                            self.hooks[hook.key] = hook
            return

    def _inner_attach(self, mod, prefix):
        if not self.should_skip(mod, prefix):
            self._inner_attach_callable(mod, prefix)

        for name, children in mod.named_children():
            self._classes.add(type(children))
            children_name = prefix + "." + name

            if self.should_check_content(children, children_name):
                n = 0
                for sub_name, sub_children in children.named_children():
                    sub_children_name = children_name + "." + sub_name
                    if not self.should_skip(sub_children, sub_children_name):
                        n += 1
                if n == 0:
                    # if all sub-modules in a container is trivial, just skip it
                    continue            

            if self.should_skip(children, children_name):
                # known modules, no need to jump inside
                continue

            if hasattr(children, "register_forward_hook"):
                cid = id(children)
                if cid not in self._hooked:
                    # h = children.register_forward_hook(_hook_torch_module(self, children_name))
                    hook = HookMethod(children, "forward", children_name)
                    print(f"hooking method {children.__class__}.forward, {prefix}")
                    self._hooked[cid] = (children, hook)
                    self.hooks[hook.key] = hook

            if isinstance(children, torch.nn.Module):
                # dfs
                self._inner_attach(children, children_name)

    def seen_classes(self, path_mapping: callable):
        """ return classes when attaching hooks """
        ret = {}
        for c in self._classes:
            key = c.__module__ + '.' + c.__qualname__
            _, lineno = inspect.findsource(c)
            pth = path_mapping(inspect.getabsfile(c))
            ret[key] = (key, f"{pth}:{lineno+1}")
        return [ret[key] for key in sorted(ret.keys(), reverse=True)]

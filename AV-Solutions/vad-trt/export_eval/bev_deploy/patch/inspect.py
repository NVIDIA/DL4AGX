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

import io
import os
import inspect
from collections import OrderedDict
import numpy as np
import types
import sys

import onnx
import onnxsim
import onnx_graphsurgeon as gs

import torch
from torch.autograd import Function
import torch.nn.functional as F

try:
    from mmcv.runner import force_fp32, auto_fp16
    has_mmcv = True
except ImportError as _:
    has_mmcv = False

from bev_deploy.infer_shape import infer
from bev_deploy.hook import Hook
from bev_deploy.hook.traceable_dict import TraceableDict

class InspectHelper(object):
    def __init__(self, session, key):
        self.session = session
        self.key = key

    def setup(self, ch):
        ch._capture_calls[self.key] = []

    def _unpack(self, ch):
        raise NotImplementedError("must provide _unpack")

    def _export_names(self):
        return [], []

    def _export(self, model_trt, args):
        os.umask(0)
        os.makedirs(f"scratch/{self.session}", exist_ok=True)

        input_names, output_names = self._export_names()
        with io.BytesIO() as f, torch.no_grad():
            torch.onnx.export(
                model_trt, tuple(args), f,
                input_names=input_names, output_names=output_names,
                do_constant_folding=False, # True,
                # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                keep_initializers_as_inputs=True,
                opset_version=15,
                export_modules_as_functions=False,
                verbose=True,)

            f.seek(0)
            onnx_mod = onnx.load(f) # , onnx.ModelProto)
            onnx.save(onnx_mod, f"scratch/{self.session}/{self.session}.onnx")
            
            g = gs.import_onnx(onnx_mod)
            g.toposort().cleanup()
            onnx_mod = gs.export_onnx(g)
            onnx_mod, _ = onnxsim.simplify(onnx_mod)

            onnx_mod = infer.infer_shape(onnx_mod)

            # output =[node.name for node in model.graph.output]
            # input_all = [node.name for node in model.graph.input]
            # input_initializer =  [node.name for node in model.graph.initializer]
            # net_feed_input = list(set(input_all)  - set(input_initializer))

            # print('Inputs: ', net_feed_input)
            # print('Outputs: ', output)

            onnx.save(onnx_mod, f"scratch/{self.session}/sim_{self.session}.onnx")

    def _dump_names(self):
        return [], []

    def _dump(self, args, ret):
        input_names, output_names = self._dump_names()
        for n, t in zip(input_names, args):
            if isinstance(t, torch.Tensor):
                print(n, t.shape, t.dtype)
                if t.dtype == torch.float64:
                    t = t.to(torch.float32)
                t.cpu().detach().numpy().tofile(f"scratch/{self.session}/{n}.bin")
            elif isinstance(t, float):
                print(n, "float")
                np.array([t], dtype=np.float32).tofile(f"scratch/{self.session}/{n}.bin")
            else:
                pass
        if not isinstance(ret, (list, tuple)):
            ret = [ret]
        for n, t in zip(output_names, ret):
            print(n, t.shape, t.dtype)
            if t.dtype == torch.float16:
                t = t.to(torch.float32)
            t.cpu().detach().numpy().tofile(f"scratch/{self.session}/{n}.bin")

    def _model_trt(self, ch, model):
        raise NotImplementedError("must provide _model_trt")

    def _verify(self, ch, ret):
        return True

    def inspect(self, ch, model):
        mod = self._model_trt(ch, model)
        args = self._unpack(ch)
        ret = mod(*args)
        matched = self._verify(ch, ret)
        if not matched:
            print(f"not matched, please check {self.key}")
        try:
            self._export(mod, args)
        except RuntimeError as _:
            print(_)
        self._dump(args, ret)

class StatefulInspectHelper(InspectHelper):
    def __init__(self, session, key):
        self.session = session
        self.key = key

    def _unpack(self, ch):
        raise NotImplementedError("must provide _unpack")
    
    def _states(self, ch):
        raise NotImplementedError("must provide _unpack")

    def inspect(self, ch, model):
        mod = self._model_trt(ch, model)
        args = self._unpack(ch)
        ret = mod(*args)
        matched = self._verify(ch, ret)
        if not matched:
            print(f"not matched, please check {self.key}")
        self._export(mod, args)
        self._dump(args, ret)

class StaefulModuleTrt(torch.nn.Module):
    def __init__(self, mod, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod = mod

    def set_state(self, *args):
        raise NotImplementedError("missing set_state impl")

    def get_state(self):
        raise NotImplementedError("missing get_state impl")

    def forward(self, *args, **kwargs):
        remain = self.set_state(*args)
        ret = self.mod.forward(*remain, **kwargs)
        ss = self.get_state()
        return ret + ss

class AutoArg(object):
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name

class AutoConst(object):
    def __init__(self, const, name):
        self.const = const
        self.name = name

class AutoModule(object):
    def __init__(self, mod, name):
        self.mod = mod
        self.name = name

class AutoUnpackArgs(object):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        self.ignore = set()
        self.idx = 0
        self.unpacked = []
        self.mappings = dict()
        self.mods = []
    
    def _inner_unpack(self, element, name):
        if name in self.ignore:
            return None
        if isinstance(element, np.ndarray):
            raise RuntimeError(f"{name} is np.ndarray")
        elif isinstance(element, torch.Tensor):            
            a = AutoArg(self.idx, name); self.idx += 1
            self.unpacked.append((name, element))
            return a
        elif isinstance(element, (tuple, list)):
            tp = type(element)
            return tp([self._inner_unpack(e, f"{name}.{i}") for i, e in enumerate(element)])
        elif isinstance(element, dict):
            return {k: self._inner_unpack(v, f"{name}[{k}]") for k, v in element.items()}
        elif isinstance(element, TraceableDict):
            d = element.dic
            ac = set([r[2] for r in element.access])
            is_all_element_tensor = True
            for k, v in d.items():
                if not isinstance(v, torch.Tensor):
                    is_all_element_tensor = False
                    break
            keys = list(ac)
            if len(ac) == 0 and is_all_element_tensor:
                keys = list(d.keys())
            return {k: self._inner_unpack(d[k], f"{name}[{k}]") for k in keys}
        elif isinstance(element, torch.nn.Module):
            self.mods.append(AutoModule(element, name))
            return AutoModule(element, name)
        else:
            return AutoConst(element, name)
    
    def unpack(self):
        a = []
        for k, v in self.args.items():
            a.append(self._inner_unpack(v, k))
        kw = {}
        if "kwargs" not in self.ignore:
            for k, v in self.kwargs.items():
                kw[k] = self._inner_unpack(v, k)
        self.mappings = dict(args=a, kwargs=kw, mod=self.mods)

class AutoInspectModule(torch.nn.Module):
    def __init__(self, mod, attr, mappings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod = mod
        self.attr = attr
        self.func = getattr(mod, attr)  # <-- must do this within module?
        self.mappings = mappings

        # just in case some pass module as argument
        for v in self.mappings["mod"]:
            if isinstance(v, AutoModule):
                setattr(self, v.name, v.mod)
    
    def pack_args(self, element, *args):
        # rebuild args according to give mappings
        # will also contain some constant values
        if isinstance(element, AutoArg):
            return args[element.idx]
        elif isinstance(element, AutoConst):
            return element.const
        elif isinstance(element, AutoModule):
            return getattr(self, element.name)
        elif isinstance(element, (tuple, list)):
            tp = type(element)
            return tp(self.pack_args(x, *args) for x in element)
        elif isinstance(element, dict):
            return {k: self.pack_args(v, *args) for k, v in element.items()}
        elif element is None:
            return None
        else:
            raise NotImplementedError(type(element))

    def pack_kwargs(self, element, *args):
        # looking for kwargs in mappings
        if isinstance(element, AutoArg):
            return args[element.idx]
        elif isinstance(element, AutoConst):
            return element.const
        elif isinstance(element, (tuple, list)):
            tp = type(element)
            return tp(self.pack_args(x, *args) for x in element)
        elif isinstance(element, dict):
            return {k: self.pack_args(v, *args) for k, v in element.items()}
        elif element is None:
            return None
        else:
            raise NotImplementedError(type(element))

    def forward(self, *args):
        args_ = self.pack_args(self.mappings["args"], *args)
        kwargs_ = self.pack_kwargs(self.mappings["kwargs"], *args)
        return self.func(*args_, **kwargs_)  # NOTE: what if function return a dict? Or maybe a very long list?

def _apply_tdict(obj):
    if isinstance(obj, dict):
        return TraceableDict(obj)
    elif isinstance(obj, tuple):
        return tuple(_apply_tdict(x) for x in obj)
    elif isinstance(obj, list):
        return [_apply_tdict(x) for x in obj]
    else:
        return obj

class _DecideName(object):
    def __init__(self):
        self.names = []
        self.rets = []

    def decide_name(self, name, x):
        if isinstance(x, (tuple, list)):
            for i, v in enumerate(x):
                self.decide_name(f"{name}.{i}", v)
        elif isinstance(x, dict):
            for k, v in x.items():
                self.decide_name(f"{name}.{k}", v)
        elif isinstance(x, torch.Tensor):
            self.names.append(name)
            self.rets.append(x)
        else:
            pass
        return

class AutoInspectHelper(object):
    def __init__(self, hook, update_input=None):
        self.hook: Hook = hook
        self.msgs = []
        self.update_input = [] if update_input is None else update_input
        self.exported = False
        self.input_names = []
        self.output_names = []
    
    def _verify(self, a, b):
        return True

    def _re_trace(self, just_save=False):
        if self.hook.key not in Hook.cache._capture:
            raise RuntimeError(f"missing {self.hook.key} in cache")
        runs = Hook.cache._capture[self.hook.key]
        if len(runs) == 0:
            return
        run = runs[0]

        # update the inputs, like convert from numpy to torch.tensor
        args, kwargs = run["args"], run["kwargs"]

        for fn in self.update_input:
            args, kwargs = fn(args, kwargs)

        # holds the original return object
        self.ret = run["ret"]

        # analysis signature
        sig = inspect.signature(self.hook.func)
        bd = sig.bind(*args, **kwargs)
        bd.apply_defaults()

        # convert dict to traceable dict
        args_ = [_apply_tdict(a) for a in bd.args]
        kwargs_ = {k: _apply_tdict(v) for k, v in bd.kwargs.items()}
        # TODO: this is buggy when just_save
        
        if not just_save:
            # clone the tracing setup
            cc = {k: [] for k in Hook.cache._capture.keys()}
            with Hook.capture(mode="capture", buf=cc) as _, torch.no_grad() as _:
                # recall the func, with traceable dict, so we know which key we accessed
                ret = self.hook.func(*args_, **kwargs_)
            # retraced results will go into 'cc'
            # we can trace down with cc, and see if any divergence happens in the middle
            if not self._verify(run["ret"], ret):
                exit(-1)

        # flatten to plain args
        named_args_ = OrderedDict()
        for i, k in enumerate(list(bd.arguments.keys())[:len(args_)]):
            named_args_[k] = args_[i] # bd.arguments[k]
        self._up(named_args_, kwargs_)

    def _up(self, named_args_, kwargs_):
        self.up = AutoUnpackArgs(named_args_, kwargs_)
        if hasattr(self.hook.func, "_unused"):
            self.up.ignore = self.up.ignore.union(self.hook.func._unused)
        self.up.unpack()
        print(self.up)
        # TODO: we should always verify if patched forward is identical to original one
        #       we may also check the cached chain

    def get_module(self):
        model_trt = AutoInspectModule(self.hook.body, self.hook.attr, self.up.mappings)
        model_trt = model_trt.eval().cuda().float()
        return model_trt

    def before_save(self, args, ret):
        return args, ret

    def export_onnx(self, model_trt, args, input_names, output_names, session=None):
        if self.exported:
            return self.output_names

        with io.BytesIO() as f, torch.no_grad():
            torch.onnx.export(
                model_trt, tuple(args), f,
                input_names=input_names, output_names=output_names,
                do_constant_folding=False, # True,                
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX, # ONNX, ONNX_FALLTHROUGH
                keep_initializers_as_inputs=True,
                opset_version=15,
                export_modules_as_functions=False,
                verbose=True,)

            f.seek(0)
            onnx_mod = onnx.load(f)
            onnx.save(onnx_mod, f"scratch/{session}/{session}.onnx")

            g = gs.import_onnx(onnx_mod)
            g.toposort().cleanup()
            onnx_mod = gs.export_onnx(g)

            input_all = [node.name for node in onnx_mod.graph.input]
            input_initializer =  [node.name for node in onnx_mod.graph.initializer]
            input_feed = list(set(input_all)  - set(input_initializer))
            output = [node.name for node in onnx_mod.graph.output]

            try:
                onnx_mod, _ = onnxsim.simplify(onnx_mod)
                onnx_mod = infer.infer_shape(onnx_mod)
            except RuntimeError as _:
                # TODO: log onnxsim error
                pass

            # should verify if inputs are as expected, sometimes there is some strange difference at this point
            print('input_feed: ', input_feed)  
            print('output: ', output)

            onnx.save(onnx_mod, f"scratch/{session}/sim_{session}.onnx")
            
        self.output_names = output
        return output
    
    def export(self, session=None, just_save=False):
        session = self.hook.key if session is None else session
        dir_name = "demo_data" if just_save else "scratch"
        os.umask(0)        
        os.makedirs(f"{dir_name}/{session}", exist_ok=True)
        
        self._re_trace(just_save)
        model_trt = self.get_module()
        args = [e[1] for e in self.up.unpacked]
        input_names = [e[0] for e in self.up.unpacked]
        
        if not just_save:
            ret = model_trt.forward(*args)
            d = _DecideName()
            d.decide_name("out", ret)
            output_names = d.names
            output = self.export_onnx(model_trt, args, input_names, output_names, session=session)
        else:
            ret = self.ret
            d = _DecideName()
            d.decide_name("out", ret)
            output = d.names
        
        # dump inputs and outputs
        # TODO: some dtype stuff
        for n, t in zip(input_names, args):
            if isinstance(t, torch.Tensor):
                print(n, t.shape, t.dtype)
                if t.dtype == torch.float64:
                    t = t.to(torch.float32)
                t.cpu().detach().numpy().tofile(f"{dir_name}/{session}/{n}.bin")
            elif isinstance(t, float):
                print(n, "float")
                np.array([t], dtype=np.float32).tofile(f"{dir_name}/{session}/{n}.bin")
            else:
                pass

        if not isinstance(ret, (list, tuple)):
            ret = [ret]

        # let's use onnx's result
        if len(output) != len(d.rets):
            exit(-1)

        for n, t in zip(output, d.rets):
            print(n, t.shape, t.dtype)
            # if t.dtype == torch.float16:
            #     t = t.to(torch.float32)
            t.cpu().detach().numpy().tofile(f"{dir_name}/{session}/{n}.bin")

class AutoStatufulModule(AutoInspectModule):
    def __init__(self, mod, attr, mappings, *args, **kwargs):
        super().__init__(mod, attr, mappings, *args, **kwargs)

    def pack_states(self, element, *args):
        for k, v in element.items():
            setattr(self.mod, k, args[v].detach().clone())

    def forward(self, *args):
        self.pack_states(self.mappings["states"], *args)
        args_ = self.pack_args(self.mappings["args"], *args)
        kwargs_ = self.pack_kwargs(self.mappings["kwargs"], *args)
        return self.func(*args_, **kwargs_)

class AutoStatefulInspectHelper(AutoInspectHelper):
    def __init__(self, hook, update_input=None, states=[]):
        super().__init__(hook, update_input)
        self.states = states

    def _re_trace(self):
        # super(AutoStatefulInspectHelper, self)._re_trace()

        if self.hook.key not in Hook.cache._capture:
            raise RuntimeError(f"missing {self.hook.key} in cache")
        
        self.runs = Hook.cache._capture[self.hook.key]
        if len(self.runs.buf) == 0:
            return

        run = self.runs.buf[0]
        self.before = self.runs.buf_before[0]

        args, kwargs = run["args"], run["kwargs"]
        
        for fn in self.update_input:
            args, kwargs = fn(args, kwargs)

        self.ret = run["ret"]
        sig = inspect.signature(self.hook.func)
        bd = sig.bind(*args, **kwargs)
        bd.apply_defaults()

        args_ = [_apply_tdict(a) for a in bd.args]
        kwargs_ = {k: _apply_tdict(v) for k, v in bd.kwargs.items()}

        # clone the tracing setup
        # retraced results will go into 'cc'
        cc = {k: [] for k in Hook.cache._capture.keys()}

        # NOTE: this ret has state issue? So should we reset the states here?
        with Hook.capture(mode="capture", buf=cc) as _, torch.no_grad() as _:
            # recall the func, with traceable dict, so we know which key we accessed
            ret = self.hook.func(*args_, **kwargs_)

        # flatten to plain args
        named_args_ = OrderedDict()
        for i, k in enumerate(list(bd.arguments.keys())[:len(args_)]):
            named_args_[k] = args_[i] # bd.arguments[k]
        
        self._unpack(named_args_, kwargs_)

    def _unpack(self, named_args_, kwargs_):
        self.up = AutoUnpackArgs(named_args_, kwargs_)
        if hasattr(self.hook.func, "_unused"):
            self.up.ignore = self.up.ignore.union(self.hook.func._unused)

        self.up.unpack()
        self.up.mappings["states"] = {
            k: self.up.idx + i for i, k in enumerate(list(self.before.keys()))
        }

    def get_module(self):
        model_trt = AutoStatufulModule(self.hook.body, self.hook.attr, self.up.mappings)
        model_trt = model_trt.eval().cuda().float()
        return model_trt

    def export(self, session=None):
        session = self.hook.key if session is None else session

        model_trt = self.get_module()
        args = [e[1] for e in self.up.unpacked]
        input_names = [e[0] for e in self.up.unpacked]
        for k in self.before.keys():
            args.append(self.before[k].detach().clone())
            input_names.append(f"state_{k}")
        cc = {k: [] for k in Hook.cache._capture.keys()}

        with Hook.capture(mode="capture", buf=cc) as _, torch.no_grad() as _:
            ret = model_trt.forward(*args)

        if not self._verify(self.ret, ret):
            exit(-1)

        d = _DecideName()
        d.decide_name("out", ret)
        output_names = d.names

        output = self.export_onnx(model_trt, args, input_names, output_names, session=session)
        
        args, rets = self.before_save(args, d.rets)

        # TODO: some dtype stuff
        # TODO: duplicate code
        for n, t in zip(input_names, args):
            if isinstance(t, torch.Tensor):
                print(n, t.shape, t.dtype)
                if t.dtype == torch.float64:
                    t = t.to(torch.float32)
                t.cpu().detach().numpy().tofile(f"scratch/{self.hook.key}/{n}.bin")
            elif isinstance(t, float):
                print(n, "float")
                np.array([t], dtype=np.float32).tofile(f"scratch/{self.hook.key}/{n}.bin")
            else:
                pass

        for n, t in zip(output, rets):
            print(n, t.shape, t.dtype)
            if t.dtype == torch.float16:
                t = t.to(torch.float32)
            t.cpu().detach().numpy().tofile(f"scratch/{self.hook.key}/{n}.bin")

# collect intermediate features as outputs
class AutoStatufulInterModule(AutoStatufulModule):
    def __init__(self, mod, attr, mappings, *args, **kwargs):
        super().__init__(mod, attr, mappings, *args, **kwargs)
        self.inters = {}

    def forward(self, *args):
        self.pack_states(self.mappings["states"], *args)
        args_ = self.pack_args(self.mappings["args"], *args)
        kwargs_ = self.pack_kwargs(self.mappings["kwargs"], *args)
        with Hook.capture(mode="capture", buf=self.inters) as _:
            ret = self.func(*args_, **kwargs_)
        return ret, self.inters

class AutoStatefulInterInspectHelper(AutoStatefulInspectHelper):
    def get_module(self):
        model_trt = AutoStatufulInterModule(self.hook.body, self.hook.attr, self.up.mappings)
        model_trt = model_trt.eval().cuda().float()
        return model_trt

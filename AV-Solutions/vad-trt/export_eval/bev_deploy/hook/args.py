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

"""
Helper functions for printing metainfo for args
"""
import json
from collections import OrderedDict
import inspect

import torch
import numpy as np

from .traceable_dict import TraceableDict

class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small lists on single lines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            if self._is_single_line_list(o):
                return "[" + ", ".join(json.dumps(el) for el in o) + "]"
            elif len(o) > 20:
                self.indentation_level += 1
                output = [self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[" + ", ".join(output) + "]"
            else:
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
        elif isinstance(o, (dict, TraceableDict)):
            self.indentation_level += 1
            output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
            self.indentation_level -= 1
            if len(output) == 0:
                return "{}"
            return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
        else:
            return json.dumps(o)

    def _is_single_line_list(self, o):
        if isinstance(o, (list, tuple)):
            f1 = not any(isinstance(el, (list, tuple, dict)) for el in o)
            f2 = len(str(o)) < 200
            return f1 or f2
        return False

    @property
    def indent_str(self) -> str:
        return " " * self.indentation_level * self.indent
    
    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

class ArgWalker(object):
    def is_builtin(obj):
        return obj.__class__.__module__ == 'builtins'

    def __init__(self) -> None:
        pass

    def walk(self, obj):
        return self._walk(obj)

    def _walk(self, obj):
        if isinstance(obj, torch.Tensor):
            return f"torch.Tensor(shape={[*obj.shape]}, dtype={str(obj.dtype)})"
        elif isinstance(obj, np.ndarray):
            return f"np.ndarray[shape={[*obj.shape]}, dtype={str(obj.dtype)}]"
        elif isinstance(obj, tuple):
            return tuple([self._walk(i) for i in obj])
        elif isinstance(obj, list):
            return [self._walk(i) for i in obj]
        elif isinstance(obj, (dict, OrderedDict)):
            ret = {}
            for k, v in obj.items():
                ret[k] = self._walk(v)
            return ret
        elif isinstance(obj, (TraceableDict)):
            accessed = set([a[2] for a in obj.access])
            ret = {}
            for k, v in obj.items():
                if k in accessed:
                    k = "*" + k + "*"
                ret[k] = self._walk(v)
            return ret
        elif isinstance(obj, (int, float)):
            return obj
        else:
            return type(obj).__name__

def meta(obj):
    aw = ArgWalker()
    return json.dumps(aw.walk(obj), cls=CompactJSONEncoder, indent=2)

def MatchArguments(sig: inspect.Signature, args, kwargs):
    pass

if __name__ == "__main__":
    obj = ("asdf", 1.0, 1, [])
    aw = ArgWalker(obj)
    print(aw)

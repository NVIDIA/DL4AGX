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

import types
from contextlib import contextmanager

class StackItem(object):
    def __init__(self, name, desc):
        self.items = []
        self.name = name
        self.desc = desc

    def add(self, item):
        self.items.append(item)

class StackManager(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self._stack = [StackItem("root", "")]

    @contextmanager
    def scope(self, key, desc):
        top = self._stack[-1]
        me = StackItem(key, desc)
        try:
            top.add(me)
            self._stack.append(me)
            yield None
        finally:
            self._stack.pop()
            pass

    def view(self, fn=None):
        # dfs
        q = [(self._stack[0], 0)]
        while len(q) > 0:
            t, lvl = q.pop()
            desc = fn(t.desc) if fn is not None and callable(fn) else t.desc
            print(("  " * lvl) + t.name, desc)
            for c in t.items[::-1]:
                q.append((c, lvl + 1))

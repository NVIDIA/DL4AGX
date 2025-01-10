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

class Cache(object):
    def __init__(self) -> None:
        self._capture = dict()
        self._mirror = dict()

class BaseCustomCallback(object):
    def __init__(self) -> None:
        self.buf = []
        self.buf_before = []

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, key):
        ret = self.buf[key]
        if len(self.buf_before) > key:
            ret["before"] = self.buf_before[key]
        return ret
    
    def before(self, func, func_self):
        pass

    def append(self, item):
        self.buf.append(item)

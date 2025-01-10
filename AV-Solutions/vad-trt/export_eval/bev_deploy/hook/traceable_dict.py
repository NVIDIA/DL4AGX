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

class TraceableDict(object):
    """ dict, but keeps the key access record """
    def __init__(self, dic) -> None:
        self.dic = dic
        self.access = []

    def _caller_info(self, frame):
        pass
    
    def get(self, key, default=None):
        ret = self.__getitem__(key)
        if ret is None:
            return default

    def items(self):
        return self.dic.items()

    def __contains__(self, key):
        return key in self.dic

    def __getitem__(self, key):
        # get caller loc
        record = ("get", "??", key)
        self.access.append(record)
        return self.dic.get(key)

    def __setitem__(self, key, item):
        record = ("set", "??", key)
        self.access.append(record)
        self.dic[key] = item

if __name__ == "__main__":
    d = dict(roi=1, x=2, y=3)
    dd = TraceableDict(d)
    print("y" in dd)

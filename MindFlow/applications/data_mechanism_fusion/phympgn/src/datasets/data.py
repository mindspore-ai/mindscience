# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""data"""
from mindspore import ops


class Graph:
    """Graph"""
    def __init__(self, **kwargs):
        self.pos = None
        self.edge_index = None
        self.edge_attr = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def detach(self):
        new_graph = Graph()
        for attr, value in self.__dict__.items():
            new_graph.__setattr__(str(attr), ops.stop_gradient(value))
        return new_graph

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        infos = []
        for attr, value in self.__dict__.items():
            if value is None:
                continue
            out = str(list(value.shape))
            key = str(attr)
            infos.append(f'{key}={out}')
        infos = ', '.join(infos)
        return f'{cls}({infos})'

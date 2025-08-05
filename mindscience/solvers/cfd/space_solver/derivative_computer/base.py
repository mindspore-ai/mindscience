# Copyright 2022 Huawei Technologies Co., Ltd
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
"""abstract base class for derivative computer."""
from abc import abstractmethod


class DerivativeComputer:
    """Abstract base class for derivative computer."""

    def __init__(self, mesh_info):
        self.mesh_info = mesh_info
        self.pad = mesh_info.pad
        self.size = [5,] + mesh_info.number_of_cells

    @abstractmethod
    def derivative(self, var, dxi, axis):
        raise NotImplementedError()

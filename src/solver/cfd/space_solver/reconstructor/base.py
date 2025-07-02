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
"""abstract base class for reconstructor."""
from abc import abstractmethod


class Reconstructor:
    """Abstract base class for reconstructor."""

    def __init__(self, mesh_info):
        self.mesh_info = mesh_info
        self.pad = mesh_info.pad

    def reconstruct_from_left(self, var, axis):
        """Reconstruct variables from left side."""
        return self._reconstruct_on_face(var, axis, 0)

    def reconstruct_from_right(self, var, axis):
        """Reconstruct variables from right side."""
        return self._reconstruct_on_face(var, axis, 1)

    @abstractmethod
    def _reconstruct_on_face(self, var, axis, j):
        raise NotImplementedError()

    def _slice(self, inputs, output_size):
        """
        Take slice of the input tensor according to output_size.

        Args:
            inputs: Tensor. Input tensor.
            output_size : List. Output size.
        Returns:
            Tensor. Output tensor with shape of output_size.
        """
        starts = []
        ends = []

        for i in range(3):
            if inputs.shape[i + 1] == output_size[i + 1]:
                starts.append(0)
                ends.append(inputs.shape[i + 1])
            else:
                starts.append(self.mesh_info.pad)
                ends.append(-self.mesh_info.pad)

        return inputs[:, starts[0]: ends[0], starts[1]: ends[1], starts[2]: ends[2]]

# Copyright 2024 Huawei Technologies Co., Ltd
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
"""init"""

from mindspore import Tensor, nn, int32, ops

from ....e3.utils import Ncon


class MakeWeightedChannels(nn.Cell):
    """MakeWeightedChannels
    """

    def __init__(self, irreps_in, multiplicity_out: int):
        super().__init__()
        w_index = sum(([i] * ir.dim for i, (mul, ir) in enumerate(irreps_in.data)), [])
        n_pad = 0
        w_index += [w_index[-1]] * n_pad
        self.num_irreps = len(irreps_in)
        self.w_index = Tensor(w_index, dtype=int32)
        self.multiplicity_out = multiplicity_out
        self.weight_numel = len(irreps_in) * multiplicity_out
        self.ncon = Ncon([[-1, -3], [-1, -2, -3]])
        self.axis = 2

    def construct(self, edge_attr, weights):
        return self.ncon([edge_attr, ops.index_select(weights.view(-1, self.multiplicity_out, self.num_irreps),
                                                      self.axis, self.w_index)])

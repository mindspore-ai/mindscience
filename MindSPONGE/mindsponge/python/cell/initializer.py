# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
# ============================================================================
"""initializer"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.common.initializer import TruncatedNormal

TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978, dtype=np.float32)


def lecun_init(fan_in, initializer_name='linear'):
    """lecun init"""
    scale = 1.0
    if initializer_name == 'relu':
        scale *= 2
    weight_init = TruncatedNormal(sigma=np.sqrt(scale / fan_in) / TRUNCATED_NORMAL_STDDEV_FACTOR)
    return weight_init


def glorot_uniform(fan_in, fan_out, weight_shape):
    """glorot uniform"""
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=weight_shape)


class OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return F.cast(self._op(x), mstype.float16)


def do_keep_cell_fp32(network):
    """Do keep cell fp32."""
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Softmax, nn.LayerNorm)):
            network._cells[name] = OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            do_keep_cell_fp32(subcell)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())

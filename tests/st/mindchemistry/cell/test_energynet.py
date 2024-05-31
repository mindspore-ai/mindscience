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
# ============================================================================
"""test mindchemistry EnergyNet"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor

from mindchemistry.cell import EnergyNet

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_energynet():
    """
    Feature: Test EnergyNet in platform ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    nergy_net = EnergyNet(
        irreps_embedding_out='16x0e',
        irreps_conv_out='64x0o+64x0e+64x1o+64x1e+64x2o+64x2e',
        chemical_embedding_irreps_out='64x0e',
        num_layers=5,
        num_type=4,
        r_max=4,
        hidden_mul=64,
        pred_force=False,
        dtype=ms.float32
        )

    batch = Tensor(np.ones(60), ms.int32)
    x = Tensor(np.ones(60), ms.int32)
    pos = Tensor(np.random.rand(60, 3), ms.float32)
    edge_src = Tensor(np.ones(660), ms.int64)
    edge_dst = Tensor(np.ones(660), ms.int64)
    batch_size = 5

    final_out = 1

    out = nergy_net(batch, x, pos, edge_src, edge_dst, batch_size)
    assert out.shape == (batch_size, final_out), f"For `EnergyNet`, the output should be\
         ({batch_size}, {final_out}), but got {out.shape}."

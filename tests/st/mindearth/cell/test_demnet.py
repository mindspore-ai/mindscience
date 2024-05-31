# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test mindearth """

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor

from mindearth.cell import DEMNet

@pytest.mark.level1
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_demnet():
    """
    Feature: Test DEMNet in platform gpu and ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    Need to adaptive 910B.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_images = np.random.rand(64, 1, 32, 32).astype(np.float32)
    net = DEMNet(in_channels=1, out_channels=256, kernel_size=3, scale=5, num_blocks=42)
    output = net(Tensor(input_images, ms.float32))
    assert output.shape == (64, 1, 160, 160), f"For `DEMNet`, the output should be (64, 1, 160, 160), \
        but got {output.shape}."

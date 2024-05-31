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
"""test mindearth DgmrDiscriminator"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor

from mindearth.cell import DgmrDiscriminator

@pytest.mark.level1
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dgmrdiscriminator():
    """
    Feature: Test DgmrGenerator in platform gpu and ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    Need to adaptive 910B.
    """
    context.set_context(mode=context.GRAPH_MODE)
    real_and_generator = np.random.rand(2, 22, 1, 256, 256).astype(np.float32)
    net = DgmrDiscriminator(in_channels=1, num_spatial_frames=8, conv_type="standard")
    output = net(Tensor(real_and_generator, ms.float32))
    assert output.shape == (2, 2, 1), f"For `DgmrDiscriminator`, the output should be (2, 2, 1), \
        but got {output.shape}."

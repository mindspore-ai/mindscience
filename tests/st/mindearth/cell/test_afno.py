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
"""test mindearth AFNONet"""
import pytest

import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.common.initializer import initializer, Normal
from mindearth.cell import AFNONet

#pylint: disable=C0103

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_afno():
    """
    Feature: Test AFNONet in platform gpu and ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    # attention: error in mindspore 9.5 with graph mode
    context.set_context(mode=context.PYNATIVE_MODE)
    B, C, H, W = 16, 69, 128, 256
    input_ = initializer(Normal(), [B, C, H, W])
    net = AFNONet(image_size=(H, W), in_channels=C, out_channels=C, compute_dtype=mstype.float32)
    out = net(input_)
    assert out.shape == (16, 512, 4416), f"For `AFNONet`, the output should be (16, 512, 4416), \
        but got {out.shape}."
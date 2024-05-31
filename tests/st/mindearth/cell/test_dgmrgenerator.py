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
"""test mindearth DgmrGenerator"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor

from mindearth.cell import DgmrGenerator

@pytest.mark.level1
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dgmrgenerator():
    """
    Feature: Test DgmrGenerator in platform gpu and ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    Need to adaptive 910B.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_frames = np.random.rand(1, 4, 1, 256, 256).astype(np.float32)
    net = DgmrGenerator(
        forecast_steps=18,
        in_channels=1,
        out_channels=256,
        conv_type="standard",
        latent_channels=768,
        context_channels=384,
        generation_steps=1
    )
    output = net(Tensor(input_frames, ms.float32))
    assert output.shape == (1, 18, 1, 256, 256), f"For `DgmrGenerator`, the output should be (1, 18, 1, 256, 256), \
        but got {output.shape}."

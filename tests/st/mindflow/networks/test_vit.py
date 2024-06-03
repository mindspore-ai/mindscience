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
"""mindflow ut testcase"""

import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindflow.cell import ViT


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_vit_output():
    """
    Feature: Test ViT network in platform gpu and ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    Need to adaptive 910B
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_tensor = Tensor(np.ones((32, 3, 192, 384)), mstype.float32)
    print('input_tensor.shape: ', input_tensor.shape)
    print('input_tensor.dtype: ', input_tensor.dtype)

    model = ViT(in_channels=3,
                out_channels=3,
                encoder_depths=6,
                encoder_embed_dim=768,
                encoder_num_heads=12,
                decoder_depths=6,
                decoder_embed_dim=512,
                decoder_num_heads=16,
                )

    output_tensor = model(input_tensor)
    print('output_tensor.shape: ', output_tensor.shape)
    print('output_tensor.dtype: ', output_tensor.dtype)
    assert output_tensor.shape == (32, 288, 768)
    assert output_tensor.dtype == mstype.float32

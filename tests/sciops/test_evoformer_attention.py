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
"""test sciops evoformer attention"""

import pytest
import numpy as np

import mindspore as ms
from mindspore import ops, Tensor

from mindscience.sciops import evo_attention

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_evoformer_attention_shape():
    """
    Feature: Test Evoformer in platform ascend 910B.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    ms.set_device(device_target="Ascend")
    b, n, s, d = 2048, 1, 2048, 8

    query = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
    key = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
    value = Tensor(np.random.uniform(-0.1, 0.1, (b, s, n, d)), ms.bfloat16)
    bias = Tensor(np.random.uniform(-0.1, 0.1, (1, n, s, s)), ms.bfloat16)

    mask = np.concatenate((np.ones((b, 1, 1, s - 5)).astype(np.float32),
                           np.zeros((b, 1, 1, 5)).astype(np.float32)), axis=-1)
    evo_mask = Tensor(1 - mask.astype(np.uint8))

    output = evo_attention(query, key, value, n, bias, evo_mask, scale_value=1.0, input_layout="BSND")
    assert output.shape == (2048, 2048, 1, 8), f"For `Evoformer_Attention`, the output should be (2048, 2048, 1, 8), \
        but got {output.shape}."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_evoformer_attention_precision():
    """
    Feature: Test Evoformer in platform ascend 910B.
    Description: The forward output should has expected precision.
    Expectation: Success or throw AssertionError.
    """
    ms.set_device(device_target="Ascend")
    b, n, s, d = 128, 1, 128, 2
    scale_value = 2.0

    query = Tensor(np.random.uniform(-0.1, 0.1, (b, n, s, d)), ms.float16)
    key = Tensor(np.random.uniform(-0.1, 0.1, (b, n, s, d)), ms.float16)
    value = Tensor(np.random.uniform(-0.1, 0.1, (b, n, s, d)), ms.float16)
    bias = Tensor(np.random.uniform(-0.1, 0.1, (1, n, s, s)), ms.float16)

    mask = np.concatenate((np.ones((b, 1, 1, s - 5)).astype(np.float32),
                           np.zeros((b, 1, 1, 5)).astype(np.float32)), axis=-1)
    evo_mask = Tensor(1 - mask.astype(np.uint8))

    expected_output = evo_attention(query, key, value, n, bias, evo_mask, scale_value=scale_value, input_layout="BNSD")

    attention_mask = 1e12 * (Tensor(mask) - 1)
    logits = ops.BatchMatMul(transpose_b=True)(query, key)
    logits = logits * scale_value
    logits = ops.add(logits, attention_mask.astype(ms.float16))
    logits = ops.add(logits, bias)
    weight = ops.Softmax()(logits)
    actual_output = ops.BatchMatMul()(weight, value).asnumpy()

    np.testing.assert_allclose(actual_output, expected_output, atol=1e-4, rtol=1e-7)

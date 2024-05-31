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
"""test mindearth get_activation"""

import pytest
import numpy as np

from mindspore import Tensor

from mindearth.cell import get_activation

@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_get_activation():
    """
    Feature: Test get_activation in platform gpu and ascend.
    Description: The learning rate computed by get_activation should have 95% accuracy.
    Expectation: Success or throw AssertionError.
    """
    input_x = Tensor(np.array([[1.2, 0.1], [0.2, 3.2]], dtype=np.float32))
    sigmoid = get_activation('sigmoid')
    output = sigmoid(input_x).asnumpy()
    ans = np.array([[0.7685248, 0.5249792], [0.54983395, 0.96083426]])
    ret = np.isclose(output, ans)
    assert np.sum(ret == 0) / len(ret) < 0.05

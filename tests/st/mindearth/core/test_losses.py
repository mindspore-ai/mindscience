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
"""test mindearth losses"""

import pytest
import numpy as np

import mindspore
from mindspore import Tensor

from mindearth.core import RelativeRMSELoss

@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_relative_rmse_loss():
    """
    Feature: Test RelativeRMSELoss in platform gpu and ascend.
    Description: The learning rate computed by RelativeRMSELoss should have 95% accuracy.
    Expectation: Success or throw AssertionError.
    """
    prediction = Tensor(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), mindspore.float32)
    labels = Tensor(np.array([[1, 2, 2], [1, 2, 3], [1, 2, 3]]), mindspore.float32)
    loss_fn = RelativeRMSELoss()
    loss = loss_fn(prediction, labels).asnumpy()
    assert np.isclose(loss, 0.11111112)

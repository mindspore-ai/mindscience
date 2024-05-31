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
"""test mindearth get_warmup_cosine_annealing_lr"""

import pytest
import numpy as np

from mindearth import get_warmup_cosine_annealing_lr

@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_get_warmup_cosine_annealing_lr():
    """
    Feature: Test get_warmup_cosine_annealing_lr in platform gpu and ascend.
    Description: The learning rate computed by get_warmup_cosine_annealing_lr should have 95% accuracy.
    Expectation: Success or throw AssertionError.
    """
    lr_init = 0.001
    steps_per_epoch = 3
    last_epoch = 5
    warmup_epochs = 1
    lr = get_warmup_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, warmup_epochs=warmup_epochs)
    ans = [3.3333333e-04, 6.6666666e-04, 1.0000000e-03, 9.0460398e-04, 9.0460398e-04,
           9.0460398e-04, 6.5485400e-04, 6.5485400e-04, 6.5485400e-04, 3.4614600e-04,
           3.4614600e-04, 3.4614600e-04, 9.6396012e-05, 9.6396012e-05, 9.6396012e-05]
    ret = np.isclose(lr, ans)
    assert np.sum(ret == 0) / len(ret) < 0.05

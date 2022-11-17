# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Test mindsponge version."""
import pytest
import mindspore as ms
from mindsponge import _mindspore_version_check


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_version():
    """
    Feature: check mindspore version
    Description: None
    Expectation: ms_version >= required_mindspore_version
    """
    ms_version = ms.__version__[:5]
    required_mindspore_version = '1.8.1'
    print(ms_version)
    _mindspore_version_check()
    assert ms_version >= required_mindspore_version

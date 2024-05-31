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
import datetime
import pytest

from mindearth.utils import get_datapath_from_date

@pytest.mark.level0
@platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_get_datapath_from_date():
    """
    Feature: Test get_datapath_from_date in platform gpu and ascend.
    Description: The date_file_name and static_file_name are as expected.
    Expectation: Success or throw AssertionError.
    """
    date = datetime.datetime(2019, 1, 1, 0, 0, 0)
    idx = 1
    date_file_name, static_file_name = get_datapath_from_date(date, idx)
    assert date_file_name == "2019/2019_01_01_2.npy"
    assert static_file_name == "2019/2019.npy"

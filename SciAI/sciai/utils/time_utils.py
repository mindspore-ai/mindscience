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
# ==============================================================================
"""time utils"""
import time
from datetime import datetime, timezone


def time_second():
    """
    Get time in milliseconds number, e.g., 1678243339.780746.

    Returns:
        float, time in millisecond.
    """
    return time.time()


def time_str():
    """
    Get time in string representation, e.g., "2000-12-31-23-59-59".

    Returns:
        str, time in string representation.
    """
    return f"{datetime.now(tz=timezone.utc):%Y-%m-%d-%H-%M-%S}"

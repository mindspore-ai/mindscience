# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
pretty
"""


SEP = ">" * 30
LINE = "-" * 30


def time(seconds):
    """
    Format time as a string.

    Args:
        seconds (float): time in seconds
    """
    sec_per_min = 60
    sec_per_hour = 60 * 60
    sec_per_day = 24 * 60 * 60

    if seconds > sec_per_day:
        return f"{seconds / sec_per_day:.2f} days"
    if seconds > sec_per_hour:
        return f"{seconds / sec_per_hour:.2f} hours"
    if seconds > sec_per_min:
        return f"{seconds / sec_per_min:.2f} mins"
    return f"{seconds:.2f} secs"


def long_array(array, truncation=10, display=3):
    """
    Format an array as a string.

    Args:
        array (array_like): array-like data
        truncation (int, optional): truncate array if its length exceeds this threshold
        display (int, optional): number of elements to display at the beginning and the end in truncated mode

    Returns:
        array (str): formatted string.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    if len(array) <= truncation:
        return f"{array}"
    return f"{str(array[:display])[:-1]}, ..., {str(array[-display:])[1:]}"

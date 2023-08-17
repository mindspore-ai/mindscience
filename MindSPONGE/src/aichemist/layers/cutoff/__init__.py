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
Cutoff functions
"""

from .base import Cutoff
from .cosine import CosineCutoff
from .gaussian import GaussianCutoff
from .hard import HardCutoff
from .mollifier import MollifierCutoff
from .smooth import SmoothCutoff


__all__ = [
    'Cutoff',
    'CosineCutoff',
    'GaussianCutoff',
    'HardCutoff',
    'MollifierCutoff',
    'SmoothCutoff',
]

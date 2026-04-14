# Copyright 2022 Huawei Technologies Co., Ltd
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
"""init"""
from .dataset import create_training_dataset, create_test_dataset
from .utils import visual, calculate_l2_error
from .model import Burgers1D

__all__ = [
    "create_training_dataset",
    "visual",
    "calculate_l2_error",
    "create_test_dataset",
    "Burgers1D",
]

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
from .activation import Activation
from .gate import Gate
from .fc import FullyConnectedNet
from .normact import NormActivation
from .scatter import Scatter
from ..nn.one_hot import SoftOneHotLinspace, soft_one_hot_linspace, soft_unit_step, OneHot
from .batchnorm import BatchNorm

__all__ = [
    "Activation",
    "Gate",
    "FullyConnectedNet",
    "NormActivation",
    "Scatter",
    "SoftOneHotLinspace",
    "soft_one_hot_linspace",
    "soft_unit_step",
    "OneHot",
    "BatchNorm"
]
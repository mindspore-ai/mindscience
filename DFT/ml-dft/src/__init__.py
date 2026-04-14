# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
#
# Copyright 2022
# Georgia Tech Research Corporation
# Atlanta, Georgia 30332-4024
# All Rights Reserved
#
# This file is part of the ML-DFT originally developed by Georgia Tech Research Corporation (GTRC).
# Modifications have been made by Cen Jianhuan at Huawei Technologies Co., Ltd.
#
# This Program is licensed under the GENERAL PUBLIC USE LICENSE AGREEMENT provided by GTRC.
# You may not use this file except in compliance with the License.
#
# You should have received a copy of the GENERAL PUBLIC USE LICENSE AGREEMENT along with this program.
# If not, contact Georgia Tech Research Corporation for further information.
#
# Modifications made:
# - Replaced the neural network implementation originally based on Keras with an implementation using Huawei's MindSpore framework.
# - Completed training and inference on the Ascend NPU 910B.
#
# DISCLAIMER OF WARRANTIES AND LIMITATION ON LIABILITY:
# The Program is provided "AS IS", without warranty of any kind, express or implied, including but not limited to
# the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
#
# ============================================================================
"""init"""

from .dataset import *
from .utils import *
from .CHG import *
from .DOS import *
from .Energy import *

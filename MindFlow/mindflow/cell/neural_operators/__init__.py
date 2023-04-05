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
"""init"""
from .fno1d import FNO1D
from .fno2d import FNO2D
from .fno3d import FNO3D
from .kno1d import KNO1D
from .kno2d import KNO2D
from .pdenet import PDENet

__all__ = ["FNO1D", "FNO2D", "FNO3D", "KNO1D", "KNO2D", "PDENet"]

__all__.sort()

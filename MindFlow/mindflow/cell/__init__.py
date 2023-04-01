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
from .activation import get_activation
from .basic_block import LinearBlock, ResBlock, InputScale, FCSequential, MultiScaleFCSequential
from .neural_operators import FNO1D, FNO2D, KNO1D, KNO2D, PDENet
from .transformer import ViT
from .unet2d import UNet2D

__all__ = ['FNO1D', 'FNO2D', 'KNO1D', 'KNO2D', 'ViT', 'PDENet', 'UNet2D']
__all__.extend(activation.__all__)
__all__.extend(basic_block.__all__)

# Copyright 2025 Huawei Technologies Co., Ltd
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
Layers Package

This package contains various neural network layers and building blocks
that are used across different models in the MindScience toolkit.
It includes activation functions, basic blocks, and specialized layers
like UNet2D.
"""
from .activation import get_activation
from .basic_block import LinearBlock, ResBlock, InputScale, FCSequential, MultiScaleFCSequential, DropPath
from .unet2d import UNet2D
from .mask import MaskedLayerNorm

__all__ = ["get_activation", "LinearBlock", "ResBlock", "InputScale", "FCSequential",
           "MultiScaleFCSequential", "DropPath", "UNet2D", "MaskedLayerNorm"]

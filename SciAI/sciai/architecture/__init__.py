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
"""sciai architecture"""
from .activation import Swish, SReLU, get_activation, AdaptActivation
from .basic_block import MLP, MLPAAF, MLPShortcut, MSE, SSE, FirstOutputCell, NoArgNet, Normalize
from .neural_operators import FNO1D, FNO2D, FNO3D, KNO1D, KNO2D, PDENet
from .transformer import ViT

__all__ = []
__all__.extend(["Swish", "SReLU", "get_activation", "AdaptActivation"])
__all__.extend(["MLP", "MLPAAF", "MLPShortcut", "MSE", "SSE", "FirstOutputCell", "NoArgNet", "Normalize"])
__all__.extend(["FNO1D", "FNO2D", "FNO3D", "KNO1D", "KNO2D", "PDENet"])
__all__.extend(["ViT"])

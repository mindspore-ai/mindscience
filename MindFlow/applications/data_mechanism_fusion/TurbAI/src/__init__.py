# ============================================================================
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
from .build_feature import BuildFeature
from .callback import Callback2D, Callback3D
from .dataset import DatasetGenerator2D, DatasetGenerator3D, data_parallel_2d, \
    data_parallel_3d
from .loss import CustomWithLossCell2D, LossFunc2D, LossToEval2D, LossFunc3D, \
    LossToEval3D, CustomWithLossCell3D
from .lr_scheduler import StepLR
from .model import MLP, ResMLP
from .normalization import Normalization, get_mean_std_data_from_txt, get_min_max_data_from_txt
from .postprocess import PostProcess2DMinMax, PostProcess2DStd, PostProcess3DMinMax
from .read_data import get_datalist_from_txt, get_tensor_data
from .visualization import plt_error_distribute, plt_loss_func

__all__ = [
    "BuildFeature",
    "Callback2D",
    "Callback3D",
    "DatasetGenerator2D",
    "DatasetGenerator3D",
    "data_parallel_2d",
    "data_parallel_3d",
    "CustomWithLossCell2D",
    "LossFunc2D",
    "LossToEval2D",
    "LossFunc3D",
    "LossToEval3D",
    "CustomWithLossCell3D",
    "StepLR",
    "MLP",
    "ResMLP",
    "Normalization",
    "get_mean_std_data_from_txt",
    "get_min_max_data_from_txt",
    "PostProcess2DMinMax",
    "PostProcess2DStd",
    "PostProcess3DMinMax",
    "get_datalist_from_txt",
    "get_tensor_data",
    "plt_error_distribute",
    "plt_loss_func"
]

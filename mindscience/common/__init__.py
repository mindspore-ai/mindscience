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
"""init"""
from .lr_scheduler import get_poly_lr, get_multi_step_lr, get_warmup_cosine_annealing_lr
from .losses import get_loss_metric, WaveletTransformLoss, MTLWeightedLoss, RelativeRMSELoss
from .derivatives import batched_hessian, batched_jacobian
from .optimizers import AdaHessian
from .math import get_grid_1d, get_grid_2d, get_grid_3d
from .utils import to_2tuple, to_3tuple, unpatchify, patchify, get_2d_sin_cos_pos_embed, \
    pixel_shuffle, pixel_unshuffle, PixelShuffle, PixelUnshuffle, SpectralNorm
from .initializer import lecun_init, glorot_uniform

__all__ = ["get_poly_lr",
           "get_multi_step_lr",
           "get_warmup_cosine_annealing_lr",
           "get_loss_metric",
           "WaveletTransformLoss",
           "MTLWeightedLoss",
           "RelativeRMSELoss",
           "batched_hessian",
           "batched_jacobian",
           "AdaHessian",
           "get_grid_1d", "get_grid_2d", "get_grid_3d",
           "to_2tuple", "to_3tuple", "unpatchify", "patchify", "get_2d_sin_cos_pos_embed",
           "pixel_shuffle", "pixel_unshuffle", "PixelShuffle", "PixelUnshuffle",
           "SpectralNorm",
           "lecun_init", "glorot_uniform",
           ]

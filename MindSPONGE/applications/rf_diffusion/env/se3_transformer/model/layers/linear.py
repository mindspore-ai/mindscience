# Modified from se3-transformer-public (https://github.com/FabianFuchsML/se3-transformer-public)
# Original license: MIT License
#
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


from typing import Dict

import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from se3_transformer.model.fiber import Fiber


class LinearSE3(nn.Cell):
    """
    Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.
    Maps a fiber to a fiber with the same degrees (channels may be different).
    No interaction between degrees, but interaction between channels.

    type-0 features (C_0 channels) ────> Linear(bias=False) ────> type-0 features (C'_0 channels)
    type-1 features (C_1 channels) ────> Linear(bias=False) ────> type-1 features (C'_1 channels)
                                                 :
    type-k features (C_k channels) ────> Linear(bias=False) ────> type-k features (C'_k channels)
    """

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber):
        super().__init__()
        self.weights = {}
        for degree_out, channels_out in fiber_out:
            p = ms.Parameter(
                ms.mint.randn(channels_out, fiber_in[degree_out])
                / np.sqrt(fiber_in[degree_out]),
                name="weights_" + str(degree_out),
            )
            self.weights[str(degree_out)] = p
            self.insert_param_to_cell(p.name, p)

    def construct(
        self, features: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return {
            degree: self.weights[degree] @ features[degree]
            for degree, weight in self.weights.items()
        }

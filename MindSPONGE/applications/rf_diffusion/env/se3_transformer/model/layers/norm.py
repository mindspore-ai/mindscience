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
from mindspore import Tensor
from se3_transformer.model.fiber import Fiber


class NormSE3(nn.Cell):
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 ┌──> feature_norm ──> LayerNorm() ──> ReLU() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────┘
    """

    # Minimum positive subnormal for FP16
    NORM_CLAMP = 2**-24

    def __init__(self, fiber: Fiber, nonlinearity: nn.Cell = nn.ReLU()):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            # Fuse all the layer normalizations into a group normalization
            self.group_norm = nn.GroupNorm(
                num_groups=len(fiber.degrees), num_channels=sum(fiber.channels)
            )
        else:
            # Use multiple layer normalizations
            self.layer_norms = nn.CellDict(
                {
                    str(degree): nn.LayerNorm((channels,), epsilon=1e-5)
                    for degree, channels in fiber
                }
            )

    def construct(
        self, features: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        output = {}
        if hasattr(self, "group_norm"):
            # Compute per-degree norms of features
            norms = [
                features[str(d)].norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP)
                for d in self.fiber.degrees
            ]
            fused_norms = ms.mint.cat(norms, dim=-2)

            # Transform the norms only
            new_norms = self.nonlinearity(
                self.group_norm(fused_norms.squeeze(-1))
            ).unsqueeze(-1)
            new_norms = ms.mint.chunk(new_norms, chunks=len(self.fiber.degrees), dim=-2)

            # Scale features to the new norms
            for norm, new_norm, d in zip(norms, new_norms, self.fiber.degrees):
                output[str(d)] = features[str(d)] / norm * new_norm
        else:
            for degree, feat in features.items():
                norm = feat.norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP)
                new_norm = self.nonlinearity(
                    self.layer_norms[degree](norm.squeeze(-1)).unsqueeze(-1)
                )
                output[degree] = new_norm * feat / norm

        return output

# Modified from RFdiffusion (https://github.com/RosettaCommons/RFdiffusion)
# Original license: BSD License
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


import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Zero, initializer


class DistanceNetwork(nn.Cell):
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()

        self.proj_symm = nn.Linear(n_feat, 37 * 2)
        self.proj_asymm = nn.Linear(n_feat, 37 + 19)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer for final logit prediction
        self.proj_symm.weight.set_data(
            initializer(
                Zero(), self.proj_symm.weight.shape, self.proj_symm.weight.dtype
            )
        )
        self.proj_symm.bias.set_data(
            initializer(Zero(), self.proj_symm.bias.shape, self.proj_symm.bias.dtype)
        )
        self.proj_asymm.weight.set_data(
            initializer(
                Zero(), self.proj_asymm.weight.shape, self.proj_asymm.weight.dtype
            )
        )
        self.proj_asymm.bias.set_data(
            initializer(Zero(), self.proj_asymm.bias.shape, self.proj_asymm.bias.dtype)
        )

    def construct(self, x):
        # input: pair info (B, L, L, C)

        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:, :, :, :37].permute(0, 3, 1, 2)
        logits_phi = logits_asymm[:, :, :, 37:].permute(0, 3, 1, 2)

        # predict dist, omega
        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + logits_symm.permute(0, 2, 1, 3)
        logits_dist = logits_symm[:, :, :, :37].permute(0, 3, 1, 2)
        logits_omega = logits_symm[:, :, :, 37:].permute(0, 3, 1, 2)

        return logits_dist, logits_omega, logits_theta, logits_phi


class MaskedTokenNetwork(nn.Cell):
    def __init__(self, n_feat):
        super(MaskedTokenNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, 21)

        self.reset_parameter()

    def reset_parameter(self):
        self.proj.weight.set_data(
            initializer(Zero(), self.proj.weight.shape, self.proj.weight.dtype)
        )
        self.proj.bias.set_data(
            initializer(Zero(), self.proj.bias.shape, self.proj.bias.dtype)
        )

    def construct(self, x):
        B, N, L = x.shape[:3]
        logits = self.proj(x).permute(0, 3, 1, 2).reshape(B, -1, N * L)

        return logits


class LDDTNetwork(nn.Cell):
    def __init__(self, n_feat, n_bin_lddt=50):
        super(LDDTNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)

        self.reset_parameter()

    def reset_parameter(self):
        self.proj.weight.set_data(
            initializer(Zero(), self.proj.weight.shape, self.proj.weight.dtype)
        )
        self.proj.bias.set_data(
            initializer(Zero(), self.proj.bias.shape, self.proj.bias.dtype)
        )

    def construct(self, x):
        logits = self.proj(x)  # (B, L, 50)

        return logits.permute(0, 2, 1)


class ExpResolvedNetwork(nn.Cell):
    def __init__(self, d_msa, d_state, p_drop=0.1):
        super(ExpResolvedNetwork, self).__init__()
        self.norm_msa = nn.LayerNorm((d_msa,), epsilon=1e-5)
        self.norm_state = nn.LayerNorm((d_state,), epsilon=1e-5)
        self.proj = nn.Linear(d_msa + d_state, 1)

        self.reset_parameter()

    def reset_parameter(self):
        self.proj.weight.set_data(
            initializer(Zero(), self.proj.weight.shape, self.proj.weight.dtype)
        )
        self.proj.bias.set_data(
            initializer(Zero(), self.proj.bias.shape, self.proj.bias.dtype)
        )

    def construct(self, seq, state):
        B, L = seq.shape[:2]

        seq = self.norm_msa(seq)
        state = self.norm_state(state)
        feat = ms.mint.cat((seq, state), dim=-1)
        logits = self.proj(feat)
        return logits.reshape(B, L)

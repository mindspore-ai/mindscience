# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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

"""target feat embedding"""

import mindspore as ms
from mindspore import nn, mint
from protenix.model.modules.atom_cross_attention import AtomCrossAttEncoder


class CreateTargetFeatEmbedding(nn.Cell):
    """
    A class that creates target feature embeddings by combining raw features with cross-attention encoded features.

    Args:
        config (Config): Configuration object containing parameters for the target feature embedding.
        global_config (GlobalConfig): Global configuration object.

    Inputs:
        - **batch** (dict) - Dictionary containing batch features.

    Outputs:
        - **target_feat** (Tensor) - Tensor of target feature embeddings.
    """
    def __init__(self, config, global_config, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config  = global_config
        self.dtype = dtype
        self.atom_cross_att_encoder = AtomCrossAttEncoder(
            self.config.per_atom_conditioning, self.global_config, with_cond=False, dtype=dtype
        )
    def construct(self, batch):
        """construct"""
        enc = self.atom_cross_att_encoder(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )
        batch_shape = batch.ref_structure.restype.shape[:-1]
        target_feat = mint.cat(
            [enc.token_act.astype(self.dtype)]
            + [
                batch.ref_structure.restype.reshape(*batch_shape, -1),
                batch.ref_structure.profile.reshape(*batch_shape, -1),
                batch.ref_structure.deletion_mean.reshape(*batch_shape, -1),
            ],
            dim=-1,
        )
        return target_feat

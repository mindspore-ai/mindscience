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

"""diffusion"""

from dataclasses import dataclass
import math
import numpy as np
import mindspore as ms
from mindspore import nn, mint
from protenix.model.modules.module_utils.common import residue_names
from protenix.model import base_config
from protenix.model.modules import base_modules as bm
from protenix.model.modules.atom_cross_attention import AtomCrossAttEncoder, AtomCrossAttEncoderConfig
from protenix.model.modules.transformer import AtomCrossAttDecoder, AtomCrossAttDecoderConfig
from protenix.model.modules import transformer as diffusion_transformer
from protenix.model.modules import featurization

SIGMA_DATA = 16.0


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
    noise_level = (
        SIGMA_DATA
        * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
    )
    noise_level[-1] = 0.0
    return noise_level


@dataclass
class ConditioningConfig(base_config.BaseConfig):
    pair_channel: int
    seq_channel: int
    prob: float


@dataclass
class SampleConfig(base_config.BaseConfig):
    steps: int
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    num_samples: int = 1


class DiffusionHead(nn.Cell):
    """Denoising Diffusion Head.

    Args:
        config (Config): Configuration object containing parameters for the diffusion head.
        global_config (GlobalConfig): Global configuration object containing shared parameters.
        in_shape (tuple): Input shape for the module.
        max_relative_chain (int): Maximum number of relative chains for positional encoding. Default: ``2``.
        max_relative_idx (int): Maximum relative index for positional encoding. Default: ``32``.

    Inputs:
        - **positions_noisy** (Tensor) - Noisy atomic positions tensor.
        - **noise_level** (Tensor) - Tensor representing the noise level.
        - **batch** (Batch) - Batch of input data containing token features and structure information.
        - **embeddings** (dict) - Dictionary of embeddings for single and pair features.
        - **use_conditioning** (bool) - Flag to enable or disable conditioning.

    Outputs:
        - **position_update** (Tensor) - Refined atomic positions tensor.
    """

    class Config(
        AtomCrossAttEncoderConfig,
        AtomCrossAttDecoderConfig,
    ):
        """Configuration for DiffusionHead."""
        eval_batch_size: int = 5
        eval_batch_dim_shard_size: int = 5
        conditioning: ConditioningConfig = base_config.autocreate(
            prob=0.8, pair_channel=128, seq_channel=384
        )
        eval: SampleConfig = base_config.autocreate(
            num_samples=5,
            steps=200,
        )
        train: SampleConfig = base_config.autocreate(
            num_samples=1,
            steps=1,
        )
        train_mini: SampleConfig = base_config.autocreate(
            num_samples=1,
            steps=20,
        )
        transformer: diffusion_transformer.Transformer.Config = (
            base_config.autocreate()
        )

    def __init__(self, config, global_config, in_shape, max_relative_chain=2,
                 max_relative_idx=32, ndim=3, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.dtype = dtype
        in_channel = in_shape[-1]
        self.max_relative_chain = max_relative_chain
        self.max_relative_idx = max_relative_idx

        # _conditioning modules
        in_channel_pair = in_channel * 2
        self.pair_cond_initial_norm = bm.LayerNorm(
            in_shape[:-1] + (in_channel_pair,),
            create_beta=False, gamma_init="ones",
            name='pair_cond_initial_norm', dtype=dtype)
        self.pair_cond_initial_projection = bm.LinearAMP(
            in_channel_pair, self.config.conditioning.pair_channel, has_bias=False, dtype=ms.float32)
        self.transition_block1 = diffusion_transformer.TransitionBlock(
            in_channel, 2, with_single_cond=False, dtype=dtype)
        self.transition_block2 = diffusion_transformer.TransitionBlock(
            in_channel, 2, with_single_cond=False, dtype=dtype)
        in_channel_single = self.config.conditioning.seq_channel * 2 \
            + residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP * 2 + 3  # 1
        self.single_cond_initial_norm = bm.LayerNorm(
            in_shape[:-1] + (in_channel_single,),
            create_beta=False, gamma_init="ones",
            name='single_cond_initial_norm', dtype=dtype)
        self.single_cond_initial_projection = bm.LinearAMP(
            in_channel_single, self.config.conditioning.seq_channel, has_bias=False, dtype=dtype)
        self.num_noise_embedding = 256
        self.layer_norm_noise = bm.LayerNorm(
            in_shape[:-1]+(self.num_noise_embedding,),
            create_beta=False, gamma_init="ones",
            name='noise_embedding_initial_norm', dtype=dtype)
        self.linear_noise = bm.LinearAMP(self.num_noise_embedding,
                                         self.config.conditioning.seq_channel, has_bias=False, dtype=dtype)
        self.single_transition1 = diffusion_transformer.TransitionBlock(
            self.config.conditioning.seq_channel, 2, ndim=2, with_single_cond=False, dtype=dtype)
        self.single_transition2 = diffusion_transformer.TransitionBlock(
            self.config.conditioning.seq_channel, 2, ndim=2, with_single_cond=False, dtype=dtype)

        # modules
        self.layer_norm_act = bm.LayerNorm(
            (in_channel,)+(self.config.conditioning.seq_channel,),
            create_beta=False, gamma_init="ones",
            name='single_cond_embedding_norm', dtype=dtype)
        self.linear_act = bm.LinearAMP(self.config.conditioning.seq_channel,
                                       self.config.per_token_channels, has_bias=False, dtype=dtype)
        self.layer_norm_out = bm.LayerNorm(
            in_shape[:-1]+(self.config.per_token_channels,),
            create_beta=False, gamma_init="ones",
            name='output_norm', dtype=dtype)
        self.atom_cross_att_encoder = AtomCrossAttEncoder(
            self.config, self.global_config, dtype=dtype
        )
        self.transformer = diffusion_transformer.Transformer(
            self.config.transformer, self.global_config, in_shape[:-1] + (
                self.config.conditioning.seq_channel * 2,),
            in_shape, ndim=ndim, using_pair_act=True, dtype=dtype
        )
        self.atom_cross_att_decoder = AtomCrossAttDecoder(
            self.config, self.global_config, dtype=dtype
        )

        # add for protenix
        self.linear_rel_features = bm.CustomDense(
            4 * self.max_relative_idx + 4 + 2 * self.max_relative_chain + 2 + 1,
            self.config.conditioning.pair_channel, ndim=3)
        self.weight = ms.Parameter(np.random.random(256).astype(np.float32))
        self.bias = ms.Parameter(np.random.random(256).astype(np.float32))

    def _conditioning(self, batch, embeddings, noise_level, use_conditioning):
        """_conditioning"""
        single_embedding = use_conditioning * embeddings['single']
        pair_embedding = use_conditioning * embeddings['pair']
        rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=self.max_relative_idx, max_relative_chain=self.max_relative_chain
        ).astype(pair_embedding.dtype)
        features_2d = mint.concat(
            [pair_embedding, self.linear_rel_features(rel_features)], dim=-1)
        pair_cond = self.pair_cond_initial_projection(self.pair_cond_initial_norm(
            features_2d.astype(ms.float32))).astype(pair_embedding.dtype)  # (256,256,267) -> (256,256,128)
        pair_cond = pair_cond + self.transition_block1(pair_cond)
        pair_cond = pair_cond + self.transition_block2(pair_cond)

        target_feat = embeddings['target_feat']
        features_1d = mint.concat(
            [single_embedding, target_feat.astype(single_embedding.dtype)], dim=-1)
        single_cond = self.single_cond_initial_norm(features_1d)
        single_cond = self.single_cond_initial_projection(single_cond)
        x = (1 / 4) * mint.log(noise_level / SIGMA_DATA).astype(self.dtype)
        noise_embedding = mint.cos(
            2 * math.pi * (x[..., None] * self.weight + self.bias))
        single_cond = single_cond.unsqueeze(-3) + self.linear_noise(
            self.layer_norm_noise(noise_embedding)).unsqueeze(-2)  # (1,256) -> (1,384)
        single_cond = single_cond + self.single_transition1(single_cond)
        single_cond = single_cond + self.single_transition2(single_cond)

        return single_cond, pair_cond

    def construct(self, positions_noisy, noise_level, batch, embeddings_pair,
                  embeddings_single, embeddings_target, use_conditioning):
        """construct"""
        embeddings = {
            'pair': embeddings_pair,
            'single': embeddings_single,
            'target_feat': embeddings_target,
        }
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # Position features
        act = positions_noisy / \
            mint.sqrt(noise_level**2 + SIGMA_DATA**2)[..., None, None]
        enc = self.atom_cross_att_encoder(
            act, embeddings["single"], trunk_pair_cond.astype(ms.float32), batch)

        act = enc.token_act
        act = act + self.linear_act(self.layer_norm_act(trunk_single_cond))
        act = act.astype(ms.float32)
        trunk_single_cond = trunk_single_cond.astype(ms.float32)
        trunk_pair_cond = trunk_pair_cond.astype(ms.float32)
        act = self.transformer(act, trunk_single_cond, trunk_pair_cond)
        act = self.layer_norm_out(act)
        position_update = self.atom_cross_att_decoder(act, enc, batch)
        skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
        out_scaling = (
            noise_level * SIGMA_DATA /
            mint.sqrt(noise_level**2 + SIGMA_DATA**2)
        )
        out = mint.mul(skip_scaling[:, None, None], positions_noisy)
        out = out + out_scaling[:, None, None] * position_update
        return out

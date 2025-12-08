# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend

"""Diffusion Head."""

from dataclasses import dataclass
from collections.abc import Callable
import math
import numpy as np
import mindspore as ms
from mindspore import mint, nn
from alphafold3.constants import residue_names
from alphafold3.model import base_config
from alphafold3.model.components import base_modules as bm
from alphafold3.model.components import utils
from alphafold3.model.diffusion import atom_cross_attention
from alphafold3.model.diffusion import diffusion_transformer
from alphafold3.model.diffusion import featurization


# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0
WEIGHT = ms.Tensor(np.load("./alphafold3/model/diffusion/random/weight.npy"), dtype=ms.float32)
BIAS = ms.Tensor(np.load("./alphafold3/model/diffusion/random/bias.npy"), dtype=ms.float32)

def fourier_embeddings(x):
    """Compute Fourier embeddings."""
    return mint.cos(2 * math.pi * (x[..., None] * WEIGHT + BIAS))

def random_rotation(key):
    """Generate random rotation matrix."""
    # Create a random rotation (Gram-Schmidt orthogonalization of two
    # random normal vectors)
    np.random.seed(key)
    v0, v1 = ms.Tensor(np.random.normal(0, 1, (2, 3)), dtype=ms.float32)
    e0 = v0 / mint.maximum(1e-10, mint.norm(v0))
    v1 = v1 - e0 * mint.matmul(v1, e0)
    e1 = v1 / mint.maximum(1e-10, mint.norm(v1))
    e2 = mint.cross(e0, e1)
    return mint.stack([e0, e1, e2])

def random_augmentation(rng_key, positions, mask):
    """Apply random rigid augmentation.
    Args:
        rng_key: random key
        positions: atom positions of shape (<common_axes>, 3)
        mask: per-atom mask of shape (<common_axes>,)
    Returns:
        Transformed positions with the same shape as input positions.
    """
    center = utils.mask_mean(
        mask.unsqueeze(-1), positions, axis=(-2, -3), keepdims=True, eps=1e-6
    ).astype(ms.float32)
    rot = random_rotation(rng_key)
    np.random.seed(rng_key)
    translation = ms.Tensor(np.random.normal(0, 1, (3,)), dtype=ms.float32)

    augmented_positions = (
        mint.einsum(
            '...i,ij->...j',
            (positions - center).astype(ms.float32),
            rot,
        )
        + translation
    )
    return augmented_positions * mask[..., None]

def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
    """Compute noise schedule."""
    return (
        SIGMA_DATA
        * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
    )

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
            atom_cross_attention.AtomCrossAttEncoderConfig,
            atom_cross_attention.AtomCrossAttDecoderConfig,
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
        transformer: diffusion_transformer.Transformer.Config = (
            base_config.autocreate()
        )

    def __init__(self, config, global_config, in_shape, max_relative_chain=2, max_relative_idx=32, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.dtype = dtype
        in_channel = in_shape[-1]
        self.max_relative_chain = max_relative_chain
        self.max_relative_idx = max_relative_idx

        # _conditioning modules
        in_channel_pair = in_channel + 4 * self.max_relative_idx + 4 + 2 * self.max_relative_chain + 2 + 1
        self.pair_cond_initial_norm = bm.LayerNorm(
            in_shape[:-1] + (in_channel_pair,),
            create_beta=False, gamma_init="ones",
            name='pair_cond_initial_norm', dtype=dtype)
        self.pair_cond_initial_projection = nn.Dense(in_channel_pair, self.config.conditioning.pair_channel,
                                                     has_bias=False, dtype=ms.float32)
        self.transition_block1 = diffusion_transformer.TransitionBlock(
            in_channel, 2, with_single_cond=False,
            dtype=dtype
        )
        self.transition_block2 = diffusion_transformer.TransitionBlock(
            in_channel, 2, with_single_cond=False,
            dtype=dtype
        )
        in_channel_single = self.config.conditioning.seq_channel * 2 \
            + residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP * 2 + 1
        self.single_cond_initial_norm = bm.LayerNorm(
            in_shape[:-1] + (in_channel_single,),
            create_beta=False, gamma_init="ones",
            name='single_cond_initial_norm', dtype=dtype)
        self.single_cond_initial_projection = nn.Dense(in_channel_single, self.config.conditioning.seq_channel,
                                                       has_bias=False, dtype=dtype)
        self.num_noise_embedding = 256
        self.layer_norm_noise = bm.LayerNorm(
            in_shape[:-1]+(self.num_noise_embedding,),
            create_beta=False, gamma_init="ones",
            name='noise_embedding_initial_norm', dtype=dtype)
        self.linear_noise = nn.Dense(self.num_noise_embedding, self.config.conditioning.seq_channel,
                                     has_bias=False, dtype=dtype)
        self.single_transition1 = diffusion_transformer.TransitionBlock(
            self.config.conditioning.seq_channel, 2,
            ndim=2, with_single_cond=False,
            dtype=dtype
        )
        self.single_transition2 = diffusion_transformer.TransitionBlock(
            self.config.conditioning.seq_channel, 2,
            ndim=2, with_single_cond=False,
            dtype=dtype
        )

        # modules
        self.layer_norm_act = bm.LayerNorm(
            (in_channel,)+(self.config.conditioning.seq_channel,),
            create_beta=False, gamma_init="ones",
            name='single_cond_embedding_norm', dtype=dtype)
        self.linear_act = nn.Dense(self.config.conditioning.seq_channel,
                                   self.config.per_token_channels, has_bias=False, dtype=dtype)
        self.layer_norm_out = bm.LayerNorm(
            in_shape[:-1]+(self.config.per_token_channels,),
            create_beta=False, gamma_init="ones",
            name='output_norm', dtype=dtype)
        self.atom_cross_att_encoder = atom_cross_attention.AtomCrossAttEncoder(
            self.config, self.global_config, "", dtype=dtype
        )
        self.transformer = diffusion_transformer.Transformer(
            self.config.transformer, self.global_config, in_shape[:-1] + (self.config.conditioning.seq_channel * 2,),
            in_shape, using_pair_act=True, dtype=dtype
        )
        self.atom_cross_att_decoder = atom_cross_attention.AtomCrossAttDecoder(
            self.config, self.global_config, '', dtype=dtype
        )

    def _conditioning(self, batch, embeddings, noise_level, use_conditioning):
        """conditioning"""
        single_embedding = use_conditioning * embeddings['single']
        pair_embedding = use_conditioning * embeddings['pair']
        rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=self.max_relative_idx, max_relative_chain=self.max_relative_chain
        ).astype(pair_embedding.dtype)
        features_2d = mint.concat([pair_embedding, rel_features], dim=-1)
        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )
        pair_cond += self.transition_block1(pair_cond)
        pair_cond += self.transition_block2(pair_cond)

        target_feat = embeddings['target_feat']
        features_1d = mint.concat([single_embedding, target_feat], dim=-1)
        single_cond = self.single_cond_initial_norm(features_1d)
        single_cond = self.single_cond_initial_projection(single_cond)
        noise_embedding = fourier_embeddings(
            (1 / 4) * mint.log(noise_level / SIGMA_DATA)
        )
        single_cond += self.linear_noise(self.layer_norm_noise(noise_embedding))
        single_cond += self.single_transition1(single_cond)
        single_cond += self.single_transition2(single_cond)

        return single_cond, pair_cond

    def construct(self, positions_noisy, noise_level, batch, embeddings, use_conditioning):
        """diffusion head"""
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # Extract features
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask
        # Position features
        act = positions_noisy * atom_mask[..., None]
        act = act / mint.sqrt(noise_level**2 + SIGMA_DATA**2)
        enc = self.atom_cross_att_encoder(act, embeddings["single"], trunk_pair_cond, batch)

        act = enc.token_act
        act += self.linear_act(self.layer_norm_act(trunk_single_cond))
        act = self.transformer(act, trunk_single_cond, sequence_mask, trunk_pair_cond)
        act = self.layer_norm_out(act)
        position_update = self.atom_cross_att_decoder(act, enc, batch)
        skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
        out_scaling = (
            noise_level * SIGMA_DATA / mint.sqrt(noise_level**2 + SIGMA_DATA**2)
        )
        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask[..., None]

def sample(denoising_step, batch, key, config, init_positions=None):
    """Sample using denoiser on batch.

    Args:
        denoising_step: the denoising function.
        batch: the batch
        key: random key
        config: config for the sampling process (e.g. number of denoising steps,
        etc.)

    Returns:
        a dict
            {
                'atom_positions': ms.Tensor       # shape (<common_axes>, 3)
                'mask': ms.Tensor                 # shape (<common_axes>,)
            }
        where the <common_axes> are
        (num_samples, num_tokens, max_atoms_per_token)
    """

    mask = batch.predicted_structure_info.atom_mask
    # get weight and bias from Jax, this two values cannot be randomly generated

    def apply_denoising_step(carry, noise_level):
        key, positions, noise_level_prev = carry

        positions = random_augmentation(
            rng_key=key, positions=positions, mask=mask,
        )
        gamma = config.gamma_0 * (noise_level > config.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = config.noise_scale * mint.sqrt(t_hat**2 - noise_level_prev**2)
        np.random.seed(key)
        noise = noise_scale * ms.Tensor(np.random.normal(0, 1, positions.shape), dtype=ms.float32)
        positions_noisy = positions + noise

        positions_denoised = denoising_step(positions_noisy, t_hat)
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + config.step_scale * d_t * grad

        return (key, positions_out, noise_level), positions_out

    num_samples = config.num_samples

    noise_levels = noise_schedule(mint.linspace(0, 1, config.steps + 1))

    noise_key, key = key, key + 1
    np.random.seed(noise_key)
    if init_positions is None:
        init_positions = ms.Tensor(np.random.normal(0, 1, (num_samples,) + mask.shape + (3,)), dtype=ms.float32)
    init_positions *= noise_levels[0]
    init = (ms.Tensor([key + i for i in range(num_samples)]).reshape((-1, 1)),
            init_positions,
            mint.tile(noise_levels[None, 0], (num_samples,)).reshape((-1, 1)))
    count = 0
    for noise_level in noise_levels[1:]:
        for i in range(num_samples):
            temp, _ = apply_denoising_step((count * 10 + i, init[1][i], init[2][i]), noise_level)
            init[0][i], init[1][i], init[2][i] = temp
        count += 1
    _, positions_out, _ = init

    final_dense_atom_mask = mint.tile(mask[None], (num_samples, 1, 1))

    return {'atom_positions': positions_out, 'mask': final_dense_atom_mask}

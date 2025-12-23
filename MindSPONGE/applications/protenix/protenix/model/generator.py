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

"""generator"""

from typing import Any, Callable, Optional
import numbers
from collections import abc
import numpy as np
from scipy.spatial.transform import Rotation
import mindspore as ms
from mindspore import mint
from mindspore import _no_grad


def mask_mean(mask, value, axis=None, keepdims=False, eps=1e-10):
    """Masked mean."""

    mask_shape = mask.shape
    value_shape = value.shape

    if len(mask_shape) != len(
        value_shape
    ):
        raise ValueError(f"Shapes are not compatible, shapes: {mask_shape}, {value_shape}")

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    if not isinstance(
        axis, abc.Iterable
    ):
        raise TypeError("axis needs to be either an iterable, integer or None")

    broadcast_factor = 1.0
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            error = f'Shapes are not compatible, shapes: {mask_shape}, {value_shape}'
            if mask_size != value_size:
                raise ValueError(error)
    return mint.sum(mask * value, keepdim=keepdims, dim=axis) / (
        mint.maximum(
            mint.sum(mask, keepdim=keepdims, dim=axis) *
            broadcast_factor, eps
        )
    )


def random_rotation(n_sample):
    rotation = Rotation.random(num=n_sample, random_state=1234)
    rot_matrix = ms.Tensor(rotation.as_matrix()).float()  # [n_sample, 3, 3]
    return rot_matrix


def random_augmentation(positions, mask=None):
    """Apply random rigid augmentation.
    Args:
        positions: atom positions of shape (<common_axes>, 3)
        mask: per-atom mask of shape (<common_axes>,)
    Returns:
        Transformed positions with the same shape as input positions.
    """
    if mask is None:
        center = mint.mean(
            positions, dim=-2, keepdim=True
        )
    else:
        center = mask_mean(
            mask.unsqueeze(-1), positions, axis=(-2, -3), keepdims=True, eps=1e-6
        ).astype(ms.float32)
    rot = random_rotation(positions.shape[-3])
    translation = ms.Tensor(np.random.randn(3).astype(np.float32))

    augmented_positions = (
        rot_vec_mul(
            r=rot, t=(positions - center).astype(ms.float32)
        ) + translation
    )
    if mask is not None:
        augmented_positions = augmented_positions * mask[..., None]
    return augmented_positions


def rot_vec_mul(r: ms.Tensor, t: ms.Tensor) -> ms.Tensor:
    """Apply rot matrix to vector
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r (ms.Tensor): the rotation matrices
            [..., 3, 3]
        t (ms.Tensor): the coordinate tensors
            [..., 3]

    Returns:
        ms.Tensor: the rotated coordinates
    """
    r = r.unsqueeze(-1)
    x, y, z = mint.unbind(input=t, dim=-1)
    return mint.stack(
        tensors=[
            r[..., 0, 0, :] * x + r[..., 0, 1, :] * y + r[..., 0, 2, :] * z,
            r[..., 1, 0, :] * x + r[..., 1, 1, :] * y + r[..., 1, 2, :] * z,
            r[..., 2, 0, :] * x + r[..., 2, 1, :] * y + r[..., 2, 2, :] * z,
        ],
        dim=-1,
    )


class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(
        self, size
    ) -> ms.Tensor:
        """Sampling

        Args:
            size (tuple): the target size

        Returns:
            ms.Tensor: sampled noise-level
        """
        rnd_normal = ms.Tensor(np.random.randn(*size).astype(np.float32))
        noise_level = (rnd_normal * self.p_std +
                       self.p_mean).exp() * self.sigma_data
        return noise_level


@_no_grad()
def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    embeddings: ms.Tensor,
    noise_schedule: ms.Tensor,
    n_sample: int = 5,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,  # pylint: disable=unused-argument
) -> ms.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (ms.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, feature_input]
        s_trunk (ms.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, single_channel]
        z_trunk (ms.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, pair_channel]
        noise_schedule (ms.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        n_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.

    Returns:
        ms.Tensor: the denoised coordinates of x in inference stage
            [..., n_sample, n_atom, 3]
    """
    n_atom = input_feature_dict.atom_cross_att.token_atoms_to_queries.shape[0]
    batch_shape = embeddings['single'].shape[:-2]

    def _chunk_sample_diffusion(chunk_n_sample):
        x_l = noise_schedule[0] * ms.Tensor(np.random.randn(
            *batch_shape, chunk_n_sample, n_atom, 3).astype(np.float32
                                                            ))
        for _, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., n_sample, n_atom, 3]
            x_l = (
                random_augmentation(x_l)
                .squeeze(axis=-3)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = ms.ops.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * ms.Tensor(np.random.randn(
                *x_l.shape).astype(np.float32))

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
            )
            x_denoised = denoise_net(
                positions_noisy=x_noisy,
                noise_level=t_hat,
                batch=input_feature_dict,
                embeddings_single=embeddings['single'],
                embeddings_pair=embeddings['pair'],
                embeddings_target=embeddings['target_feat'],
                use_conditioning=True
            )

            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
        return x_l.unsqueeze(0)

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(n_sample)
    else:
        x_l = []
        no_chunks = n_sample // diffusion_chunk_size + (
            n_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else n_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample
            )
            x_l.append(chunk_x_l)
        x_l = ms.ops.cat(x_l, -3)  # [..., n_sample, n_atom, 3]

    final_dense_atom_mask = None
    return {'atom_positions': x_l, 'mask': final_dense_atom_mask}


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    embeddings: ms.Tensor,
    n_sample: int = 32,
    diffusion_chunk_size: Optional[int] = None,
):
    """Sample diffusion for training."""
    batch_size_shape = label_dict["coordinate"].shape[:-3]
    coordinate = label_dict["coordinate"].unsqueeze(0)
    coordinate = coordinate.tile((n_sample, 1, 1))
    x_gt_augment = (
        random_augmentation(coordinate)
        .squeeze(axis=-3)
    )
    sigma = noise_sampler(size=(*batch_size_shape, n_sample)).squeeze()
    noise = ms.Tensor(np.random.randn(*(x_gt_augment.shape)
                                      ).astype(np.float32)) * sigma[..., None, None]

    diffusion_chunk_size = 4

    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            positions_noisy=x_gt_augment + noise,
            noise_level=sigma,
            batch=input_feature_dict,
            embeddings_single=embeddings['single'],
            embeddings_pair=embeddings['pair'],
            embeddings_target=embeddings['target_feat'],
            use_conditioning=True
        )
    else:
        x_denoised = []
        no_chunks = n_sample // diffusion_chunk_size + (
            n_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size: (i + 1) * diffusion_chunk_size, :, :
            ]
            x_denoised_i = denoise_net(
                positions_noisy=x_noisy_i,
                noise_level=sigma[
                    ..., i * diffusion_chunk_size: (i + 1) * diffusion_chunk_size
                ],
                batch=input_feature_dict,
                embeddings_single=embeddings['single'],
                embeddings_pair=embeddings['pair'],
                embeddings_target=embeddings['target_feat'],
                use_conditioning=True
            )
            x_denoised.append(x_denoised_i)
        x_denoised = mint.cat(x_denoised, dim=-3)
    final_dense_atom_mask = None

    return x_gt_augment, x_denoised, sigma, final_dense_atom_mask

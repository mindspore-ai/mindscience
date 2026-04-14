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
"Latent Diffusion Model"
import warnings
from typing import Sequence, Dict, Any, Callable
from copy import deepcopy
from functools import partial
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, mint

from src.utils import (
    DiagonalGaussianDistribution,
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    default,
    parse_layout_shape,
    disabled_train,
    layout_to_in_out_slice,
    calculate_ssim,
    SEVIRSkillScore,
)
from src.sevir_dataset import SEVIRDataModule
from src.vae import AutoencoderKL
from src.knowledge_alignment.alignment_net import AvgIntensityAlignment
from .cuboid_transformer_unet import CuboidTransformerUNet


class LatentDiffusion(nn.Cell):
    """
    Base class for latent space diffusion models. Implements core diffusion processes including
    noise scheduling, model application, loss calculation, and latent space operations. Integrates
    main UNet model, VAE, and conditioning modules with support for temporal alignment.
    """

    def __init__(
            self,
            main_model: nn.Cell,
            layout: str = "NTHWC",
            data_shape: Sequence[int] = (10, 128, 128, 4),
            timesteps=1000,
            beta_schedule="linear",
            loss_type="l2",
            monitor="val/loss",
            log_every_t=100,
            clip_denoised=False,
            linear_start=1e-4,
            linear_end=2e-2,
            cosine_s=8e-3,
            given_betas=None,
            original_elbo_weight=0.0,
            v_posterior=0.0,
            l_simple_weight=1.0,
            learn_logvar=False,
            logvar_init=0.0,
            latent_shape: Sequence[int] = (10, 16, 16, 4),
            first_stage_model: nn.Cell = None,
            cond_stage_forward=None,
            scale_by_std=False,
            scale_factor=1.0,
    ):
        super().__init__()

        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.main_model = main_model
        self.layout = layout
        self.data_shape = data_shape
        self.parse_layout_shape(layout=layout)
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        logvar = ops.full(fill_value=logvar_init, size=(self.num_timesteps,)).astype(
            ms.float32
        )
        if self.learn_logvar:
            self.logvar = Parameter(logvar, requires_grad=True)
        else:
            self.logvar = Parameter(logvar, name="logvar", requires_grad=False)

        self.latent_shape = latent_shape
        self.scale_by_std = scale_by_std
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.logvar = Parameter(
                scale_factor, name="scale_factor", requires_grad=False
            )

        self.instantiate_first_stage(first_stage_model)
        self.instantiate_cond_stage(cond_stage_forward)

    def set_alignment(self, alignment_fn: Callable = None):
        """
        Sets alignment function for denoising process after initialization.
        Args:
            alignment_fn (Callable): Alignment function with signature
                `alignment_fn(zt, t, zc=None, y=None, **kwargs)`
        """
        self.alignment_fn = alignment_fn

    def parse_layout_shape(self, layout):
        """
        Parses data layout string to determine axis indices.
        Args:
            layout (str): Data layout specification (e.g., 'NTHWC')
        """
        parsed_dict = parse_layout_shape(layout=layout)
        self.batch_axis = parsed_dict["batch_axis"]
        self.t_axis = parsed_dict["t_axis"]
        self.h_axis = parsed_dict["h_axis"]
        self.w_axis = parsed_dict["w_axis"]
        self.c_axis = parsed_dict["c_axis"]
        self.all_slice = [
            slice(None, None),
        ] * len(layout)

    def extract_into_tensor(self, a, t, x_shape):
        """Extracts schedule parameters into tensor format for current batch."""
        return extract_into_tensor(
            a=a, t=t, x_shape=x_shape, batch_axis=self.batch_axis
        )

    @property
    def loss_mean_dim(self):
        """Computes mean dimensions for loss calculation excluding batch axis."""
        if not hasattr(self, "loss_m_dim"):
            loss_m_dim = list(range(len(self.layout)))
            loss_m_dim.pop(self.batch_axis)
            self.loss_m_dim = tuple(loss_m_dim)
        return self.loss_m_dim

    def get_batch_latent_shape(self, batch_size=1):
        """
        Generates latent shape with specified batch size.
        Args:
            batch_size (int): Desired batch size
        """
        batch_latent_shape = deepcopy(list(self.latent_shape))
        batch_latent_shape.insert(self.batch_axis, batch_size)
        self.batch_latent_shape = tuple(batch_latent_shape)
        return self.batch_latent_shape

    def register_schedule(
            self,
            given_betas=None,
            beta_schedule="linear",
            timesteps=1000,
            linear_start=1e-4,
            linear_end=2e-2,
            cosine_s=8e-3,
    ):
        """
        Registers diffusion schedule parameters and precomputes necessary tensors.
        Args:
            given_betas (Tensor): Custom beta values
            beta_schedule (str): Schedule type ('linear', 'cosine')
            timesteps (int): Number of diffusion steps
            linear_start (float): Linear schedule start value
            linear_end (float): Linear schedule end value
            cosine_s (float): Cosine schedule parameter
        """
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_mindspore = partial(Tensor, dtype=ms.float32)
        self.betas = Parameter(to_mindspore(betas), name="betas", requires_grad=False)
        self.alphas_cumprod = Parameter(
            to_mindspore(alphas_cumprod), name="alphas_cumprod", requires_grad=False
        )
        self.alphas_cumprod_prev = Parameter(
            to_mindspore(alphas_cumprod_prev),
            name="alphas_cumprod_prev",
            requires_grad=False,
        )
        self.sqrt_alphas_cumprod = Parameter(
            to_mindspore(np.sqrt(alphas_cumprod)),
            name="sqrt_alphas_cumprod",
            requires_grad=False,
        )
        self.sqrt_one_minus_alphas_cumprod = Parameter(
            to_mindspore(np.sqrt(1.0 - alphas_cumprod)),
            name="sqrt_one_minus_alphas_cumprod",
            requires_grad=False,
        )
        self.log_one_minus_alphas_cumprod = Parameter(
            to_mindspore(np.log(1.0 - alphas_cumprod)),
            name="log_one_minus_alphas_cumprod",
            requires_grad=False,
        )
        self.sqrt_recip_alphas_cumprod = Parameter(
            to_mindspore(np.sqrt(1.0 / alphas_cumprod)),
            name="sqrt_recip_alphas_cumprod",
            requires_grad=False,
        )
        self.sqrt_recipm1_alphas_cumprod = Parameter(
            to_mindspore(np.sqrt(1.0 / alphas_cumprod - 1)),
            name="sqrt_recipm1_alphas_cumprod",
            requires_grad=False,
        )

        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        self.posterior_variance = Parameter(
            to_mindspore(posterior_variance),
            name="posterior_variance",
            requires_grad=False,
        )
        self.posterior_log_variance_clipped = Parameter(
            to_mindspore(np.log(np.maximum(posterior_variance, 1e-20))),
            name="posterior_log_variance_clipped",
            requires_grad=False,
        )
        self.posterior_mean_coef1 = Parameter(
            to_mindspore(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
            name="posterior_mean_coef1",
            requires_grad=False,
        )
        self.posterior_mean_coef2 = Parameter(
            to_mindspore(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
            name="posterior_mean_coef2",
            requires_grad=False,
        )

        lvlb_weights = self.betas**2 / (
            2
            * self.posterior_variance
            * to_mindspore(alphas)
            * (1 - self.alphas_cumprod)
        )
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = Parameter(
            lvlb_weights, name="lvlb_weights", requires_grad=False
        )
        assert not ops.isnan(self.lvlb_weights).all()

    def instantiate_first_stage(self, first_stage_model):
        """
        Initializes and freezes the first stage autoencoder model.
        Args:
            first_stage_model (nn.Cell): Autoencoder model instance
        """
        if isinstance(first_stage_model, nn.Cell):
            model = first_stage_model
        else:
            assert first_stage_model is None
            raise NotImplementedError("No default first_stage_model supported yet!")
        self.first_stage_model = model.set_train(False)
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.trainable_params():
            param.requires_grad = False

    def instantiate_cond_stage(self, cond_stage_forward):
        """Configures conditioning stage encoder with spatial rearrangement."""
        self.cond_stage_model = self.first_stage_model
        for param in self.cond_stage_model.trainable_params():
            param.requires_grad = False
        cond_stage_forward = self.cond_stage_model.encode

        def wrapper(cond_stage_forward: Callable):
            def func(c: Dict[str, Any]):
                c = c.get("y")
                batch_size = c.shape[self.batch_axis]
                c = c.transpose(0, 1, 4, 2, 3)
                n_new, t_new, c_new, h_new, w_new = c.shape
                c = c.reshape(n_new * t_new, c_new, h_new, w_new)
                c = cond_stage_forward(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
                n_new, c_new, h_new, w_new = c.shape
                c = c.reshape(batch_size, -1, c_new, h_new, w_new)
                c = c.transpose(0, 1, 3, 4, 2)
                return c

            return func

        self.cond_stage_forward = wrapper(cond_stage_forward)

    def get_first_stage_encoding(self, encoder_posterior):
        """
        Extracts latent representation from encoder output.
        Args:
            encoder_posterior (Tensor/DiagonalGaussianDistribution): Encoder output
        """
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    @property
    def einops_layout(self):
        """Returns Einops layout string for data rearrangement."""
        return " ".join(self.layout)

    @property
    def einops_spatial_layout(self):
        """Generates spatial Einops pattern for 2D/3D data handling."""
        if not hasattr(self, "_einops_spatial_layout"):
            assert len(self.layout) == 4 or len(self.layout) == 5
            self._einops_spatial_layout = (
                "(N T) C H W" if self.layout.find("T") else "N C H W"
            )
        return self._einops_spatial_layout

    def decode_first_stage(self, z):
        """
        Decodes latent representation to data space with spatial rearrangement.
        Args:
            z (Tensor): Latent tensor
        """
        z = 1.0 / self.scale_factor * z
        batch_size = z.shape[self.batch_axis]
        z = rearrange(
            z.asnumpy(), f"{self.einops_layout} -> {self.einops_spatial_layout}"
        )
        z = Tensor.from_numpy(z)
        output = self.first_stage_model.decode(z)
        output = rearrange(
            output.asnumpy(),
            f"{self.einops_spatial_layout} -> {self.einops_layout}",
            N=batch_size,
        )
        output = Tensor.from_numpy(output)
        return output

    def encode_first_stage(self, x):
        """
        Encodes input data into latent space.
        Args:
            x (Tensor): Input data tensor
        """
        encoder_posterior = self.first_stage_model.encode(x)
        output = self.get_first_stage_encoding(encoder_posterior)
        return output

    def apply_model(self, x_noisy, t, cond):
        """
        Applies main UNet model to denoise inputs.
        Args:
            x_noisy (Tensor): Noisy input tensor
            t (Tensor): Time step tensor
            cond (Dict): Conditioning information
        Returns:
            Tensor: Denoising model output
        """
        x_recon = self.main_model(x_noisy, t, cond)
        if isinstance(x_recon, tuple):
            return x_recon[0]
        return x_recon

    def q_sample(self, x_start, t, noise=None):
        """
        Adds noise to clean data according to diffusion schedule.
        Args:
            x_start (Tensor): Clean data tensor
            t (Tensor): Time step tensor
            noise (Tensor): Optional noise tensor
        Returns:
            Tensor: Noisy data tensor
        """
        noise = default(noise, lambda: ops.randn_like(x_start))
        return (
            self.extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + self.extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def get_loss(self, pred, target, mean=True):
        """
        Calculates loss between prediction and target.
        Args:
            pred (Tensor): Model predictions
            target (Tensor): Target values
            mean (bool): Whether to return mean loss
        Returns:
            Tensor: Loss value(s)
        """
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = mint.nn.functional.mse_loss(target, pred)
            else:
                loss = mint.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        """
        Computes diffusion training loss for given time steps.
        Args:
            x_start (Tensor): Clean data tensor
            cond (Dict): Conditioning information
            t (Tensor): Time step tensor
            noise (Tensor): Optional noise tensor
        Returns:
            Tensor: Total training loss
        """
        noise = default(noise, lambda: ops.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        loss_simple = self.get_loss(model_output, noise, mean=False).mean(
            axis=self.loss_mean_dim
        )

        logvar_t = self.logvar[t]

        loss = loss_simple / ops.exp(logvar_t) + logvar_t

        loss = self.l_simple_weight * loss.mean()
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Reconstructs clean data from noisy input and predicted noise.
        Args:
            x_t (Tensor): Noisy data tensor
            t (Tensor): Time step tensor
            noise (Tensor): Predicted noise tensor
        Returns:
            Tensor: Reconstructed clean data
        """
        sqrt_recip_alphas_cumprod_t = self.extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self.extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        term1 = sqrt_recip_alphas_cumprod_t * x_t
        term2 = sqrt_recipm1_alphas_cumprod_t * noise
        pred = term1 - term2
        return pred

    def q_posterior(self, x_start, x_t, t):
        """
        Calculates posterior distribution parameters for given time steps.
        Args:
            x_start (Tensor): Clean data tensor
            x_t (Tensor): Noisy data tensor
            t (Tensor): Time step tensor
        Returns:
            Tuple[Tensor]: (posterior mean, variance, log variance)
        """
        posterior_mean = (
            self.extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self.extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self,
            zt,
            zc,
            t,
            clip_denoised: bool,
            return_x0=False,
            score_corrector=None,
            corrector_kwargs=None,
    ):
        """
        Computes predicted mean and variance during denoising.
        Args:
            zt (Tensor): Current latent sample
            zc (Tensor): Conditioning tensor
            t (Tensor): Time step tensor
            clip_denoised (bool): Whether to clip denoised outputs
            return_x0 (bool): Whether to return reconstructed x0
            score_corrector (Callable): Optional score correction function
            corrector_kwargs (Dict): Correction function parameters
        Returns:
            Tuple[Tensor]: (mean, variance, log variance, [reconstructed x0])
        """
        t_in = t
        model_out = self.apply_model(zt, t_in, zc)
        if score_corrector is not None:
            model_out = score_corrector.modify_score(
                self, model_out, zt, t, zc, **corrector_kwargs
            )
        z_recon = self.predict_start_from_noise(zt, t=t, noise=model_out)
        if clip_denoised:
            z_recon = z_recon.clamp(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=z_recon, x_t=zt, t=t
        )
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, z_recon
        return model_mean, posterior_variance, posterior_log_variance

    def aligned_mean(self, zt, t, zc, y, orig_mean, orig_log_var, **kwargs):
        """
        Calculates aligned mean using gradient-based alignment function.
        Args:
            zt (Tensor): Current latent sample
            t (Tensor): Time step tensor
            zc (Tensor): Conditioning tensor
            y (Tensor): Ground truth tensor
            orig_mean (Tensor): Original mean
            orig_log_var (Tensor): Original log variance
            **kwargs: Additional alignment parameters
        Returns:
            Tensor: Aligned mean tensor
        """
        align_gradient = self.alignment_fn(zt, t, zc=zc, y=y, **kwargs)
        new_mean = orig_mean - (0.5 * orig_log_var).exp() * align_gradient
        return new_mean

    def p_sample(
            self,
            zt,
            zc,
            t,
            y=None,
            use_alignment=False,
            alignment_kwargs=None,
            clip_denoised=False,
            return_x0=False,
            temperature=1.0,
            noise_dropout=0.0,
            score_corrector=None,
            corrector_kwargs=None,
    ):
        """
        Single step diffusion sampling.
        Args:
            zt (Tensor): Current noisy sample at time step t
            zc (Tensor/Dict): Condition input (latent or processed)
            t (Tensor): Time step tensor
            y (Tensor, optional): Additional conditioning information
            use_alignment (bool): Whether to apply alignment correction
            alignment_kwargs (dict, optional): Parameters for alignment correction
            clip_denoised (bool): Clip model output to [-1,1] range
            return_x0 (bool): Return estimated x0 along with sample
            temperature (float): Noise scaling factor
            noise_dropout (float): Dropout rate for noise component
            score_corrector (object, optional): Model output corrector instance
            corrector_kwargs (dict, optional): Parameters for score correction

        Returns:
            Tensor: Next denoised sample
            Tensor (optional): Estimated x0 if return_x0 is True
        """
        batch_size = zt.shape[self.batch_axis]
        outputs = self.p_mean_variance(
            zt=zt,
            zc=zc,
            t=t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if use_alignment:
            if alignment_kwargs is None:
                alignment_kwargs = {}
            model_mean, posterior_variance, model_log_variance, *_ = outputs
            model_mean = self.aligned_mean(
                zt=zt,
                t=t,
                zc=zc,
                y=y,
                orig_mean=model_mean,
                orig_log_var=model_log_variance,
                **alignment_kwargs,
            )
            outputs = (model_mean, posterior_variance, model_log_variance, *outputs[3:])
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(zt.shape) * temperature
        if noise_dropout > 0.0:
            noise = ops.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask_shape = [
            1,
        ] * len(zt.shape)
        nonzero_mask_shape[self.batch_axis] = batch_size
        nonzero_mask = (1 - (t == 0).float()).reshape(*nonzero_mask_shape)

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
            self,
            cond,
            shape,
            y=None,
            use_alignment=False,
            alignment_kwargs=None,
            return_intermediates=False,
            x_t=None,
            verbose=False,
            timesteps=None,
            mask=None,
            x0=None,
            start_t=None,
            log_every_t=None,
    ):
        """
        Full diffusion sampling loop.
        Args:
            cond (Tensor/Dict): Conditioning input (processed)
            shape (tuple): Output tensor shape (B, C, H, W)
            y (Tensor, optional): Additional conditioning info
            use_alignment (bool): Enable alignment correction during sampling
            alignment_kwargs (dict, optional): Alignment parameters
            return_intermediates (bool): Return intermediate steps
            x_t (Tensor, optional): Initial noise sample (default: random)
            verbose (bool): Show progress bar
            timesteps (int): Number of sampling steps
            mask (Tensor, optional): Mask for conditional generation (requires x0)
            x0 (Tensor, optional): Original image for inpainting/conditional generation
            start_t (int): Override maximum time step
            log_every_t (int): Frequency of intermediate saves

        Returns:
            Tensor: Final generated sample
            list[Tensor] (optional): Intermediate samples if requested
        """

        if not log_every_t:
            log_every_t = self.log_every_t
        batch_size = shape[self.batch_axis]
        if x_t is None:
            img = ops.randn(shape)

        else:
            img = x_t

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_t is not None:
            timesteps = min(timesteps, start_t)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match
        for i in iterator:
            ts = ops.full((batch_size,), i, dtype=ms.int64)
            img = self.p_sample(
                zt=img,
                zc=cond,
                t=ts,
                y=y,
                use_alignment=use_alignment,
                alignment_kwargs=alignment_kwargs,
                clip_denoised=self.clip_denoised,
            )
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    def sample(
            self,
            cond,
            batch_size=16,
            use_alignment=False,
            alignment_kwargs=None,
            return_intermediates=False,
            x_t=None,
            verbose=False,
            timesteps=None,
            mask=None,
            x0=None,
            shape=None,
            return_decoded=True,
    ):
        """
        High-level sampling interface with conditioning handling.

        Args:
            cond (Tensor/Dict): Raw conditioning input (e.g., text/image)
            batch_size (int): Number of samples to generate
            use_alignment (bool): Enable alignment correction
            alignment_kwargs (dict, optional): Alignment parameters
            return_intermediates (bool): Return intermediate steps
            x_t (Tensor, optional): Initial noise sample
            verbose (bool): Show progress
            timesteps (int): Sampling steps
            mask (Tensor, optional): Inpainting mask (requires x0)
            x0 (Tensor, optional): Original image for conditioning
            shape (tuple, optional): Output shape override
            return_decoded (bool): Return decoded image instead of latent

        Returns:
            Tensor: Generated image (decoded if return_decoded)
            list[Tensor] (optional): Decoded intermediate steps if requested
        """
        if shape is None:
            shape = self.get_batch_latent_shape(batch_size=batch_size)
        if self.cond_stage_model is not None:
            assert cond is not None
            cond_tensor_slice = [
                slice(None, None),
            ] * len(self.data_shape)
            cond_tensor_slice[self.batch_axis] = slice(0, batch_size)
            if isinstance(cond, dict):
                zc = {
                    key: (
                        cond[key][cond_tensor_slice]
                        if not isinstance(cond[key], list)
                        else list(map(lambda x: x[cond_tensor_slice], cond[key]))
                    )
                    for key in cond
                }
            else:
                zc = (
                    [c[cond_tensor_slice] for c in cond]
                    if isinstance(cond, list)
                    else cond[cond_tensor_slice]
                )
            zc = self.cond_stage_forward(zc)
        else:
            zc = cond if isinstance(cond, Tensor) else cond.get("y", None)
        y = cond if isinstance(cond, Tensor) else cond.get("y", None)
        output = self.p_sample_loop(
            cond=zc,
            shape=shape,
            y=y,
            use_alignment=use_alignment,
            alignment_kwargs=alignment_kwargs,
            return_intermediates=return_intermediates,
            x_t=x_t,
            verbose=verbose,
            timesteps=timesteps,
            mask=mask,
            x0=x0,
        )
        if return_decoded:
            if return_intermediates:
                samples, intermediates = output
                decoded_samples = self.decode_first_stage(samples)
                decoded_intermediates = [
                    self.decode_first_stage(ele) for ele in intermediates
                ]
                output = [decoded_samples, decoded_intermediates]
            else:
                output = self.decode_first_stage(output)
        return output



class PreDiffModule(LatentDiffusion):
    """
    Main module for pre-training diffusion models with latent representations.
    Integrates configuration loading, model creation, alignment setup, metric initialization,
    and visualization parameters. Extends LatentDiffusion to handle cuboid-based UNet architectures
    and knowledge alignment for sequential data generation tasks.
    """

    def __init__(self, oc_file: str = None):
        self.oc = self._load_configs(oc_file)
        latent_model = self._create_latent_model()
        first_stage_model = self._create_vae_model()
        super().__init__(
            **self._prepare_parent_init_params(latent_model, first_stage_model)
        )
        self._setup_alignment()
        self._initialize_metrics()
        self._setup_visualization()

    def _load_configs(self, oc_file):
        """Loads all configuration files through a unified entry point."""
        oc_from_file = OmegaConf.load(open(oc_file, "r")) if oc_file else None
        return self.get_base_config(oc_from_file=oc_from_file)

    def _create_latent_model(self):
        """Builds the CuboidTransformerUNet model based on configurations."""
        latent_model_cfg = OmegaConf.to_object(self.oc.model.latent_model)
        return CuboidTransformerUNet(
            **{
                k: latent_model_cfg[k]
                for k in latent_model_cfg
            },
        )

    def _process_attention_patterns(self, cfg, num_blocks):
        """Processes attention patterns from configuration settings."""
        if isinstance(cfg["self_pattern"], str):
            return [cfg["self_pattern"]] * num_blocks
        return OmegaConf.to_container(cfg["self_pattern"])

    def _create_vae_model(self):
        """Creates and loads pretrained weights for the VAE model."""
        vae_cfg = OmegaConf.to_object(self.oc.model.vae)
        model = AutoencoderKL(
            **{
                k: vae_cfg[k]
                for k in vae_cfg
                if k not in ["pretrained_ckpt_path", "data_channels"]
            }
        )
        self._load_pretrained_weights(model, vae_cfg["pretrained_ckpt_path"])
        return model

    def _load_pretrained_weights(self, model, ckpt_path):
        """Loads pretrained weights into the given model if a checkpoint path is provided."""
        if ckpt_path:
            param_dict = ms.load_checkpoint(ckpt_path)
            param_not_load, _ = ms.load_param_into_net(model, param_dict)
            if param_not_load:
                print(f"Unloaded AutoencoderKLparameters: {param_not_load}")
        else:
            warnings.warn(
                "Pretrained weights for AutoencoderKL not set. Running sanity check only."
            )

    def _prepare_parent_init_params(self, latent_model, first_stage_model):
        """Prepares initialization parameters for the parent class."""
        diffusion_cfg = OmegaConf.to_object(self.oc.model.diffusion)
        return {
            "main_model": latent_model,
            "layout": self.oc.layout.layout,
            "loss_type": self.oc.optim.loss_type,
            "monitor": self.oc.optim.monitor,
            "first_stage_model": first_stage_model,
            **{
                k: diffusion_cfg[k]
                for k in diffusion_cfg
                if k not in ["latent_cond_shape"]
            },
        }

    def _setup_alignment(self):
        """Sets up alignment using AvgIntensityAlignment if specified in configurations."""
        # from src.knowledge_alignment.alignment_net import AvgIntensityAlignment

        knowledge_cfg = OmegaConf.to_object(self.oc.model.align)
        self.alignment_type = knowledge_cfg["alignment_type"]
        self.use_alignment = self.alignment_type is not None

        if self.use_alignment:
            self.alignment_obj = AvgIntensityAlignment(
                guide_scale=knowledge_cfg["guide_scale"],
                model_args=knowledge_cfg["model_args"],
                model_ckpt_path=knowledge_cfg["model_ckpt_path"],
            )
            self.alignment_obj.model.set_train(False)
            self.set_alignment(self.alignment_obj.get_mean_shift)
        else:
            self.set_alignment(None)

    def _initialize_metrics(self):
        """Initializes metrics for evaluation based on configurations."""
        if self.oc.eval.eval_unaligned:
            self._init_unaligned_metrics()
        if self.oc.eval.eval_aligned:
            self._init_aligned_metrics()

    def _init_unaligned_metrics(self):
        """Initializes unaligned metrics for evaluation."""
        common_args = {
            "mode": self.oc.data.metrics_mode,
            "seq_in": self.oc.layout.t_out,
            "layout": self.layout,
            "threshold_list": self.oc.data.threshold_list,
            "metrics_list": self.oc.data.metrics_list,
            "eps": 1e-4,
        }

        self.valid_score = SEVIRSkillScore(**common_args)

        self.test_ssim = calculate_ssim
        self.test_aligned_ssim = calculate_ssim
        self.test_score = SEVIRSkillScore(**common_args)

    def _init_aligned_metrics(self):
        """Initializes aligned metrics for evaluation."""
        common_args = {
            "mode": self.oc.data.metrics_mode,
            "seq_in": self.oc.layout.t_out,
            "layout": self.layout,
            "threshold_list": self.oc.data.threshold_list,
            "metrics_list": self.oc.data.metrics_list,
            "eps": 1e-4,
        }

        self.valid_aligned_score = SEVIRSkillScore(**common_args)

        self.test_aligned_ssim = nn.SSIM()
        self.test_aligned_score = SEVIRSkillScore(**common_args)

    def _setup_visualization(self):
        """Sets up visualization parameters based on configurations."""
        self.logging_prefix = self.oc.logging.logging_prefix
        self.train_example_data_idx_list = list(
            self.oc.eval.train_example_data_idx_list
        )
        self.val_example_data_idx_list = list(self.oc.eval.val_example_data_idx_list)
        self.test_example_data_idx_list = list(self.oc.eval.test_example_data_idx_list)

    def get_base_config(self, oc_from_file=None):
        """Merges base configuration with configuration loaded from file."""
        if oc_from_file is None:
            raise ValueError("oc_from_file is required but not provided")
        oc = OmegaConf.create()
        oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @classmethod
    def get_total_num_steps(
            cls, num_samples: int, total_batch_size: int, epoch: int = None
    ):
        """
        Parameters
        ----------
        num_samples:    int
            The number of samples of the datasets. `num_samples / micro_batch_size` is the number of steps per epoch.
        total_batch_size:   int
            `total_batch_size == micro_batch_size * world_size * grad_accum`
        epoch: int
        """
        if epoch is None:
            epoch = cls.get_optim_config().max_epochs
        return int(epoch * num_samples / total_batch_size)

    @staticmethod
    def get_sevir_datamodule(
            dataset_cfg, micro_batch_size: int = 1, num_workers: int = 8
    ):
        """Creates and returns a SEVIRDataModule instance based on dataset configurations."""
        dm = SEVIRDataModule(
            sevir_dir=dataset_cfg["root_dir"],
            seq_in=dataset_cfg["seq_in"],
            sample_mode=dataset_cfg["sample_mode"],
            stride=dataset_cfg["stride"],
            batch_size=micro_batch_size,
            layout=dataset_cfg["layout"],
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            verbose=False,
            aug_mode=dataset_cfg["aug_mode"],
            dataset_name=dataset_cfg["dataset_name"],
            start_date=dataset_cfg["start_date"],
            train_val_split_date=dataset_cfg["train_val_split_date"],
            train_test_split_date=dataset_cfg["train_test_split_date"],
            end_date=dataset_cfg["end_date"],
            val_ratio=dataset_cfg["val_ratio"],
            num_workers=num_workers,
            raw_seq_len=dataset_cfg["raw_seq_len"]
        )
        return dm

    @property
    def in_slice(self):
        """Returns the input slice based on layout and sequence length configurations."""
        if not hasattr(self, "_in_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.oc.layout.layout,
                t_in=self.oc.layout.t_in,
                t_out=self.oc.layout.t_out,
            )
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._in_slice

    @property
    def out_slice(self):
        """Returns the output slice based on layout and sequence length configurations."""
        if not hasattr(self, "_out_slice"):
            in_slice, out_slice = layout_to_in_out_slice(
                layout=self.oc.layout.layout,
                t_in=self.oc.layout.t_in,
                t_out=self.oc.layout.t_out,
            )
            self._in_slice = in_slice
            self._out_slice = out_slice
        return self._out_slice

    def get_input(self, batch, **kwargs):
        """Extracts input data and conditioning information from a raw data batch."""
        return self._get_input_sevirlr(
            batch=batch, return_verbose=kwargs.get("return_verbose", False)
        )

    def _get_input_sevirlr(self, batch, return_verbose=False):
        """Specific implementation of input extraction for SEVIRLR dataset."""
        seq = batch
        in_seq = seq[self.in_slice]
        out_seq = seq[self.out_slice]
        if return_verbose:
            return out_seq, {"y": in_seq}, in_seq
        return out_seq, {"y": in_seq}

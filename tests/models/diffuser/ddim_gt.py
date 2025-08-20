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
"""mo-ddim api"""
from typing import List, Optional, Union

import numpy as np

import mindspore as ms
from mindspore import ops, Tensor, set_seed

from utils import rescale_zero_terminal_snr, betas_for_alpha_bar

set_seed(0)
np.random.seed(0)


class DDIMScheduler:
    """
    `DDIMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear",
                 trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                 clip_sample: bool = True,
                 steps_offset: int = 0,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 clip_sample_range: float = 1.0,
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading",
                 rescale_betas_zero_snr: bool = False,
                 ):
        if trained_betas is not None:
            self.betas = ms.tensor(trained_betas, dtype=ms.float32)
        elif beta_schedule == "linear":
            self.betas = ms.tensor(
                np.linspace(beta_start, beta_end, num_train_timesteps), dtype=ms.float32
            )
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                ms.tensor(
                    np.linspace(beta_start**0.5, beta_end **
                                0.5, num_train_timesteps),
                    dtype=ms.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}"
            )
        self.betas = Tensor(self.betas)
        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = ops.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # we set this parameter simply to one or
        self.final_alpha_cumprod = ms.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = None
        self.timesteps = ms.Tensor(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )
        self.timestep_spacing = timestep_spacing
        self.prediction_type = prediction_type
        self.sample_max_value = sample_max_value
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.steps_offset = steps_offset
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

    def _get_variance(self, timestep, prev_timestep):
        """get variance"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep[0] >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    def _threshold_sample(self, sample: ms.Tensor) -> ms.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        (https://arxiv.org/abs/2205.11487)
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (ms.float32, ms.float64):
            # upcast for quantile calculation, and clamp not implemented for cpu half
            sample = sample.float()

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels *
                                np.prod(remaining_dims).item())

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = ms.Tensor.from_numpy(
            np.quantile(abs_sample.asnumpy(),
                        self.dynamic_thresholding_ratio, axis=1)
        )
        s = ops.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        # (batch_size, 1) because clamp will broadcast along dim=0
        s = s.unsqueeze(1)
        # "we threshold xt0 to the range [-s, s] and then divide by s"
        sample = ops.clamp(sample, -s, s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of (https://arxiv.org/abs/2305.08891)
        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_train_timesteps -
                            1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (
                (np.arange(0, num_inference_steps) * step_ratio)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(
                np.arange(self.num_train_timesteps, 0, -step_ratio)
            ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        self.timesteps = ms.Tensor(timesteps)

    def step(self,
             model_output: ms.Tensor,
             timestep: int,
             sample: ms.Tensor,
             eta: float = 0.0,
             use_clipped_model_output: bool = False,
             generator=None,
             variance_noise: Optional[ms.Tensor] = None,
             ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`ms.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`np.random.Generator`, *optional*):
                A random number generator.
            variance_noise (`ms.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].

        Returns:
            [`ms.Tensor`]:
                Sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper (https://arxiv.org/pdf/2010.02502.pdf)
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        dtype = sample.dtype
        batch_size = timestep.shape[0]
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].reshape(-1, 1, 1)
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep[0] >= 0
            else self.final_alpha_cumprod.repeat(batch_size)
        )
        alpha_prod_t_prev = alpha_prod_t_prev.reshape(-1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from (https://arxiv.org/pdf/2010.02502.pdf)
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                (sample - (beta_prod_t ** (0.5)).to(dtype) * model_output)
                / alpha_prod_t ** (0.5)
            ).to(dtype)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                (sample - (alpha_prod_t ** (0.5)).to(dtype) * pred_original_sample)
                / beta_prod_t ** (0.5)
            ).to(dtype)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5).to(dtype) * sample - (
                beta_prod_t**0.5
            ).to(dtype) * model_output
            pred_epsilon = (alpha_prod_t**0.5).to(dtype) * model_output + (
                beta_prod_t**0.5
            ).to(dtype) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # np.save('pred_original_sample.npy', pred_original_sample)
        # 4. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)

        if eta == 0:
            std_dev_t = 0
        else:
            std_dev_t = (eta * ops.sqrt(variance)).to(dtype).reshape(-1, 1, 1)
        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (
                (sample - (alpha_prod_t ** (0.5)).to(dtype) * pred_original_sample)
                / beta_prod_t ** (0.5)
            ).to(dtype)

        # 6. compute "direction pointing to x_t" of formula (12) from (https://arxiv.org/pdf/2010.02502.pdf)
        pred_sample_direction = ((1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5)).to(
            dtype
        ) * pred_epsilon
        # 7. compute x_t without "random noise" of formula (12) from (https://arxiv.org/pdf/2010.02502.pdf)
        coef = (alpha_prod_t_prev ** (0.5)).to(dtype)
        prev_sample = coef * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = ops.randn(model_output.shape, dtype=dtype)
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        return prev_sample

    def add_noise(self,
                  original_samples: ms.Tensor,
                  noise: ms.Tensor,
                  timesteps: ms.Tensor,
                  ):
        """add noise into sample"""
        broadcast_shape = original_samples.shape
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        sqrt_alpha_prod = ops.reshape(
            sqrt_alpha_prod,
            (timesteps.reshape((-1,)).shape[0],) +
            (1,) * (len(broadcast_shape) - 1),
        )

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        sqrt_one_minus_alpha_prod = ops.reshape(
            sqrt_one_minus_alpha_prod,
            (timesteps.reshape((-1,)).shape[0],) +
            (1,) * (len(broadcast_shape) - 1),
        )

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

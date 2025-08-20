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
"""mo-ddpm api"""
from typing import List, Optional, Union

import numpy as np

import mindspore as ms
from mindspore import ops
import mindspore.common.dtype as mstype

from utils import rescale_zero_terminal_snr, betas_for_alpha_bar


class DDPMScheduler:
    """
    `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

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
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
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
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "squaredcos_cap_v2",
                 trained_betas: Optional[Union[np.ndarray,
                                               List[float]]] = None,
                 variance_type: str = "fixed_small_log",
                 clip_sample: bool = True,
                 prediction_type: str = "epsilon",
                 thresholding: bool = False,
                 dynamic_thresholding_ratio: float = 0.995,
                 clip_sample_range: float = 1.0,
                 sample_max_value: float = 1.0,
                 timestep_spacing: str = "leading",
                 steps_offset: int = 0,
                 rescale_betas_zero_snr: int = False,
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
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = ms.tensor(
                np.linspace(-6, 6, num_train_timesteps), dtype=ms.float32)
            self.betas = ops.sigmoid(
                betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}"
            )

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = ops.cumprod(self.alphas, dim=0)
        self.one = ms.Tensor(1.0)

        # standard deviation of the initial noise distribution

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = ms.Tensor(
            np.arange(0, num_train_timesteps)[::-1].copy())
        self.num_train_timesteps = num_train_timesteps
        self.timestep_spacing = timestep_spacing
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.sample_max_value = sample_max_value
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.steps_offset = steps_offset
        self.rescale_betas_zero_snr = rescale_betas_zero_snr

    def set_timesteps(self,
                      num_inference_steps: Optional[int] = None,
                      timesteps: Optional[List[int]] = None,
                      ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError(
                "Can only pass one of `num_inference_steps` or `custom_timesteps`."
            )

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError(
                        "`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.train_timesteps`:"
                    f" {self.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.train_timesteps`:"
                    f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

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
                    f"{self.timestep_spacing} is not supported. Please make sure to choose"
                    "one of 'linspace', 'leading' or 'trailing'."
                )

        self.timesteps = ms.Tensor(timesteps)

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        """get variance"""
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        prev_t = ops.where(prev_t >= 0, prev_t, self.one).astype(mstype.int32)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from (https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / \
            (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = ops.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = variance
        # for rl-diffuser (https://arxiv.org/abs/2205.09991)
        elif variance_type == "fixed_small_log":
            variance = ops.log(variance)
            variance = ops.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = ops.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = ops.log(variance)
            max_log = ops.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def _threshold_sample(self, sample: ms.Tensor) -> ms.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        see (https://arxiv.org/abs/2205.11487)
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

    def step(self,
             model_output: ms.Tensor,
             timestep: int,
             sample: ms.Tensor,
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
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep
        dtype = sample.dtype

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned",
                                                                                   "learned_range",
                                                                                   ]:
            model_output, predicted_variance = ops.split(
                model_output, sample.shape[1], axis=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t].reshape(-1, 1, 1)
        prev_t = ops.where(prev_t >= 0, prev_t, self.one).astype(mstype.int32)

        alpha_prod_t_prev = self.alphas_cumprod[prev_t].reshape(-1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from (https://arxiv.org/pdf/2006.11239.pdf)
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                (sample - (beta_prod_t ** (0.5) * model_output).to(dtype))
                / alpha_prod_t ** (0.5)
            ).to(dtype)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5).to(dtype) * sample - (
                beta_prod_t**0.5
            ).to(dtype) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from (https://arxiv.org/pdf/2006.11239.pdf)
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (
            0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from (https://arxiv.org/pdf/2006.11239.pdf)
        pred_prev_sample = (
            pred_original_sample_coeff.to(dtype) * pred_original_sample
            + current_sample_coeff.to(dtype) * sample
        )

        # 6. Add noise
        variance = 0
        if t[0] > 0:
            v = self._get_variance(t, predicted_variance=predicted_variance).reshape(-1, 1, 1)
            # set pseudo noise for testcase
            # variance_noise = ops.randn(
            #     model_output.shape, dtype=model_output.dtype)
            variance_noise = model_output
            if self.variance_type == "fixed_small_log":
                variance = v.to(dtype) * variance_noise
            elif self.variance_type == "learned_range":
                variance = ops.exp(0.5 * v).to(dtype) * variance_noise
            else:
                variance = ops.sqrt(v).to(dtype) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self,
                  original_samples: ms.Tensor,
                  noise: ms.Tensor,
                  timesteps: ms.Tensor,  # ms.int32
                  ) -> ms.Tensor:
        """add noist to sample"""
        broadcast_shape = original_samples.shape
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_alpha_prod = ops.reshape(
            sqrt_alpha_prod, (timesteps.shape[0],) +
            (1,) * (len(broadcast_shape) - 1)
        )

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        #     sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = ops.reshape(
            sqrt_one_minus_alpha_prod,
            (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1),
        )

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def previous_timestep(self, timestep):
        """get previous timestep"""
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero()[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = ms.Tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps
                if self.num_inference_steps
                else self.num_train_timesteps
            )
            prev_t = timestep - self.num_train_timesteps // num_inference_steps

        return prev_t

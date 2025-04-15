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
# ==============================================================================
"""Diffusion api"""
# pylint: disable=C0301
import math

import numpy as np
from mindspore import dtype as mstype
from mindspore import ops, Tensor, jit_class, nn


def extract(a, t, x_shape):
    """calculate a[timestep]"""
    b = t.shape[0]
    out = Tensor(a).gather(t, -1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.

    Args:
        num_diffusion_timesteps (int): the number of betas to produce.
        max_beta (float): the maximum beta to use; use values lower than 1 to prevent singularities.
        alpha_transform_type (str): the type of noise schedule for alpha_bar. Choose from `cosine` or `exp`.
            Default: ``cosine``.

    Returns:
        numpy.ndarray, the betas used by the scheduler to step the model outputs.
    """
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(
            f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return np.array(betas)


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on <https://arxiv.org/pdf/2305.08891> (Algorithm 1)

    Args:
        betas (Tensor): The betas that the scheduler is being initialized with.

    Returns:
        Tensor, rescaled betas with zero terminal SNR.
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_t = alphas_bar_sqrt[-1].copy()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_t

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / \
        (alphas_bar_sqrt_0 - alphas_bar_sqrt_t)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


@jit_class
class DiffusionScheduler:
    r"""
    Diffusion Scheduler base class.

    Args:
        num_train_timesteps (int): The number of diffusion steps to train the model. Default: ``1000``.
        beta_start (float): The starting `beta` value of inference. Default: ``0.0001``.
        beta_end (float): The final `beta` value. Default: ``0.02``.
        beta_schedule (str): The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`. Default: ``squaredcos_cap_v2``.
        prediction_type (str): Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of `Imagen
            Video <https://imagen.research.google/video/paper.pdf>`_ paper). Default: ``epsilon``.
        clip_sample (bool): Clip the predicted sample for numerical stability. Default: ``True``.
        clip_sample_range (float): The maximum magnitude for sample clipping. Valid only when `clip_sample=True`. Default: ``1.0``.
        thresholding (bool): Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion. Default: ``False``.
        sample_max_value (float): The threshold value for dynamic thresholding. Valid only when `thresholding=True`. Default: ``1.0``.
        dynamic_thresholding_ratio (float): The ratio for the dynamic thresholding method. Valid only when `thresholding=True`. Default: ``0.995``.
        timestep_spacing (str): The way the timesteps should be scaled. Refer to Table 2 of the `Common Diffusion Noise Schedules and
            Sample Steps are Flawed <https://huggingface.co/papers/2305.08891>`_ for more information. Choose from ``linspace``, ``leading`` or ``trailing``.
            Default: ``leading``.
        rescale_betas_zero_snr (bool): Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_. Default: ``False``.
        compute_dtype: the dtype of compute, it can be `mstype.float32` or `mstype.float16`.  Default: ``mstype.float32``, indicates ``mindspore.float32``.
    """

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "squaredcos_cap_v2",
                 prediction_type: str = "epsilon",
                 clip_sample: bool = True,
                 clip_sample_range: float = 1.0,
                 thresholding: bool = False,
                 sample_max_value: float = 1.0,
                 dynamic_thresholding_ratio: float = 0.995,
                 rescale_betas_zero_snr: bool = False,
                 timestep_spacing: str = "leading",
                 compute_dtype=mstype.float32):

        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timestep_spacing = timestep_spacing
        self.prediction_type = prediction_type
        self.sample_max_value = sample_max_value
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.thresholding = thresholding

        self.betas = self._init_betas(beta_schedule, rescale_betas_zero_snr)

        # sampling related parameters
        alphas = 1. - self.betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.pad(
            alphas_cumprod[:-1], (1, 0), constant_values=1)

        self.alphas_cumprod = Tensor(alphas_cumprod, dtype=compute_dtype)
        self.alphas_cumprod_prev = Tensor(
            alphas_cumprod_prev, dtype=compute_dtype)
        self.num_timesteps = num_train_timesteps

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = Tensor(
            np.sqrt(alphas_cumprod), dtype=compute_dtype)
        self.sqrt_one_minus_alphas_cumprod = Tensor(
            np.sqrt(1. - alphas_cumprod), dtype=compute_dtype)

        self.sqrt_recip_alphas_cumprod = Tensor(
            np.sqrt(1. / alphas_cumprod), dtype=compute_dtype)
        self.sqrt_recipm1_alphas_cumprod = Tensor(
            np.sqrt(1. / alphas_cumprod - 1), dtype=compute_dtype)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = self.betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = np.clip(posterior_variance, 1e-20, None)
        self.posterior_variance = Tensor(
            posterior_variance, dtype=compute_dtype)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = Tensor(np.log(
            posterior_variance), dtype=compute_dtype)  # Tensor(np.log(posterior_variance))
        # See formula (7) from `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_
        self.posterior_mean_coef1 = Tensor(
            self.betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=compute_dtype)
        self.posterior_mean_coef2 = Tensor(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=compute_dtype)
        self.num_inference_steps = None
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio

    def _init_betas(self, beta_schedule="squaredcos_cap_v2", rescale_betas_zero_snr=False):
        """init noise beta schedule

        Args:
            beta_schedule (str):
                The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
                `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
            rescale_betas_zero_snr (bool): Whether to rescale the betas to have zero terminal SNR.
                This enables the model to generate very bright and dark samples instead of limiting it to samples with
                medium brightness. Loosely related to `offset_noise
                <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_.

        Returns:
            Tensor, noise coefficients beta.
        """
        betas = None
        if beta_schedule == "linear":
            betas = np.linspace(
                self.beta_start, self.beta_end, self.num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = np.linspace(
                self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(self.num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = np.linspace(-6, 6, self.num_train_timesteps)
            betas = 1. / (1 + np.exp(-betas)) * \
                (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise NotImplementedError(
                f"{beta_schedule} does is not implemented for {self.__class__}")
        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        return betas

    def set_timesteps(self, num_inference_steps):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (int): The number of diffusion steps used when generating samples with a pre-trained model.

        Raises:
            ValueError: If `num_inference_steps` is greater than `num_train_timesteps` .

        """

        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `num_train_timesteps`:"
                f" {self.num_train_timesteps} as the diffusion model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )
        self.num_inference_steps = num_inference_steps
        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of <https://arxiv.org/abs/2305.08891>
        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_train_timesteps -
                            1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int32)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) *
                         step_ratio).round()[::-1].astype(np.int32)
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps // num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(
                np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int32)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        self.num_timesteps = timesteps

    def _threshold_sample(self, sample):
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        see <https://arxiv.org/abs/2205.11487>
        """
        batch_size, channels, *remaining_dims = sample.shape
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"
        s = np.quantile(abs_sample, self.dynamic_thresholding_ratio, axis=1)

        s = ops.clamp(
            s, min=1, max=self.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        # (batch_size, 1) because clamp will broadcast along dim=0
        s = s.unsqueeze(1)
        # "we threshold xt0 to the range [-s, s] and then divide by s"
        sample = ops.clamp(sample, -s, s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)

        return sample

    def _pred_origin_sample(self, model_output: Tensor, sample: Tensor, timestep: Tensor):
        """
        Predict x_0 with x_t.

        Args:
            model_output (Tensor): The direct output from learned diffusion model.
            sample (Tensor): A current instance of a sample created by the diffusion process.
            timestep (Tensor): The current discrete timestep in the diffusion chain.

        Returns:
            Tensor, the predicted origin sample x_0.

        """
        if self.prediction_type == "epsilon":
            pred_original_sample = extract(self.sqrt_recip_alphas_cumprod, timestep, sample.shape)*sample - \
                extract(self.sqrt_recipm1_alphas_cumprod,
                        timestep, sample.shape)*model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = extract(self.sqrt_alphas_cumprod, timestep, sample.shape)*sample - \
                extract(self.sqrt_one_minus_alphas_cumprod,
                        timestep, sample.shape)*model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        # 2. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range,
                                                              self.clip_sample_range)
        return pred_original_sample

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor):
        """
        Diffusion add noise process.

        Args:
            original_samples (Tensor): The current samples.
            noise (Tensor): Random noise to be add into sample.
            timesteps (Tensor): The current discrete timestep in the diffusion chain.

        Returns:
            Tensor, the noised sample of the next step.
        """
        return (extract(self.sqrt_alphas_cumprod, timesteps, original_samples.shape)*original_samples +
                extract(self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape)*noise)

    def step(self, model_output: Tensor, sample: Tensor, timestep: Tensor):
        """
        Diffusion denoising step.

        Args:
            model_output (Tensor): The direct output from learned diffusion model.
            sample (Tensor): A current instance of a sample created by the diffusion process.
            timestep (Tensor): The current discrete timestep in the diffusion chain.

        Returns:
            Tensor, the denoised sample.
        """
        if not self.num_inference_steps:
            raise NotImplementedError(
                f"num_inference_steps is not set for {self.__class__}.Need to set timesteps first.")
        raise NotImplementedError(
            f"step function does is not implemented for {self.__class__}")


class DDPMScheduler(DiffusionScheduler):
    r"""
    `DDPMScheduler` is an implementation of the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs).
    Check `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_ for more information.

    Args:
        num_train_timesteps (int): The number of diffusion steps to train the model. Default: ``1000``.
        beta_start (float): The starting `beta` value of inference. Default: ``0.0001``.
        beta_end (float): The final `beta` value.  Default: ``0.02``.
        beta_schedule (str): The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`. Default: ``squaredcos_cap_v2``.
        prediction_type (str): Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of `Imagen
            Video <https://imagen.research.google/video/paper.pdf>`_ paper). Default: ``epsilon``.
        variance_type (str): Clip the variance when adding noise to the denoised sample. Choose from ``fixed_small``, ``fixed_small_log``,
            ``fixed_large``, ``fixed_large_log``, ``learned`` or ``learned_range``. Default: ``fixed_small_log``.
        clip_sample (bool): Clip the predicted sample for numerical stability. Default: ``True``.
        clip_sample_range (float): The maximum magnitude for sample clipping. Valid only when `clip_sample=True`. Default: ``1.0``.
        thresholding (bool): Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion. Default: ``False``.
        sample_max_value (float): The threshold value for dynamic thresholding. Valid only when `thresholding=True`. Default: ``1.0``.
        dynamic_thresholding_ratio (float): The ratio for the dynamic thresholding method. Valid only when `thresholding=True`. Default: ``0.995``.
        timestep_spacing (str): The way the timesteps should be scaled. Refer to Table 2 of the `Common Diffusion Noise Schedules and
            Sample Steps are Flawed <https://huggingface.co/papers/2305.08891>`_ for more information. Choose from ``linspace``, ``leading`` or ``trailing``.
            Default: ``leading``.
        rescale_betas_zero_snr (bool): Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_. Default: ``False``.
        compute_dtype: the dtype of compute, it can be `mstype.float32` or `mstype.float16`. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops, dtype as mstype
        >>> from mindflow.cell import DDPMScheduler
        >>> scheduler = DDPMScheduler(num_train_timesteps=1000,
        ...                           beta_start=0.0001,
        ...                           beta_end=0.02,
        ...                           beta_schedule="squaredcos_cap_v2",
        ...                           prediction_type='epsilon',
        ...                           variance_type='fixed_small_log',
        ...                           clip_sample=True,
        ...                           clip_sample_range=1.0,
        ...                           thresholding=False,
        ...                           sample_max_value=1.,
        ...                           dynamic_thresholding_ratio=0.995,
        ...                           rescale_betas_zero_snr=False,
        ...                           timestep_spacing="leading",
        ...                           compute_dtype=mstype.float32)
        >>> batch_size, seq_len, in_dim = 4, 256, 16
        >>> original_samples = ops.randn([batch_size, seq_len, in_dim])
        >>> noise = ops.randn([batch_size, seq_len, in_dim])
        >>> timesteps = ops.randint(0, 100, [batch_size, 1])
        >>> noised_sample = scheduler.add_noise(original_samples, noise, timesteps)
        >>> print(noised_sample.shape)
        (4, 256, 16)
        >>> sample_timesteps = Tensor(np.array([60]*batch_size), dtype=mstype.int32)
        >>> x_prev = scheduler.step(noise, noised_sample, sample_timesteps)
        >>> print(x_prev.shape)
        (4, 256, 16)
    """

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "squaredcos_cap_v2",
                 prediction_type: str = "epsilon",
                 variance_type: str = "fixed_small_log",
                 clip_sample: bool = True,
                 clip_sample_range: float = 1.0,
                 thresholding: bool = False,
                 sample_max_value: float = 1.0,
                 dynamic_thresholding_ratio: float = 0.995,
                 rescale_betas_zero_snr: bool = False,
                 timestep_spacing: str = "leading",
                 compute_dtype=mstype.float32):
        super().__init__(num_train_timesteps,
                         beta_start,
                         beta_end,
                         beta_schedule,
                         prediction_type,
                         clip_sample,
                         clip_sample_range,
                         thresholding,
                         sample_max_value,
                         dynamic_thresholding_ratio,
                         rescale_betas_zero_snr,
                         timestep_spacing,
                         compute_dtype)
        self.variance_type = variance_type

    def _get_variance(self, x_t, t, predicted_variance=None):
        """get DDPM variance"""
        variance = extract(self.posterior_variance, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        # hacks - were probably added for training stability
        if self.variance_type == "fixed_small":
            pass
        # for rl-diffuser `Planning with Diffusion for Flexible Behavior Synthesis <https://arxiv.org/abs/2205.09991>`_
        elif self.variance_type == "fixed_small_log":
            variance = ops.log(variance)
            variance = ops.exp(0.5 * variance)
        elif self.variance_type == "fixed_large":
            variance = beta_t
        elif self.variance_type == "fixed_large_log":
            # Glide max_log
            variance = ops.log(beta_t)
        elif self.variance_type == "learned":
            return predicted_variance
        elif self.variance_type == "learned_range":
            min_log = ops.log(variance)
            max_log = ops.log(beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def set_timesteps(self, num_inference_steps):
        """
        DDPM step inference timestep.

        Args:
            num_inference_steps (int): The denoising step number.

        Raises:
            ValueError: If `num_inference_steps` is not equal to `num_train_timesteps`.
        """
        # DDPM num_inference_steps defaults to num_train_timesteps
        if num_inference_steps != self.num_train_timesteps:
            raise ValueError(
                "DDPM num_inference_steps defaults to num_train_timesteps")
        super().set_timesteps(self.num_train_timesteps)

    # pylint: disable=W0221
    def step(self, model_output, sample, timestep, predicted_variance=None):
        """
        DDPM denoising step.

        Args:
            model_output (Tensor): The direct output from learned diffusion model.
            sample (Tensor): A current instance of a sample created by the diffusion process.
            timestep (Tensor): The current discrete timestep in the diffusion chain.
            predicted_variance (Tensor): The predicted variance. Default: ``None``.

        Returns:
            Tensor, the sample of last step.
        """
        # 1. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_

        pred_original_sample = self._pred_origin_sample(
            model_output, sample, timestep)
        # 2. Compute predicted previous sample µ_t
        # See formula (7) from `Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239>`_
        pred_prev_sample = (
            extract(self.posterior_mean_coef1, timestep, sample.shape)*pred_original_sample +
            extract(self.posterior_mean_coef2, timestep, sample.shape)*sample
        )

        # 3. Add noise
        v = self._get_variance(sample, timestep, predicted_variance)
        variance = 0
        if timestep[0] > 0:
            variance_noise = ops.randn_like(sample)
            if self.variance_type == "fixed_small_log":
                variance = v * variance_noise
            elif self.variance_type == "learned_range":
                variance = ops.exp(0.5 * v) * variance_noise
            else:
                variance = (v ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample


class DDIMScheduler(DiffusionScheduler):
    r"""
    `DDIMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models.
    Check `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_
    for more information.

    Args:
        num_train_timesteps (int): The number of diffusion steps to train the model. Default: ``1000``.
        beta_start (float): The starting `beta` value of inference. Default: ``0.0001``.
        beta_end (float): The final `beta` value.  Default: ``0.02``.
        beta_schedule (str): The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`. Default: ``squaredcos_cap_v2``.
        prediction_type (str): Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample) or `v_prediction` (see section 2.4 of `Imagen
            Video <https://imagen.research.google/video/paper.pdf>`_ paper). Default: ``epsilon``.
        clip_sample (bool): Clip the predicted sample for numerical stability. Default: ``True``.
        clip_sample_range (float): The maximum magnitude for sample clipping. Valid only when `clip_sample=True`. Default: ``1.0``.
        thresholding (bool): Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion. Default: ``False``.
        sample_max_value (float): The threshold value for dynamic thresholding. Valid only when `thresholding=True`. Default: ``1.0``.
        dynamic_thresholding_ratio (float): The ratio for the dynamic thresholding method. Valid only when `thresholding=True`. Default: ``0.995``.
        timestep_spacing (str): The way the timesteps should be scaled. Refer to Table 2 of the `Common Diffusion Noise Schedules and
            Sample Steps are Flawed <https://huggingface.co/papers/2305.08891>`_ for more information. Choose from ``linspace``, ``leading`` or ``trailing``.
            Default: ``leading``.
        rescale_betas_zero_snr (bool): Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            `offset_noise <https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506>`_. Default: ``False``.
        compute_dtype (mindspore.dtype): the dtype of compute, it can be `mstype.float32` or `mstype.float16`. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops, dtype as mstype
        >>> from mindflow.cell import DDPMScheduler
        >>> scheduler = DDIMScheduler(num_train_timesteps=1000,
        ...                           beta_start=0.0001,
        ...                           beta_end=0.02,
        ...                           beta_schedule="squaredcos_cap_v2",
        ...                           prediction_type='epsilon',
        ...                           clip_sample=True,
        ...                           clip_sample_range=1.0,
        ...                           thresholding=False,
        ...                           sample_max_value=1.,
        ...                           dynamic_thresholding_ratio=0.995,
        ...                           rescale_betas_zero_snr=False,
        ...                           timestep_spacing="leading",
        ...                           compute_dtype=mstype.float32)
        >>> batch_size, seq_len, in_dim = 4, 256, 16
        >>> original_samples = ops.randn([batch_size, seq_len, in_dim])
        >>> noise = ops.randn([batch_size, seq_len, in_dim])
        >>> timesteps = ops.randint(0, 100, [batch_size, 1])
        >>> noised_sample = scheduler.add_noise(original_samples, noise, timesteps)
        >>> print(noised_sample.shape)
        (4, 256, 16)
        >>> sample_timesteps = Tensor(np.array([60]*batch_size), dtype=mstype.int32)
        >>> x_prev = scheduler.step(noise, noised_sample, sample_timesteps)
        >>> print(x_prev.shape)
        (4, 256, 16)
    """

    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "squaredcos_cap_v2",
                 prediction_type: str = "epsilon",
                 clip_sample: bool = True,
                 clip_sample_range: float = 1.0,
                 thresholding: bool = False,
                 sample_max_value: float = 1.0,
                 dynamic_thresholding_ratio: float = 0.995,
                 rescale_betas_zero_snr: bool = False,
                 timestep_spacing: str = "leading",
                 compute_dtype=mstype.float32):
        super().__init__(num_train_timesteps,
                         beta_start,
                         beta_end,
                         beta_schedule,
                         prediction_type,
                         clip_sample,
                         clip_sample_range,
                         thresholding,
                         sample_max_value,
                         dynamic_thresholding_ratio,
                         rescale_betas_zero_snr,
                         timestep_spacing,
                         compute_dtype)

        self.final_alpha_cumprod = Tensor(1.0, dtype=compute_dtype)

    def _get_variance(self, timestep, prev_timestep):
        """get DDIM variance"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep[0] >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * \
            (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def _pred_epsilon(self, model_output: Tensor, sample: Tensor, timestep: Tensor):
        """
        Predict epsilon.

        Args:
            model_output (Tensor): The direct output from learned diffusion model.
            sample (Tensor): A current instance of a sample created by the diffusion process.
            timestep (Tensor): The current discrete timestep in the diffusion chain.

        Returns:
            Tensor, the predicted epsilon.
        """
        if self.prediction_type == "epsilon":
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            tmp = (sample - extract(self.sqrt_alphas_cumprod,
                                    timestep, sample.shape) * model_output)
            pred_epsilon = extract(ops.reciprocal(
                self.sqrt_one_minus_alphas_cumprod), timestep, tmp.shape)*tmp
        else:
            pred_epsilon = extract(self.alphas_cumprod, timestep, sample.shape) * model_output + extract(
                self.sqrt_one_minus_alphas_cumprod, timestep, sample.shape) * sample

        return pred_epsilon

    # pylint: disable=W0221
    def step(self,
             model_output: Tensor,
             sample: Tensor,
             timestep: Tensor,
             eta: float = 0.0,
             use_clipped_model_output: bool = False,
             ):
        """
        DDIM denoising step.

        Args:
            model_output (Tensor): The direct output from learned diffusion model.
            sample (Tensor): A current instance of a sample created by the diffusion process.
            timestep (Tensor): The current discrete timestep in the diffusion chain.
            eta (float): The weight of noise for added noise in diffusion step. DDIM when eta=0, DDPM when eta=1. Default: ``0.0``.
            use_clipped_model_output (bool): If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect. Default: ``False.``.

        Raises:
            ValueError: If `eta` not in [0, 1].

        Returns:
            Tensor, Denoised output x_prev.
        """
        # See formulas (12) and (16) of DDIM paper `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"
        if not 0 <= eta <= 1.:
            raise ValueError("eta must in range: [0, 1]")
        dtype = sample.dtype
        batch_size = timestep.shape[0]
        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_len = self.alphas_cumprod.shape[0]
        assert (timestep < alpha_len).all(), "timestep out of bounds"
        assert (prev_timestep < alpha_len).all()

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep[0] >= 0 else \
            self.final_alpha_cumprod.repeat(batch_size)
        # 2. get α_t/α_t−1
        beta_prod_t = 1 - alpha_prod_t
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_
        pred_original_sample = self._pred_origin_sample(
            model_output, sample, timestep)
        pred_epsilon = self._pred_epsilon(model_output, sample, timestep)
        # 4. compute variance: "sigma_t(η)" -> see formula (16) from `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * ops.sqrt(variance)).astype(dtype)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (
                (sample - (alpha_prod_t ** (0.5)).astype(dtype) *
                 pred_original_sample) / beta_prod_t ** (0.5)
            ).astype(dtype)

        # 5. compute "direction pointing to x_t" of formula (12) from `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_
        pred_sample_direction = (
            (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5)).reshape(batch_size, 1, 1) * pred_epsilon

        # 6. compute x_t without "random noise" of formula (12) from `Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502>`_
        coef = ops.sqrt(alpha_prod_t_prev).reshape(batch_size, 1, 1)
        prev_sample = coef * pred_original_sample + pred_sample_direction
        if eta > 0:
            variance_noise = ops.randn_like(model_output, dtype=dtype)
            variance = std_dev_t.reshape(batch_size, 1, 1) * variance_noise
            prev_sample += variance
        return prev_sample


class DiffusionPipeline:
    r"""
    Pipeline for diffusion generation.

    Args:
        model (nn.Cell): The diffusion backbone model.
        scheduler (`DiffusionScheduler`): A scheduler to be used in combination with `model` to denoise the encoded image.
        batch_size (int): The number of images to generate.
        seq_len (int): Sequence length of inputs.
        num_inference_steps (int): Number of Denoising steps.
        compute_dtype (mindspore.dtype): The dtype of compute, it can be mstype.float32 or mstype.float16.
            Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Raises:
        TypeError: If `scheduler` is not `DiffusionScheduler` type.
        ValueError: If `num_inference_steps` is greater than `scheduler.num_train_timesteps`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops, dtype as mstype
        >>> from mindflow.cell import DiffusionPipeline, DDPMScheduler, ConditionDiffusionTransformer
        >>> # init params
        >>> in_dim, out_dim, hidden_dim, cond_dim, layers, heads, seq_len, batch_size = 16, 16, 256, 4, 3, 4, 256, 8
        >>> # init condition
        >>> cond = ops.randn([8, 4])
        >>> # init model and scheduler
        >>> net = ConditionDiffusionTransformer(in_channels=in_dim,
        ...                                     out_channels=in_dim,
        ...                                     cond_channels=cond_dim,
        ...                                     hidden_channels=hidden_dim,
        ...                                     layers=layers,
        ...                                     heads=heads,
        ...                                     time_token_cond=True,
        ...                                     compute_dtype=mstype.float32)
        >>> num_train_timesteps = 100
        >>> scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
        ...                           beta_start=0.0001,
        ...                           beta_end=0.02,
        ...                           beta_schedule="squaredcos_cap_v2",
        ...                           clip_sample=True,
        ...                           clip_sample_range=1.0,
        ...                           thresholding=False,
        ...                           dynamic_thresholding_ratio=None,
        ...                           rescale_betas_zero_snr=False,
        ...                           timestep_spacing="leading",
        ...                           compute_dtype=mstype.float32)
        >>> # init pipeline
        >>> pipe = DiffusionPipeline(model=net, scheduler=scheduler,
        >>>                          batch_size=batch_size, seq_len=seq_len, num_inference_steps=num_train_timesteps)
        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(cond)
        >>> print(image.shape)
        (8, 256, 16)

    """

    def __init__(self, model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32):
        self.model = model
        self.scheduler = scheduler
        self.seq_len = seq_len
        self.compute_dtype = compute_dtype
        self.batch_size = batch_size
        if not isinstance(scheduler, DiffusionScheduler):
            raise TypeError('scheduler type must be DiffusionScheduler')
        if num_inference_steps > scheduler.num_train_timesteps:
            raise ValueError(
                'num_inference_steps must not greater than num_train_timesteps')
        self.scheduler.set_timesteps(num_inference_steps)

    def _pred_noise(self, sample, condition, timesteps):
        if condition is not None:
            inputs = (sample, timesteps, condition)
        else:
            inputs = (sample, timesteps)
        model_output = self.model(*inputs)
        return model_output

    def _sample_step(self, sample, condition, timesteps):
        model_output = self._pred_noise(sample, condition, timesteps)
        sample = self.scheduler.step(
            model_output=model_output, sample=sample, timestep=timesteps)
        return sample

    def __call__(self, condition=None):
        r"""
        The call function to the pipeline for generation.

        Args:
            condition (Tensor): Condition for diffusion generation process. Default: ``None``.

        Returns:
            Tensor, Predicted original samples.
        """
        sample = Tensor(np.random.randn(self.batch_size, self.seq_len,
                                        self.model.in_channels), dtype=self.compute_dtype)
        if condition is not None:
            condition = condition.reshape(self.batch_size, -1)

        for t in self.scheduler.num_timesteps:
            batched_times = ops.ones((self.batch_size,), mstype.int32) * int(t)
            sample = self._sample_step(sample, condition, batched_times)
        return sample


class DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for DDPM generation.

    Args:
        model (nn.Cell): The diffusion backbone model.
        scheduler (DDPMScheduler): A scheduler to be used in combination with `model` to denoise the noised sample.
        batch_size (int): The number of images to generate.
        seq_len (int): Sequence length of inputs.
        num_inference_steps (int): Number of Denoising steps. Default: ``1000``.
        compute_dtype (mindspore.dtype): The dtype of compute, it can be mstype.float32 or mstype.float16.
            Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Raises:
        TypeError: If `scheduler` is not `DDIMScheduler` type.
        ValueError: If `num_inference_steps` is not equal to `scheduler.num_train_timesteps`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops, dtype as mstype
        >>> from mindflow.cell import DDPMPipeline, DDPMScheduler, ConditionDiffusionTransformer
        >>> # init params
        >>> in_dim, out_dim, hidden_dim, cond_dim, layers, heads, seq_len, batch_size = 16, 16, 256, 4, 3, 4, 256, 8
        >>> # init condition
        >>> cond = ops.randn([8, 4])
        >>> # init model and scheduler
        >>> net = ConditionDiffusionTransformer(in_channels=in_dim,
        ...                                     out_channels=in_dim,
        ...                                     cond_channels=cond_dim,
        ...                                     hidden_channels=hidden_dim,
        ...                                     layers=layers,
        ...                                     heads=heads,
        ...                                     time_token_cond=True,
        ...                                     compute_dtype=mstype.float32)
        >>> num_train_timesteps = 100
        >>> scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
        ...                           beta_start=0.0001,
        ...                           beta_end=0.02,
        ...                           beta_schedule="squaredcos_cap_v2",
        ...                           clip_sample=True,
        ...                           clip_sample_range=1.0,
        ...                           thresholding=False,
        ...                           dynamic_thresholding_ratio=None,
        ...                           rescale_betas_zero_snr=False,
        ...                           timestep_spacing="leading",
        ...                           compute_dtype=mstype.float32)
        >>> # init pipeline
        >>> pipe = DDPMPipeline(model=net, scheduler=scheduler,
        >>>                     batch_size=batch_size, seq_len=seq_len, num_inference_steps=num_train_timesteps)
        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(cond)
        >>> print(image.shape)
        (8, 256, 16)
    """
    # pylint: disable=W0235

    def __init__(self, model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32):
        if not isinstance(scheduler, DDPMScheduler):
            raise TypeError('scheduler type must be DDPMScheduler')
        if num_inference_steps != scheduler.num_train_timesteps:
            raise ValueError('num_inference_steps must equal to num_train_timesteps')
        super().__init__(model, scheduler, batch_size,
                         seq_len, num_inference_steps, compute_dtype)


class DDIMPipeline(DiffusionPipeline):
    r"""
    Pipeline for DDIM generation.

    Args:
        model (nn.Cell): The diffusion backbone model.
        scheduler (DDIMScheduler): A scheduler to be used in combination with `model` to denoise samples.
        batch_size (int): The number of images to generate.
        seq_len (int): Sequence length of inputs.
        num_inference_steps (int): Number of Denoising steps. Default: ``1000``.
        compute_dtype (mindspore.dtype): The dtype of compute, it can be mstype.float32 or mstype.float16. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Raises:
        TypeError: If `scheduler` is not `DDIMScheduler` type.
        ValueError: If `num_inference_steps` is greater than `scheduler.num_train_timesteps` .

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops, dtype as mstype
        >>> from mindflow.cell import DDIMPipeline, DDPMScheduler, ConditionDiffusionTransformer
        >>> # init params
        >>> in_dim, out_dim, hidden_dim, cond_dim, layers, heads, seq_len, batch_size = 16, 16, 256, 4, 3, 4, 256, 8
        >>> # init condition
        >>> cond = ops.randn([8, 4])
        >>> # init model and scheduler
        >>> net = ConditionDiffusionTransformer(in_channels=in_dim,
        ...                                     out_channels=in_dim,
        ...                                     cond_channels=cond_dim,
        ...                                     hidden_channels=hidden_dim,
        ...                                     layers=layers,
        ...                                     heads=heads,
        ...                                     time_token_cond=True,
        ...                                     compute_dtype=mstype.float32)
        >>> num_train_timesteps = 100
        >>> scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
        ...                           beta_start=0.0001,
        ...                           beta_end=0.02,
        ...                           beta_schedule="squaredcos_cap_v2",
        ...                           clip_sample=True,
        ...                           clip_sample_range=1.0,
        ...                           thresholding=False,
        ...                           dynamic_thresholding_ratio=None,
        ...                           rescale_betas_zero_snr=False,
        ...                           timestep_spacing="leading",
        ...                           compute_dtype=mstype.float32)
        >>> # init pipeline
        >>> pipe = DDIMPipeline(model=net, scheduler=scheduler,
        >>>                     batch_size=batch_size, seq_len=seq_len, num_inference_steps=num_train_timesteps)
        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe(cond)
        >>> print(image.shape)
        (8, 256, 16)

    """
    # pylint: disable=W0235

    def __init__(self, model, scheduler, batch_size, seq_len, num_inference_steps=1000, compute_dtype=mstype.float32):
        if not isinstance(scheduler, DDIMScheduler):
            raise TypeError('scheduler type must be DDIMScheduler')
        super().__init__(model, scheduler, batch_size,
                         seq_len, num_inference_steps, compute_dtype)

    # pylint: disable=W0221
    def _sample_step(self, sample, condition, timesteps, eta, use_clipped_model_output):
        model_output = self._pred_noise(sample, condition, timesteps)
        sample = self.scheduler.step(model_output=model_output, sample=sample, timestep=timesteps,
                                     eta=eta, use_clipped_model_output=use_clipped_model_output)
        return sample

    def __call__(self, condition=None, eta=0., use_clipped_model_output=False):
        r"""
        The call function to the pipeline for generation.

        Args:
            condition (Tensor): Condition for diffusion generation process.
            eta (float): The weight of noise for added noise in diffusion step. Default: ``0``.
            use_clipped_model_output (bool): If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.scheduler.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect. Default: ``False``.

        Raises:
            ValueError: If `eta` not in [0, 1].

        Returns:
            Tensor, Predicted original samples.
        """
        if not 0 <= eta <= 1:
            raise ValueError('eta must in range [0, 1]')
        sample = Tensor(np.random.randn(self.batch_size, self.seq_len,
                                        self.model.in_channels), dtype=self.compute_dtype)
        if condition is not None:
            condition = condition.reshape(self.batch_size, -1)

        for t in self.scheduler.num_timesteps:
            batched_times = ops.ones((self.batch_size,), mstype.int32) * int(t)
            sample = self._sample_step(
                sample, condition, batched_times, eta, use_clipped_model_output)

        return sample


@jit_class
class DiffusionTrainer:
    r"""
    Diffusion Trainer base class.

    Args:
        model (nn.Cell): The diffusion backbone model.
        scheduler (DiffusionScheduler): DDPM or DDIM scheduler.
        objective (str): Prediction type of the scheduler function;
            can be `pred_noise` (predicts the noise of the diffusion process), `pred_x0` (predicts the original sample`) or
            `pred_v` (see section 2.4 of `Imagen Video <https://imagen.research.google/video/paper.pdf>`_ paper). Default: ``pred_noise``.
        p2_loss_weight_gamma (float): p2 loss weight gamma, from
            `Perception Prioritized Training of Diffusion Models <https://arxiv.org/abs/2204.00227>`_. Default: ``0``.
        p2_loss_weight_k (float): p2 loss weight k, from
            `Perception Prioritized Training of Diffusion Models <https://arxiv.org/abs/2204.00227>`_. Default: ``1``.
        loss_type (str): The type of loss, it can be l1 or l2. Default: ``l1``.

    Raises:
        TypeError: If `scheduler` is not `DiffusionScheduler` type.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops, dtype as mstype
        >>> from mindflow.cell import DDPMScheduler, ConditionDiffusionTransformer, DiffusionTrainer
        >>> # init params
        >>> batch_size, seq_len, in_dim, cond_dim, num_train_timesteps = 4, 256, 16, 4, 100
        >>> original_samples = ops.randn([batch_size, seq_len, in_dim])
        >>> noise = ops.randn([batch_size, seq_len, in_dim])
        >>> timesteps = ops.randint(0, num_train_timesteps, [batch_size, 1])
        >>> cond = ops.randn([batch_size, cond_dim])
        >>> # init model and scheduler
        >>> net = ConditionDiffusionTransformer(in_channels=in_dim,
        ...                                     out_channels=in_dim,
        ...                                     cond_channels=cond_dim,
        ...                                     hidden_channels=hidden_dim,
        ...                                     layers=layers,
        ...                                     heads=heads,
        ...                                     time_token_cond=True,
        ...                                     compute_dtype=mstype.float32)
        >>> scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
        ...                           beta_start=0.0001,
        ...                           beta_end=0.02,
        ...                           beta_schedule="squaredcos_cap_v2",
        ...                           clip_sample=True,
        ...                           clip_sample_range=1.0,
        ...                           thresholding=False,
        ...                           dynamic_thresholding_ratio=None,
        ...                           rescale_betas_zero_snr=False,
        ...                           timestep_spacing="leading",
        ...                           compute_dtype=mstype.float32)
        >>> # init trainer
        >>> trainer = DiffusionTrainer(net,
        ...                            scheduler,
        ...                            objective='pred_noise',
        ...                            p2_loss_weight_gamma=0,
        ...                            p2_loss_weight_k=1,
        ...                            loss_type='l2')
        >>> loss = trainer.get_loss(original_samples, noise, timesteps, condition)

    """

    def __init__(self,
                 model,
                 scheduler,
                 objective='pred_noise',
                 p2_loss_weight_gamma=0.,
                 p2_loss_weight_k=1,
                 loss_type='l1',
                 ):
        if not isinstance(scheduler, DiffusionScheduler):
            raise TypeError('scheduler type must be DiffusionScheduler')
        self.model = model
        self.scheduler = scheduler
        p2_loss_weight = (p2_loss_weight_k + scheduler.alphas_cumprod /
                          (1 - scheduler.alphas_cumprod)) ** -p2_loss_weight_gamma
        self.p2_loss_weight = Tensor(p2_loss_weight, mstype.float32)
        self.objective = objective
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss('none')
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss('none')
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def get_loss(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor, condition: Tensor = None):
        r"""
        Calculate the forward loss of diffusion process.

        Args:
            original_samples (Tensor): The direct output from learned diffusion model.
            noise (Tensor): A current instance of a noise sample created by the diffusion process.
            timesteps (Tensor): The current discrete timestep in the diffusion chain.
            condition (Tensor): The condition for desired outputs. Default: ``None``.

        Returns:
            Tensor, the model forward loss.
        """

        noised_sample = self.scheduler.add_noise(
            original_samples, noise, timesteps)
        if condition is None:
            inputs = (noised_sample, timesteps)
        else:
            inputs = (noised_sample, timesteps, condition)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = original_samples
        elif self.objective == 'pred_v':
            target = (extract(self.scheduler.sqrt_alphas_cumprod, timesteps, original_samples.shape)*noise -
                      extract(self.scheduler.sqrt_one_minus_alphas_cumprod, timesteps,
                              original_samples.shape)*original_samples)
        else:
            target = noise

        model_out = self.model(*inputs)
        loss = self.loss_fn(model_out, target)
        loss = loss.reshape(loss.shape[0], -1)
        loss = loss * extract(self.p2_loss_weight, timesteps, loss.shape)
        return loss.mean()

# Copyright 2024 Huawei Technologies Co., Ltd
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
"""diffution utils file"""
import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

def mindspore_random_choice(low, high, size):
    """ Mimic np.random.choice for integers in MindSpore """
    indices = ops.UniformInt()((size,), low, high)
    return indices

def cosine_beta_schedule(timesteps, s=0.008):
    """
    The beta scheduled by cosine in DDPM used for lattice diffution.
    See details in the paper of DiffCSP.
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    """
    The beta scheduled by linear in DDPM.
    """
    return np.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    """
    The beta scheduled by quadratic in DDPM.
    """
    return np.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    """
    The beta scheduled by sigmoid in DDPM.
    """
    betas = np.linspace(-6, 6, timesteps)
    return 1 / (1 + np.exp(-betas)) * (beta_end - beta_start) + beta_start

def p_wrapped_normal(x, sigma, n=10, t=1.0):
    """Utils for calcatating the score of wrapped normal distribution.
    """
    p_ = 0
    for i in range(-n, n + 1):
        p_ += np.exp(-(x + t * i) ** 2 / 2 / sigma ** 2)
    return p_

def d_log_p_wrapped_normal(x, sigma, n=10, t=1.0):
    """The score of wrapped normal distribution, which is parameterized by sigma,
       for the input value x. See details in Appendix B.1 in the paper of DiffCSP.

    Args:
        x (numpy.ndarray): Input noise.
        sigma (numpy.ndarray): The variance of wrapped normal distribution.
        n (int): The approximate parameter of the score of wrapped normal distribution. Defaults to 10.
        t (int): The period of wrapped normal distribution.  Defaults to 1.0.

    Returns:
        numpy.ndarray: The score for the input value x.
    """
    p_ = 0
    for i in range(-n, n + 1):
        p_ += (x + t * i) / sigma ** 2 * np.exp(-(x + t * i) ** 2 / 2 / sigma ** 2)
    return p_ / p_wrapped_normal(x, sigma, n, t)

def sigma_norm(sigma, t=1.0, sn=10000):
    r"""Monte-Carlo sampling for :math`\lambda_t`.
       See details in Appendix B.1 in the paper of DiffCSP.
    """
    sigmas = np.tile(sigma[None, :], (sn, 1))
    x_sample = sigma * np.random.standard_normal(sigmas.shape)
    x_sample = x_sample % t
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, t=t)
    return (normal_ ** 2).mean(axis=0)

def p_wrapped_normal_ms(x, sigma, n=10, t=1.0):
    """Utils for calcatating the score of wrapped normal distribution.
    """
    p_ = 0
    for i in range(-n, n + 1):
        p_ += ops.Exp()(-(x + t * i) ** 2 / 2 / sigma ** 2)
    return p_

def d_log_p_wrapped_normal_ms(x, sigma, n=10, t=1.0):
    """The score of wrapped normal distribution, which is parameterized by sigma,
       for the input value x. See details in Appendix B.1 in the paper of DiffCSP.

    Args:
        x (Tensor): Input noise.
        sigma (Tensor): The variance of wrapped normal distribution.
        n (int): The approximate parameter of the score of wrapped normal distribution. Defaults to 10.
        t (int): The period of wrapped normal distribution.  Defaults to 1.0.

    Returns:
        Tensor: The score for the input value x.
    """
    p_ = 0
    for i in range(-n, n + 1):
        p_ += (x + t * i) / sigma ** 2 * ops.Exp()(-(x + t * i) ** 2 / 2 / sigma ** 2)
    return p_ / p_wrapped_normal_ms(x, sigma, n, t)

class BetaScheduler(nn.Cell):
    """
    The alpha, alphas_cumprod and beta in DDPM used for lattice diffution.
    """
    def __init__(self, timesteps, scheduler_mode, beta_start=0.0001, beta_end=0.02):
        super(BetaScheduler, self).__init__()
        self.timesteps = Tensor(timesteps, mindspore.int32)
        self.timesteps_begin = Tensor(1, mindspore.int32)
        if scheduler_mode == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif scheduler_mode == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif scheduler_mode == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)

        betas = np.concatenate([np.zeros([1]), betas], axis=0)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        sigmas = np.zeros_like(betas)
        sigmas[1:] = betas[1:] * (1. - alphas_cumprod[:-1]) / (1. - alphas_cumprod[1:])
        sigmas = np.sqrt(sigmas)

        self.betas = Tensor(betas, mindspore.float32)
        self.alphas = Tensor(alphas, mindspore.float32)
        self.alphas_cumprod = Tensor(alphas_cumprod, mindspore.float32)
        self.sigmas = Tensor(sigmas, mindspore.float32)

    def uniform_sample_t(self, batch_size):
        return mindspore_random_choice(self.timesteps_begin, self.timesteps + 1, batch_size)

class SigmaScheduler(nn.Cell):
    r"""
    The sigmas and :math`\lambda_t` in SDEs used for fractional coordinates diffution.
    """
    def __init__(self, timesteps, sigma_begin=0.01, sigma_end=1.0):
        super(SigmaScheduler, self).__init__()
        self.timesteps = Tensor(timesteps, mindspore.int32)
        self.timesteps_begin = Tensor(1, mindspore.int32)
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), timesteps))
        sigmas_norm_ = sigma_norm(sigmas)

        sigmas = np.concatenate([np.zeros([1]), sigmas], axis=0)
        sigmas_norm = np.concatenate([np.ones([1]), sigmas_norm_], axis=0)

        self.sigmas = Tensor(sigmas, mindspore.float32)
        self.sigmas_norm = Tensor(sigmas_norm, mindspore.float32)

    def uniform_sample_t(self, batch_size):
        return mindspore_random_choice(self.timesteps_begin, self.timesteps + 1, batch_size)

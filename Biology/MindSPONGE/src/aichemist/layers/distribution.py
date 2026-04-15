# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Distribution
"""

from collections.abc import Sequence

import mindspore as ms
from mindspore import nn
from mindspore import numpy as mnp


class IndependentGaussian(nn.Cell):
    """
    Independent Gaussian distribution.

    Args:
        mu (Tensor): mean of shape :math:`(N,)`
        sigma2 (Tensor): variance of shape :math:`(N,)`
        learnable (bool, optional): learnable parameters or not
    """

    def __init__(self, mu, sigma2, learnable=False):
        super().__init__()
        if learnable:
            self.mu = ms.Parameter(mu)
            self.sigma2 = ms.Parameter(sigma2)
        else:
            self.mu = mu
            self.sigma2 = sigma2

        self.dim = len(mu)

    def construct(self, inputs):
        """
        Compute the likelihood of input data.

        Args:
            inputs (Tensor): input data of shape :math:`(..., N)`
        """
        log_likelihood = -0.5 * (mnp.log(2 * mnp.pi) + self.sigma2.log() + (inputs - self.mu) ** 2 / self.sigma2)
        return log_likelihood

    def sample(self, *size):
        """
        Draw samples from the distribution.

        Args:
            size (tuple of int): shape of the samples
        """
        if len(size) == 1 and isinstance(size[0], Sequence):
            size = size[0]
        size = list(size) + [self.dim]

        sample = mnp.randn(size) * mnp.sqrt(self.sigma2) + self.mu
        return sample

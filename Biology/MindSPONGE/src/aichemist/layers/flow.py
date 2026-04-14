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
Flow layers
"""

from mindspore import nn
from mindspore import numpy as mnp

from .common import MLP


class ConditionalFlow(nn.Cell):
    """
    Conditional flow transformation from `Masked Autoregressive Flow for Density Estimation`_.

    .. _Masked Autoregressive Flow for Density Estimation:
        https://arxiv.org/pdf/1705.07057.pdf

    Args:
        input_dim (int): input & output dimension
        condition_dim (int): condition dimension
        hidden_dims (list of int, optional): hidden dimensions
        activation (str or function, optional): activation function

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, condition_dim, hidden_dims=None, activation="ReLU"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        if hidden_dims is None:
            hidden_dims = []
        self.mlp = MLP(condition_dim, list(hidden_dims) + [input_dim * 2], activation)
        self.rescale = nn.Parameter(mnp.zeros(1))

    def construct(self, inputs, condition):
        """
        Transform data into latent representations.

        Args:
            inputs (Tensor): input representations
            condition (Tensor): conditional representations

        Returns:
            (Tensor, Tensor): latent representations, log-likelihood of the transformation
        """
        scale, bias = mnp.split(self.mlp(condition), 2, axis=-1)
        scale = mnp.tanh(scale) * self.rescale
        output = (inputs + bias) * mnp.exp(scale)
        log_det = scale
        return output, log_det

    def reverse(self, latent, condition):
        """
        Transform latent representations into data.

        Args:
            latent (Tensor): latent representations
            condition (Tensor): conditional representations

        Returns:
            (Tensor, Tensor): input representations, log-likelihood of the transformation
        """
        scale, bias = mnp.split(self.mlp(condition), 2, axis=-1)
        scale = mnp.tanh(scale) * self.rescale
        output = latent / mnp.exp(scale) - bias
        log_det = scale
        return output, log_det

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
Common layers
"""
import warnings
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, Initializer
from mindspore import Parameter
from ..configs import Config

from .common import MLP
from ..utils import scatter_mean
from ..utils import functional as F


class PairNorm(nn.Cell):
    """
    Pair normalization layer proposed in `PairNorm: Tackling Oversmoothing in GNNs`_.

    .. _PairNorm: Tackling Oversmoothing in GNNs:
        https://openreview.net/pdf?id=rkecl1rtwB

    Args:
        scale_individual (bool, optional): additionally normalize each node representation to have the same L2-norm

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    eps = 1e-8

    def __init__(self, scale_individual=False):
        super().__init__()
        self.scale_individual = scale_individual

    def construct(self, graph, inputs):
        """_summary_

        Args:
            graph (_type_): _description_
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        if graph.batch_size > 1:
            warnings.warn("PairNorm is proposed for a single graph, but now applied to a batch of graphs.")

        x = inputs.flatten(1)
        x = x - x.mean(axis=0)
        if self.scale_individual:
            output = x / (x.norm(axis=-1, keepdim=True) + self.eps)
        else:
            output = x * x.shape[0] ** 0.5 / (x.norm() + self.eps)
        return output.view_as(inputs)


class InstanceNorm(nn.Cell):
    """
    Instance normalization for graphs. This layer follows the definition in
    `GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training`.

    .. _GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training:
        https://arxiv.org/pdf/2009.03294.pdf

    Args:
        input_dim (int): input dimension
        eps (float, optional): epsilon added to the denominator
        affine (bool, optional): use learnable affine parameters or not

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, eps=1e-5, affine=False):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

    def construct(self, graph, inputs):
        """_summary_

        Args:
            graph (Graph): input Graph data
            inputs (ms.Tensor): input tensor data. Shape: (N, input_dim)

        Returns:
            output (ms.Tensor): output tensor data. Shape: (N, input_dim)
        """
        assert (graph.n_nodes >= 1).all()

        mean = scatter_mean(inputs, graph.node2graph, axis=0, n_axis=graph.batch_size)
        centered = inputs - mean[graph.node2graph]
        var = scatter_mean(centered ** 2, graph.node2graph, axis=0, n_axis=graph.batch_size)
        std = (var + self.eps).sqrt()
        output = centered / std[graph.node2graph]

        if self.affine:
            output = ops.Addcmul()(self.bias, self.weight, output)
        return output


class MutualInformation(nn.Cell):
    """
    Mutual information estimator from
    `Learning deep representations by mutual information estimation and maximization`_.

    .. _Learning deep representations by mutual information estimation and maximization:
        https://arxiv.org/pdf/1808.06670.pdf

    Args:
        input_dim (int): input dimension
        n_mlp_layer (int, optional): number of MLP layers
        activation (str or function, optional): activation function

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, n_mlp_layer=2, activation="relu"):
        super().__init__()
        self.x_mlp = MLP(input_dim, [input_dim] * n_mlp_layer, activation=activation)
        self.y_mlp = MLP(input_dim, [input_dim] * n_mlp_layer, activation=activation)

    def construct(self, x, y, pair_index=None):
        """_summary_

        Args:
            x (ms.Tensor): input data x
            y (ms.Tensor): input data y
            pair_index (ms.Tensor, optional): pair index data. Defaults to None.

        Returns:
            mutual_info (ms.Tensor): _description_
        """
        x = self.x_mlp(x)
        y = self.y_mlp(y)
        score = x @ y.t()
        score = score.flatten()

        if pair_index is None:
            assert len(x) == len(y)
            pair_index = ops.arange(len(x)).unsqueeze(-1).expand(-1, 2)

        index = pair_index[:, 0] * len(y) + pair_index[:, 1]
        positive = ops.zeros_like(score, dtype=bool)
        positive[index] = 1
        negative = ~positive

        mutual_info = - F.shifted_softplus(-score[positive]).mean() \
                      - F.shifted_softplus(score[negative]).mean()
        return mutual_info


class LayerNorm(nn.Cell):
    r"""Graph normalization

    Args:
        dim_feature (int):          Feature dimension

        axis (int):                 Axis to normalize. Default: -2

        alpha_init (initializer):   initializer for alpha. Default: 'one'

        beta_init (initializer):    initializer for beta. Default: 'zero'

        gamma_init (initializer):   initializer for alpha. Default: 'one'

    """

    def __init__(self,
                 n_feature: int,
                 axis: int = -2,
                 alpha_init: initializer = 'one',
                 beta_init: initializer = 'zero',
                 gamma_init: initializer = 'one',
                 eps: float = 1e-05,
                 use_alpha: bool = False,
                 ):

        super().__init__()
        self.alpha = initializer(alpha_init, n_feature)
        if use_alpha:
            self.alpha = Parameter(self.alpha, name='alpha')
        self.beta = Parameter(initializer(beta_init, n_feature), name="beta")
        self.gamma = Parameter(initializer(gamma_init, n_feature), name="gamma")
        self.eps = eps
        self.axis = axis

    def construct(self, nodes: ms.Tensor):
        """Compute graph normalization.

        Args:
            nodes (Tensor):     Tensor with shape (B, A, N, F). Data type is float.

        Returns:
            output (Tensor):    Tensor with shape (B, A, N, F). Data type is float.

        """
        mu = nodes.mean(axis=self.axis)
        std = (nodes * self.alpha).std(axis=self.axis)
        y = self.gamma * (nodes - self.alpha * mu) / std + self.beta
        return y


class GraphNorm(nn.Cell):
    r"""Graph normalization

    Args:
        dim_feature (int):          Feature dimension

        axis (int):                 Axis to normalize. Default: -2

        alpha_init (Initializer):   Initializer for alpha. Default: 'one'

        beta_init (Initializer):    Initializer for beta. Default: 'zero'

        gamma_init (Initializer):   Initializer for alpha. Default: 'one'

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim_feature: int,
                 axis: int = -2,
                 alpha_init: Initializer = 'one',
                 beta_init: Initializer = 'zero',
                 gamma_init: Initializer = 'one',
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        dim_feature = int(dim_feature)

        self.alpha = Parameter(initializer(alpha_init, dim_feature), name="alpha")
        self.beta = Parameter(initializer(beta_init, dim_feature), name="beta")
        self.gamma = Parameter(initializer(gamma_init, dim_feature), name="gamma")

        self.axis = int(axis)

    def construct(self, nodes: ms.Tensor):
        """Compute graph normalization.

        Args:
            nodes (Tensor):     Tensor with shape (B, A, N, F). Data type is float.

        Returns:
            output (Tensor):    Tensor with shape (B, A, N, F). Data type is float.

        """

        mu = ops.mean(nodes, self.axis, keep_dims=True)

        nodes2 = nodes * nodes
        mu2 = ops.mean(nodes2, self.axis, keep_dims=True)

        a = self.alpha
        sigma2 = mu2 + (a*a - 2*a) * mu * mu
        sigma = F.sqrt(sigma2)

        y = self.gamma * (nodes - a * mu) / sigma + self.beta

        return y


class CoordsNorm(nn.Cell):
    """Normalization of coordinate


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = ops.zeros(1).fill(scale_init)
        self.scale = ms.Parameter(scale)

    def construct(self, coords):
        """_summary_

        Args:
            coords (_type_): _description_

        Returns:
            _type_: _description_
        """
        norm = coords.norm(axis=-1, keep_dims=True)
        return coords * self.scale / (norm + self.eps)

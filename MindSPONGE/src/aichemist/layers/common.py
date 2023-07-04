# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
import inspect
from collections.abc import Sequence

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer
from mindspore import Parameter
from ..util import scatter_mean
from ..util import functional as F


class MLP(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim: int,
                 hidden_dims: Sequence,
                 short_cut=False,
                 batch_norm=False,
                 activation=nn.ReLU(),
                 dropout=0):
        super().__init__()
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)
        else:
            self.activation = activation
        self.dropout = nn.Dropout(dropout)
        fcs = [nn.Dense(dim, self.dims[i+1]) for i, dim in enumerate(self.dims[:-1])]
        self.layers = nn.CellList(fcs)
        if batch_norm:
            bns = [nn.BatchNorm1d(dim) for dim in self.dims[1:-1]]
            self.bns = nn.CellList(bns)

    def construct(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        hidden = inputs
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
                if self.short_cut and hidden.shape == hidden.shape:
                    hidden += inputs
        return hidden


class GaussianSmearing(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, start=0, end=5, n_kernel=100, centered=False, learnable=False):
        super().__init__()
        if centered:
            self.mu = ops.zeros(n_kernel)
            self.sigma = ops.linspace(start, end, n_kernel)
        else:
            self.mu = ops.linspace(start, end, n_kernel)
            self.sigma = ops.ones(n_kernel) * (self.mu[1] - self.mu[0])

        if learnable:
            self.mu = ms.Parameter(self.mu)
            self.sigma = ms.Parameter(self.sigma)

    def construct(self, dist):
        """
        Compute smeared gaussian features between data.

        Parameters:
            x (Tensor): data of shape :math:`(..., d)`
            y (Tensor): data of shape :math:`(..., d)`
        Returns:
            Tensor: features of shape :math:`(..., num_kernel)`
        """
        z = (dist.view(-1, 1) - self.mu) / self.sigma
        prob = ops.exp(-0.5 * z * z)
        return prob


class PairNorm(nn.Cell):
    """
    Pair normalization layer proposed in `PairNorm: Tackling Oversmoothing in GNNs`_.

    .. _PairNorm: Tackling Oversmoothing in GNNs:
        https://openreview.net/pdf?id=rkecl1rtwB

    Parameters:
        scale_individual (bool, optional): additionally normalize each node representation to have the same L2-norm
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

    Parameters:
        input_dim (int): input dimension
        eps (float, optional): epsilon added to the denominator
        affine (bool, optional): use learnable affine parameters or not
    """

    def __init__(self, input_dim, eps=1e-5, affine=False):
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps
        self.affine = affine

    def construct(self, graph, inputs):
        """_summary_

        Args:
            graph (_type_): _description_
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert (graph.n_nodes >= 1).all()

        mean = scatter_mean(inputs, graph.node2graph(), axis=0, n_axis=graph.batch_size)
        centered = inputs - mean[graph.node2graph()]
        var = scatter_mean(centered ** 2, graph.node2graph(), axis=0, n_axis=graph.batch_size)
        std = (var + self.eps).sqrt()
        output = centered / std[graph.node2graph()]

        if self.affine:
            output = ops.Addcmul()(self.bias, self.weight, output)
        return output


class MutualInformation(nn.Cell):
    """
    Mutual information estimator from
    `Learning deep representations by mutual information estimation and maximization`_.

    .. _Learning deep representations by mutual information estimation and maximization:
        https://arxiv.org/pdf/1808.06670.pdf

    Parameters:
        input_dim (int): input dimension
        n_mlp_layer (int, optional): number of MLP layers
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, n_mlp_layer=2, activation="relu"):
        super().__init__()
        self.x_mlp = MLP(input_dim, [input_dim] * n_mlp_layer, activation=activation)
        self.y_mlp = MLP(input_dim, [input_dim] * n_mlp_layer, activation=activation)

    def construct(self, x, y, pair_index=None):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
            pair_index (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
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


class Sequential(nn.SequentialCell):
    """
    Improved sequential container.
    Modules will be called in the order they are passed to the constructor.

    Compared to the vanilla nn.Sequential, this layer additionally supports the following features.

    1. Multiple input / output arguments.

    >>> # layer1 signature: (...) -> (a, b)
    >>> # layer2 signature: (a, b) -> (...)
    >>> layer = layers.Sequential(layer1, layer2)

    2. Global arguments.

    >>> # layer1 signature: (graph, a) -> b
    >>> # layer2 signature: (graph, b) -> c
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))

    Note the global arguments don't need to be present in every layer.

    >>> # layer1 signature: (graph, a) -> b
    >>> # layer2 signature: b -> c
    >>> # layer3 signature: (graph, c) -> d
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))

    3. Dict outputs.

    >>> # layer1 signature: a -> {"b": b, "c": c}
    >>> # layer2 signature: b -> d
    >>> layer = layers.Sequential(layer1, layer2, allow_unused=True)

    When dict outputs are used with global arguments, the global arguments can be explicitly
    overwritten by any layer outputs.

    >>> # layer1 signature: (graph, a) -> {"graph": graph, "b": b}
    >>> # layer2 signature: (graph, b) -> c
    >>> # layer2 takes in the graph output by layer1
    >>> layer = layers.Sequential(layer1, layer2, global_args=("graph",))
    """

    def __init__(self, *args, global_args=None, allow_unused=False):
        super().__init__(*args)
        if global_args is not None:
            self.global_args = set(global_args)
        else:
            self.global_args = {}
        self.allow_unused = allow_unused

    def construct(self, *args, **kwargs):
        """_summary_

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        global_kwargs = {}
        for module in self._modules.values():
            sig = inspect.signature(module.construct)
            parameters = list(sig.parameters.values())
            param_names = [param.name for param in parameters]
            j = 0
            for name in param_names:
                if j == len(args):
                    break
                if name in kwargs:
                    continue
                if name in global_kwargs and name not in kwargs:
                    kwargs[name] = global_kwargs.get(name)
                    continue
                kwargs[name] = args[j]
                j += 1
            if self.allow_unused:
                param_names = set(param_names)
                # pop unused kwargs
                kwargs = {k: v for k, v in kwargs.items() if k in param_names}
            if j < len(args):
                raise TypeError("too many positional arguments")

            output = module(**kwargs)

            global_kwargs.update({k: v for k, v in kwargs.items() if k in self.global_args})
            args = []
            kwargs = {}
            if isinstance(output, dict):
                kwargs.update(output)
            elif isinstance(output, Sequence):
                args += list(output)
            else:
                args.append(output)
        return output


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
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.is_node = is_node
        self.affine = affine

        self.gamma = Parameter(ops.ones(self.num_features), 'gamma')
        self.beta = Parameter(ops.zeros(self.num_features), 'beta')

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def construct(self, h, graph_size):
        """_summary_

        Args:
            h (_type_): _description_
            graph_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_list = ops.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = ops.concat(norm_list, 0)
        return self.gamma * norm_x + self.beta


class CoordsNorm(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
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


def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes):
    """_summary_

    Args:
        ligand_batch_num_nodes (_type_): _description_
        receptor_batch_num_nodes (_type_): _description_

    Returns:
        _type_: _description_
    """
    rows = ligand_batch_num_nodes.sum()
    cols = receptor_batch_num_nodes.sum()
    mask = ops.zeros(rows, cols)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask

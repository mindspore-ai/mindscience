# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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
"""aggregators"""

import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from .blocks import MLP, Dense
from .base import SoftmaxWithMask
from .base import MultiheadAttention

__all__ = [
    "Aggregator",
    "get_aggregator",
    "TensorSummation",
    "TensorMean",
    "SoftmaxGeneralizedAggregator",
    "PowermeanGeneralizedAggregator",
    "ListAggregator",
    "get_list_aggregator",
    "ListSummation",
    "ListMean",
    "LinearTransformation",
    "MultipleChannelRepresentation",
]

_AGGREGATOR_ALIAS = dict()
_LIST_AGGREGATOR_ALIAS = dict()


def _aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _AGGREGATOR_ALIAS:
            _AGGREGATOR_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _AGGREGATOR_ALIAS:
                _AGGREGATOR_ALIAS[alias] = cls

        return cls

    return alias_reg


def _list_aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _LIST_AGGREGATOR_ALIAS:
            _LIST_AGGREGATOR_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _LIST_AGGREGATOR_ALIAS:
                _LIST_AGGREGATOR_ALIAS[alias] = cls

        return cls

    return alias_reg


class Aggregator(nn.Cell):
    def __init__(self, dim=None, axis=-2,):
        super().__init__()

        self.name = 'aggregator'

        self.dim = dim
        self.axis = axis
        self.reduce_sum = P.ReduceSum()


class ListAggregator(nn.Cell):
    """list aggretor"""
    def __init__(self, dim=None, num_agg=None, n_hidden=0, activation=None,):
        super().__init__()

        self.dim = dim
        self.num_agg = num_agg
        self.n_hidden = n_hidden
        self.activation = activation

        self.stack = P.Stack(-1)
        self.reduce_sum = P.ReduceSum()


@_aggregator_register('sum')
class TensorSummation(Aggregator):
    """tensor summation"""
    def __init__(self, dim=None, axis=-2,):
        super().__init__(dim=None, axis=axis,)

        self.name = 'sum'

    def __str__(self):
        return "sum"

    def construct(self, nodes, node_mask=None):
        if node_mask is not None:
            nodes = nodes * node_mask
        agg = self.reduce_sum(nodes, self.axis)
        return agg


@_aggregator_register('mean')
class TensorMean(Aggregator):
    """tensor mean"""
    def __init__(self, dim=None, axis=-2,):
        super().__init__(dim=None, axis=axis,)
        self.name = 'mean'

        self.reduce_mean = P.ReduceMean()
        self.mol_sum = P.ReduceSum(keep_dims=True)

    def __str__(self):
        return "mean"

    def construct(self, nodes, node_mask=None, nodes_number=None):
        if node_mask is not None:
            nodes = nodes * node_mask
            agg = self.reduce_sum(nodes, self.axis)
            return agg / nodes_number
        return self.reduce_mean(nodes, self.axis)


# Softmax-based generalized mean-max-sum aggregator
@_aggregator_register('softmax')
class SoftmaxGeneralizedAggregator(Aggregator):
    """softmax generalized aggregator"""
    def __init__(self, dim, axis=-2,):
        super().__init__(dim=dim, axis=axis,)

        self.name = 'softmax'

        self.beta = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('zero', 1), name="rho")

        self.softmax = P.Softmax(axis=self.axis)
        self.softmax_with_mask = SoftmaxWithMask(axis=self.axis)
        self.mol_sum = P.ReduceSum(keep_dims=True)

        self.expand_ones = P.Ones()((1, 1, self.dim), ms.int32)

    def __str__(self):
        return "softmax"

    def construct(self, nodes, node_mask=None, nodes_number=None):
        """construct"""
        if nodes_number is None:
            nodes_number = nodes.shape[self.axis]

        scale = nodes_number / (1 + self.beta * (nodes_number - 1))
        px = nodes * self.rho

        if node_mask is None:
            agg_nodes = self.softmax(px) * nodes
        else:
            mask = (self.expand_ones * node_mask) > 0
            agg_nodes = self.softmax_with_mask(px, mask) * nodes * node_mask

        agg_nodes = self.reduce_sum(agg_nodes, self.axis)

        return scale * agg_nodes


# PowerMean-based generalized mean-max-sum aggregator
@_aggregator_register('powermean')
class PowermeanGeneralizedAggregator(Aggregator):
    """power mean generalized aggregator"""
    def __init__(self, dim, axis=-2,):
        super().__init__(dim=dim, axis=axis,)
        self.name = 'powermean'
        self.beta = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('one', 1), name="rho")

        self.power = P.Pow()
        self.mol_sum = P.ReduceSum(keep_dims=True)

    def __str__(self):
        return "powermean"

    def construct(self, nodes, node_mask=None, nodes_number=None):
        """construct"""
        if nodes_number is None:
            nodes_number = nodes.shape[self.axis]

        scale = nodes_number / (1 + self.beta * (nodes_number - 1))
        xp = self.power(nodes, self.rho)
        if node_mask is not None:
            xp = xp * node_mask
        agg_nodes = self.reduce_sum(xp, self.axis)

        return self.power(scale * agg_nodes, 1.0 / self.rho)


@_aggregator_register('transformer')
class TransformerAggregator(Aggregator):
    """trasnformer aggregator"""
    def __init__(self, dim, axis=-2, n_heads=8,):
        super().__init__(
            dim=dim,
            axis=axis,
        )

        self.name = 'transformer'

        self.a2q = Dense(dim, dim, has_bias=False)
        self.a2k = Dense(dim, dim, has_bias=False)
        self.a2v = Dense(dim, dim, has_bias=False)

        self.layer_norm = nn.LayerNorm((dim,), -1, -1)

        self.multi_head_attention = MultiheadAttention(
            dim, n_heads, dim_tensor=3)

        self.squeeze = P.Squeeze(-1)
        self.mean = TensorMean(dim, axis)

    def __str__(self):
        return "transformer"

    def construct(self, nodes, node_mask=None, nodes_number=None):
        r"""Transformer type aggregator.

        Args:
            nodes (Mindspore.Tensor[float] [B, A, F]):

        Returns:
            Mindspore.Tensor [..., F]: multi-head attention output.

        """
        # [B, A, F]
        x = self.layer_norm(nodes)

        # [B, A, F]
        q = self.a2q(x)
        k = self.a2k(x)
        v = self.a2v(x)

        if node_mask is not None:
            mask = self.squeeze(node_mask)
        else:
            mask = node_mask

        # [B, A, F]
        x = self.multi_head_attention(q, k, v, mask)

        # [B, 1, F]
        return self.mean(x, node_mask, nodes_number)


@_list_aggregator_register('sum')
class ListSummation(ListAggregator):
    """list summation"""
    def __init__(self,
                 dim=None,
                 num_agg=None,
                 n_hidden=0,
                 activation=None,
                 ):
        super().__init__(
            dim=None,
            num_agg=None,
            n_hidden=0,
            activation=None,
        )

    def __str__(self):
        return "sum"

    def construct(self, xlist, node_mask=None):
        xt = self.stack(xlist)
        y = self.reduce_sum(xt, -1)
        if node_mask is not None:
            y = y * node_mask
        return y


@_list_aggregator_register('mean')
class ListMean(ListAggregator):
    """list mean"""
    def __init__(self, dim=None, num_agg=None, n_hidden=0, activation=None,):
        super().__init__(
            dim=None,
            num_agg=None,
            n_hidden=0,
            activation=None,
        )

        self.reduce_mean = P.ReduceMean()

    def __str__(self):
        return "mean"

    def construct(self, xlist, node_mask=None):
        xt = self.stack(xlist)
        y = self.reduce_mean(xt, -1)
        if node_mask is not None:
            y = y * node_mask
        return y


@_list_aggregator_register('linear')
class LinearTransformation(ListAggregator):
    """linear transformation"""
    def __init__(self,
                 dim,
                 num_agg=None,
                 n_hidden=0,
                 activation=None,
                 ):
        super().__init__(
            dim=dim,
            num_agg=None,
            n_hidden=0,
            activation=None,
        )
        self.scale = ms.Parameter(
            initializer(
                Normal(1.0), [self.dim,]), name="scale")
        self.shift = ms.Parameter(
            initializer(
                Normal(1.0), [self.dim,]), name="shift")

    def __str__(self):
        return "linear"

    def construct(self, ylist, node_mask=None):
        yt = self.stack(ylist)
        ysum = self.reduce_sum(yt, -1)
        y = self.scale * ysum + self.shift
        if node_mask is not None:
            y = y * node_mask
        return y

# Multiple-Channel Representation Readout


@_list_aggregator_register('mcr')
class MultipleChannelRepresentation(ListAggregator):
    """multiple channel representation"""
    def __init__(self,
                 dim,
                 num_agg,
                 n_hidden=0,
                 activation=None,
                 ):
        super().__init__(
            dim=dim,
            num_agg=num_agg,
            n_hidden=n_hidden,
            activation=activation,
        )

        sub_dim = self.dim // self.num_agg
        last_dim = self.dim - (sub_dim * (self.num_agg - 1))
        sub_dims = [sub_dim for _ in range(self.num_agg - 1)]
        sub_dims.append(last_dim)

        if self.n_hidden > 0:
            hidden_layers = [dim,] * self.n_hidden
            self.mcr = nn.CellList([
                MLP(self.dim, sub_dims[i], hidden_layers, activation=self.activation)
                for i in range(self.um_agg)
            ])
        else:
            self.mcr = nn.CellList([
                Dense(self.dim, sub_dims[i], activation=self.activation)
                for i in range(self.num_agg)
            ])

        self.concat = P.Concat(-1)

    def __str__(self):
        return "MCR"

    def construct(self, xlist, node_mask=None):
        xt = ()
        for i in range(self.num_agg):
            xt = xt + (self.mcr[i](xlist[i]),)
        y = self.concat(xt)
        if node_mask is not None:
            y = y * node_mask
        return y


def get_aggregator(obj, dim, axis=-2):
    if obj is None or isinstance(obj, Aggregator):
        return obj
    if isinstance(obj, str):
        if obj.lower() not in _AGGREGATOR_ALIAS.keys():
            raise ValueError(
                "The class corresponding to '{}' was not found.".format(obj))
        return _AGGREGATOR_ALIAS[obj.lower()](dim=dim, axis=axis)
    raise TypeError("Unsupported Aggregator type '{}'.".format(type(obj)))


def get_list_aggregator(obj, dim, num_agg, n_hidden=0, activation=None,):
    """get list aggregator"""
    if obj is None or isinstance(obj, ListAggregator):
        return obj
    if isinstance(obj, str):
        if obj.lower() not in _LIST_AGGREGATOR_ALIAS.keys():
            raise ValueError(
                "The class corresponding to '{}' was not found.".format(obj))
        return _LIST_AGGREGATOR_ALIAS[obj.lower()](
            dim=dim,
            num_agg=num_agg,
            n_hidden=n_hidden,
            activation=activation,
        )
    raise TypeError("Unsupported ListAggregator type '{}'.".format(type(obj)))

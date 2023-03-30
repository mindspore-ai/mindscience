# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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
Aggregator for readout network
"""

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from mindsponge.function import get_integer

from .block import MLP, Dense
from .base import SoftmaxWithMask
from .base import MultiheadAttention

__all__ = [
    "Aggregator",
    "get_aggregator",
    "get_interaction_aggregator",
    "TensorSummation",
    "TensorMean",
    "SoftmaxGeneralizedAggregator",
    "PowermeanGeneralizedAggregator",
    "InteractionAggregator",
    "InteractionSummation",
    "InteractionMean",
    "LinearTransformation",
    "MultipleChannelRepresentation",
]

_AGGREGATOR_BY_KEY = dict()
_INTERACTION_AGGREGATOR_BY_KEY = dict()


def _aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _AGGREGATOR_BY_KEY:
            _AGGREGATOR_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _AGGREGATOR_BY_KEY:
                _AGGREGATOR_BY_KEY[alias] = cls

        return cls

    return alias_reg


def _interaction_aggregator_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _INTERACTION_AGGREGATOR_BY_KEY:
            _INTERACTION_AGGREGATOR_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _INTERACTION_AGGREGATOR_BY_KEY:
                _INTERACTION_AGGREGATOR_BY_KEY[alias] = cls

        return cls

    return alias_reg


class Aggregator(nn.Cell):
    r"""Network to aggregate the outputs of each atoms.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2
                 ):

        super().__init__()

        self.reg_key = 'aggregator'
        self.name = 'aggregator'

        self.dim = get_integer(dim)
        self.axis = get_integer(axis)
        self.reduce_sum = P.ReduceSum()

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """Aggregate the outputs of each atoms.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        raise NotImplementedError


class InteractionAggregator(nn.Cell):
    r"""Network to aggregate the representation of each interaction layer.

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self,
                 dim: int,
                 num_agg: int = None,
                 n_hidden: int = 0,
                 activation: Cell = None
                 ):

        super().__init__()

        self.reg_key = 'none'

        self.dim = dim
        self.num_agg = num_agg
        self.n_hidden = n_hidden
        self.activation = activation

        self.stack = P.Stack(-1)
        self.reduce_sum = P.ReduceSum()

    def construct(self, ylist: list, atom_mask: Tensor = None):
        """Aggregate the representations of each interaction layer.

        Args:
            ylist (list):       List of representation of interactions layers.
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        raise NotImplementedError


@_aggregator_register('sum')
class TensorSummation(Aggregator):
    r"""A aggregator to sum all the tensors.

    Args:

        dim (int):  Feature dimension. Default: None

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int = None,
                 axis: int = -2
                 ):

        super().__init__(
            dim=dim,
            axis=axis
        )

        self.reg_key = 'sum'
        self.name = 'sum'

    def __str__(self):
        return "sum"

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To sum all tensors.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if atom_mask is not None:
            # (B,A,X) * (B,A,1)
            nodes = nodes * F.expand_dims(atom_mask, -1)
        # (B,X) <- (B,A,X)
        agg = self.reduce_sum(nodes, self.axis)
        return agg


@_aggregator_register('mean')
class TensorMean(Aggregator):
    r"""A aggregator to average all the tensors.

    Args:

        dim (int):  Feature dimension. Default: None

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int = None,
                 axis: int = -2
                 ):

        super().__init__(
            dim=dim,
            axis=axis
        )

        self.reg_key = 'mean'
        self.name = 'mean'

        self.reduce_mean = P.ReduceMean()
        self.mol_sum = P.ReduceSum(True)

    def __str__(self):
        return "mean"

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To average all tensors.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if atom_mask is None:
            # (B,X) <- (B,A,X)
            return self.reduce_mean(nodes, self.axis)

        # (B,A,X) * (B,A,1)
        nodes = nodes * F.expand_dims(atom_mask, -1)
        # (B,X) <- (B,A,X)
        agg = self.reduce_sum(nodes, self.axis)
        # (B,X) / (B,1)
        return agg / num_atoms


@_aggregator_register('softmax')
class SoftmaxGeneralizedAggregator(Aggregator):
    r"""A Softmax-based generalized mean-max-sum aggregator.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2
                 ):

        super().__init__(
            dim=dim,
            axis=axis
        )

        self.reg_key = 'softmax'
        self.name = 'softmax'

        self.beta = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('zero', 1), name="rho")

        self.softmax = P.Softmax(axis=self.axis)
        self.softmax_with_mask = SoftmaxWithMask(axis=self.axis)
        self.mol_sum = P.ReduceSum(True)

        self.expand_ones = P.Ones()((1, 1, self.dim), ms.int32)

    def __str__(self):
        return "softmax"

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To aggregate all tensors using softmax-based generalized mean-max-sum.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if num_atoms is None:
            num_atoms = nodes.shape[self.axis]

        scale = num_atoms / (1 + self.beta * (num_atoms - 1))
        px = nodes * self.rho

        if atom_mask is None:
            agg_nodes = self.softmax(px) * nodes
        else:
            atom_mask = F.expand_dims(atom_mask, -1)
            mask = (self.expand_ones * atom_mask) > 0
            agg_nodes = self.softmax_with_mask(px, mask) * nodes * atom_mask

        agg_nodes = self.reduce_sum(agg_nodes, self.axis)

        return scale * agg_nodes


@_aggregator_register('powermean')
class PowermeanGeneralizedAggregator(Aggregator):
    r"""A PowerMean-based generalized mean-max-sum aggregator.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2
                 ):

        super().__init__(
            dim=dim,
            axis=axis
        )

        self.reg_key = 'powermean'
        self.name = 'powermean'
        self.beta = ms.Parameter(initializer('one', 1), name="beta")
        self.rho = ms.Parameter(initializer('one', 1), name="rho")

        self.power = P.Pow()
        self.mol_sum = P.ReduceSum(True)

    def __str__(self):
        return "powermean"

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To aggregate all tensors using PowerMean-based generalized mean-max-sum.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        if num_atoms is None:
            num_atoms = nodes.shape[self.axis]

        scale = num_atoms / (1 + self.beta * (num_atoms - 1))
        xp = self.power(nodes, self.rho)
        if atom_mask is not None:
            xp = xp * F.expand_dims(atom_mask, -1)
        agg_nodes = self.reduce_sum(xp, self.axis)

        return self.power(scale*agg_nodes, 1.0/self.rho)


@_aggregator_register('transformer')
class TransformerAggregator(Aggregator):
    r"""A transformer-type aggregator.

    Args:

        dim (int):  Feature dimension.

        axis (int): Axis to aggregate. Default: -2

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        O:  Last dimension of the tensor for each node (atom).

    """

    def __init__(self,
                 dim: int,
                 axis: int = -2,
                 n_heads: int = 8
                 ):

        super().__init__(
            dim=dim,
            axis=axis
        )

        self.reg_key = 'transformer'
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

    def construct(self, nodes: Tensor, atom_mask: Tensor = None, num_atoms: Tensor = None):
        """To aggregate all tensors using PowerMean-based generalized mean-max-sum.

        Args:
            nodes (Tensor):     Tensor of shape (B, A, X). Data type is float.
                                Output vectors for each atom (node).
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.
                                Default: None
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms.
                                Default: None

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        #pylint: disable=invalid-name

        # [B, A, F]
        x = self.layer_norm(nodes)

        # [B, A, F]
        Q = self.a2q(x)
        K = self.a2k(x)
        V = self.a2v(x)

        # [B, A, F]
        x = self.multi_head_attention(Q, K, V, atom_mask)

        # [B, 1, F]
        return self.mean(x, atom_mask, num_atoms)


@_interaction_aggregator_register('sum')
class InteractionSummation(InteractionAggregator):
    r"""A interaction aggregator to summation all representations of interaction layers

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self,
                 dim: int = None,
                 num_agg: int = None,
                 n_hidden: int = 0,
                 activation: Cell = None
                 ):

        super().__init__(
            dim=None,
            num_agg=None,
            n_hidden=0,
            activation=None
        )

        self.reg_key = 'sum'

    def __str__(self):
        return "sum"

    def construct(self, ylist, atom_mask=None):
        xt = self.stack(ylist)
        y = self.reduce_sum(xt, -1)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@_interaction_aggregator_register('mean')
class InteractionMean(InteractionAggregator):
    r"""A interaction aggregator to average all representations of interaction layers

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self,
                 dim: int = None,
                 num_agg: int = None,
                 n_hidden: int = 0,
                 activation: Cell = None
                 ):

        super().__init__(
            dim=None,
            num_agg=None,
            n_hidden=0,
            activation=None
        )

        self.reg_key = 'mean'
        self.reduce_mean = P.ReduceMean()

    def __str__(self):
        return "mean"

    def construct(self, ylist: list, atom_mask: Tensor = None):
        xt = self.stack(ylist)
        y = self.reduce_mean(xt, -1)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@_interaction_aggregator_register('linear')
class LinearTransformation(InteractionAggregator):
    r"""A interaction aggregator to aggregate all representations of interaction layers
        by using linear transformation

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self,
                 dim: int = None,
                 num_agg: int = None,
                 n_hidden: int = 0,
                 activation: Cell = None
                 ):

        super().__init__(
            dim=None,
            num_agg=None,
            n_hidden=0,
            activation=None
        )

        self.reg_key = 'linear'

        self.scale = ms.Parameter(initializer(
            Normal(1.0), [self.dim]), name="scale")
        self.shift = ms.Parameter(initializer(
            Normal(1.0), [self.dim]), name="shift")

    def __str__(self):
        return "linear"

    def construct(self, ylist: list, atom_mask: Tensor = None):
        yt = self.stack(ylist)
        ysum = self.reduce_sum(yt, -1)
        y = self.scale * ysum + self.shift
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@_interaction_aggregator_register('mcr')
class MultipleChannelRepresentation(InteractionAggregator):
    r"""A Multiple-Channel Representation (MCR) interaction aggregator to
        aggregate all representations of interaction layers

    Args:

        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: None

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: None

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self,
                 dim: int,
                 num_agg: int,
                 n_hidden: int = 0,
                 activation: Cell = None
                 ):

        super().__init__(
            dim=dim,
            num_agg=num_agg,
            n_hidden=n_hidden,
            activation=activation
        )

        self.reg_key = 'mcr'

        sub_dim = self.dim // self.num_agg
        last_dim = self.dim - (sub_dim * (self.num_agg - 1))
        sub_dims = [sub_dim for _ in range(self.num_agg - 1)]
        sub_dims.append(last_dim)

        if self.n_hidden > 0:
            hidden_layers = [dim] * self.n_hidden
            self.mcr = nn.CellList([
                MLP(self.dim, sub_dims[i], hidden_layers,
                    activation=self.activation)
                for i in range(self.num_agg)
            ])
        else:
            self.mcr = nn.CellList([
                Dense(self.dim, sub_dims[i], activation=self.activation)
                for i in range(self.num_agg)
            ])

        self.concat = P.Concat(-1)

    def __str__(self):
        return "MCR"

    def construct(self, ylist: list, atom_mask: Tensor = None):
        readouts = ()
        for i in range(self.num_agg):
            readouts = readouts + (self.mcr[i](ylist[i]),)
        y = self.concat(readouts)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


_AGGREGATOR_BY_NAME = {
    agg.__name__: agg for agg in _AGGREGATOR_BY_KEY.values()}

_INTERACTION_AGGREGATOR_BY_NAME = {
    agg.__name__: agg for agg in _INTERACTION_AGGREGATOR_BY_KEY.values()}


def get_aggregator(aggregator: Aggregator,
                   dim: int,
                   axis: int = -2
                   ) -> Aggregator:
    """get aggregator by name"""
    if aggregator is None or isinstance(aggregator, Aggregator):
        return aggregator
    if isinstance(aggregator, str):
        if aggregator.lower() == 'none':
            return None
        if aggregator.lower() in _AGGREGATOR_BY_KEY.keys():
            return _AGGREGATOR_BY_KEY[aggregator.lower()](dim=dim, axis=axis)
        if aggregator in _AGGREGATOR_BY_NAME.keys():
            return _AGGREGATOR_BY_NAME[aggregator](dim=dim, axis=axis)
        raise ValueError(
            "The Aggregator corresponding to '{}' was not found.".format(aggregator))
    raise TypeError(
        "Unsupported Aggregator type '{}'.".format(type(aggregator)))


def get_interaction_aggregator(aggregator: InteractionAggregator,
                               dim: int,
                               axis: int = -2
                               ) -> InteractionAggregator:
    """get aggregator by name"""
    if aggregator is None or isinstance(aggregator, InteractionAggregator):
        return aggregator
    if isinstance(aggregator, str):
        if aggregator.lower() == 'none':
            return None
        if aggregator.lower() in _INTERACTION_AGGREGATOR_BY_KEY.keys():
            return _INTERACTION_AGGREGATOR_BY_KEY[aggregator.lower()](dim=dim, axis=axis)
        if aggregator in _INTERACTION_AGGREGATOR_BY_NAME.keys():
            return _INTERACTION_AGGREGATOR_BY_NAME[aggregator](dim=dim, axis=axis)
        raise ValueError(
            "The Interaction Aggregator corresponding to '{}' was not found.".format(aggregator))
    raise TypeError(
        "Unsupported Interaction Aggregator type '{}'.".format(type(aggregator)))

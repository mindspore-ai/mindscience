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
Aggregator for node vector
"""

from typing import List

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from ...configs import Config

from ..residuals import MLP
from ...configs import Registry as R


__all__ = [
    "InteractionAggregator",
    "InteractionSummation",
    "InteractionMean",
    "LinearTransformation",
    "MultipleChannelRepresentation",
]


class InteractionAggregator(nn.Cell):
    r"""Network to aggregate the representation of each interaction layer.

    Args:
        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: ``None``.

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        """Aggregate the representations of each interaction layer.

        Args:
            ylist (list):       List of representation of interactions layers.
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool.
                                Mask for atoms.Default: ``None``.

        Returns:
            output (Tensor):    Tensor of shape (B, X). Data type is float.

        """
        raise NotImplementedError


@R.register('aggregator.interaction.sum')
class InteractionSummation(InteractionAggregator):
    r"""A interaction aggregator to summation all representations of interaction layers

    Args:
        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: ``None``.

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def __str__(self):
        return "sum"

    def construct(self, ylist, atom_mask=None):
        xt = ops.stack(ylist, axis=-1)
        y = F.reduce_sum(xt, -1)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@R.register('aggregator.interaction.mean')
class InteractionMean(InteractionAggregator):
    r"""A interaction aggregator to average all representations of interaction layers

    Args:
        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: ``None``.

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def __str__(self):
        return "mean"

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        xt = ops.stack(ylist, axis=-1)
        y = F.reduce_mean(xt, -1)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@R.register('aggregator.interaction.linear')
class LinearTransformation(InteractionAggregator):
    r"""A interaction aggregator to aggregate all representations of interaction layers
        by using linear transformation

    Args:
        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: ``None``.

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.scale = ms.Parameter(initializer(Normal(1.0), [self.dim]), name="scale")
        self.shift = ms.Parameter(initializer(Normal(1.0), [self.dim]), name="shift")

    def __str__(self):
        return "linear"

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        yt = ops.stack(ylist, axis=-1)
        ysum = F.reduce_sum(yt, -1)
        y = self.scale * ysum + self.shift
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y


@R.register('aggregator.interaction.mcr')
class MultipleChannelRepresentation(InteractionAggregator):
    r"""A Multiple-Channel Representation (MCR) interaction aggregator to
        aggregate all representations of interaction layers

    Args:
        dim (int):          Feature dimension.

        num_agg (int):      Number of interaction layer to be aggregate. Default: ``None``.

        n_hidden (int):     Number of hidden layers. Default: 0

        activation (Cell):  Activation function. Default: ``None``.

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        F:  Feature dimension.

    """

    def __init__(self,
                 dim: int,
                 num_agg: int,
                 n_hidden: int = 0,
                 activation: Cell = None,
                 **kwargs,
                 ):

        super().__init__(
            dim=dim,
            num_agg=num_agg,
            n_hidden=n_hidden,
            activation=activation,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.dim = dim
        self.num_agg = num_agg
        self.n_hidden = n_hidden
        self.activation = activation

        sub_dim = self.dim // self.num_agg
        last_dim = self.dim - (sub_dim * (self.num_agg - 1))
        sub_dims = [sub_dim for _ in range(self.num_agg - 1)]
        sub_dims.append(last_dim)

        hidden_layers = [dim] * self.n_hidden
        self.mcr = nn.CellList([
            MLP(self.dim, hidden_layers + [sub_dims[i]],
                activation=self.activation)
            for i in range(self.num_agg)
        ])

    def __str__(self):
        return "MCR"

    def construct(self, ylist: List[Tensor], atom_mask: Tensor = None):
        readouts = ()
        for i in range(self.num_agg):
            readouts = readouts + (self.mcr[i](ylist[i]),)
        y = ops.concat(readouts, axis=-1)
        if atom_mask is not None:
            y = y * F.expand_dims(atom_mask, -1)
        return y

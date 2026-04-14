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
Basic functions
"""
from typing import Union
from collections.abc import Sequence
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import nn
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.common.initializer import Constant
from mindspore.common.initializer import Initializer
from ..configs import Config

from .cutoff import SmoothCutoff
from ..configs import Registry as R

__all__ = [
    'MLP',
    "SmoothReciprocal",
    "SoftmaxWithMask",
    "PositionalEmbedding",
    "MultiheadAttention",
    "Pondering",
    "ACTWeight",
]


@R.register('layer.mlp')
class MLP(nn.Cell):
    r"""Multi-layer perceptron

    Args:
        input_dims (int): The number of channels in the input space.
        hidden_dims (List[int]): Dimension of hidden layers.Default: ``None``.
        activation (Cell): Activation function. Default: ``None``.
        short_cut (bool): Add inputs or not. Default: ``False``.
        batch_norm (bol): Batch normalization for each layer or not. Default: ``False``.
        weight_init (Union[Initializer, str]): The trainable weight_init parameter. Default: 'xavier_uniform'
        bias_init (Union[Initializer, str]): The trainable bias_init parameter. Default: 'zeros'

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim: int,
                 hidden_dims: Sequence,
                 short_cut=False,
                 batch_norm=False,
                 activation='relu',
                 weight_init: Union[Initializer, str] = 'xavier_uniform',
                 bias_init: Union[Initializer, str] = 'zero',
                 dropout=0,
                 **kwargs):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), **kwargs)

        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut
        if isinstance(activation, str):
            self.activation = R.get('activation', activation)()
        else:
            self.activation = activation
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        fcs = [nn.Dense(dim, self.dims[i+1], weight_init=weight_init,
                        bias_init=bias_init) for i, dim in enumerate(self.dims[:-1])]
        self.layers = nn.CellList(fcs)
        self.batch_norms = None
        if batch_norm:
            bns = [nn.BatchNorm1d(dim) for dim in self.dims[1:-1]]
            self.batch_norms = nn.CellList(bns)

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
                if self.batch_norms is not None:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
                if self.short_cut and hidden.shape == hidden.shape:
                    hidden += inputs
        return hidden


class SmoothReciprocal(Cell):
    r"""A smooth reciprocal function

    Args:
        dmax (float):           Maximum distance

        cutoff_network (Cell):  Cutoff network. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dmax: float,
                 cutoff_network: Cell = None,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        if cutoff_network is None:
            self.cutoff_network = SmoothCutoff(dmax)
        else:
            self.cutoff_network = cutoff_network(dmax)

    def construct(self, rij: Tensor, mask: Tensor):
        """calculate smooth reciprocal of Tensor

        Args:
            rij (Tensor):   Tensor with shape (..., X, ...). Data type is float.
            mask (Tensor):  Tensor with shape (..., X, ...). Data type is bool.

        Returns:
            output (Tensor):    Tensor with shape (..., X, ...). Data type is float.

        """
        phi2rij, _ = self.cutoff_network(rij*2, mask)

        r_near = phi2rij * msnp.reciprocal(F.sqrt(rij * rij + 1.0))
        r_far = msnp.where(rij > 0, (1.0 - phi2rij) * msnp.reciprocal(rij), 0)

        reciprocal = r_near + r_far
        if mask is not None:
            reciprocal = reciprocal * mask

        return reciprocal


class SoftmaxWithMask(Cell):
    r"""Softmax function with mask

    Args:
        axis (int): Axis of Tensor to do softmax. Default: -1

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.softmax = ops.Softmax(int(axis))

        self.large_neg = Tensor(-5e4, ms.float16)

    def construct(self, x: Tensor, mask: Tensor):
        """Compute softmax of Tensor with mask

        Args:
            x (Tensor):     Tensor with shape (..., X, ...). Data type is float.
            mask (Tensor):  Tensor with shape (..., X, ...). Data type is bool.

        Returns:
            output (Tensor):    Tensor with shape (..., X, ...). Data type is float.

        """

        xm = msnp.where(mask, x, self.large_neg)
        return self.softmax(xm)


class PositionalEmbedding(Cell):
    r"""Positional embedding to generate query, key and value for self-attention

    Args:
        dim (int):                      Last dimension of Tensor.

        use_distances (bool):           Whether to use distance information. Default: ``True``.

        use_bonds (bool):               Whether to use bond information. Default: ``False``.

        use_public_layer_norm (bool):   Whether to share layer normalization network. Default: ``True``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim: int,
                 use_public_layer_norm: bool = True,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        dim = int(dim)

        if use_public_layer_norm:
            self.norm = nn.LayerNorm((dim,), -1, -1)
            self.x_norm = self.norm
            self.g_norm = self.norm
        else:
            self.x_norm = nn.LayerNorm((dim,), -1, -1)
            self.g_norm = nn.LayerNorm((dim,), -1, -1)

        self.x2q = nn.Dense(dim, dim, weight_init='xavier_uniform', has_bias=False)
        self.x2k = nn.Dense(dim, dim, weight_init='xavier_uniform', has_bias=False)
        self.x2v = nn.Dense(dim, dim, weight_init='xavier_uniform', has_bias=False)

    def construct(self,
                  x_i: Tensor,
                  g_ij: Tensor,
                  t: float = 0,
                  ):
        """Get query, key and query from atom types and positions

        Args:
            x_i (Tensor): Tensor with shape `(B, A, F)`. Data type is float.
            g_ij (Tensor): Tensor with shape `(B, A, A, F)`. Data type is float.

        Note:
            B:  Batch size
            A:  Number of atoms
            F:  Dimensions of feature space

        Returns:
            query (Tensor): Tensor with shape (B, A, 1, F). Data type is float.
            key (Tensor):   Tensor with shape (B, A, A, F). Data type is float.
            value (Tensor): Tensor with shape (B, A, A, F). Data type is float.

        """
        # The shape looks like (B, 1, F) <- (B, F) <- (B, A, A, F)
        g_i = F.expand_dims(g_ij[..., 0, 0, :], -2)

        # The shape looks like (B, A, F) * (B, 1, F)
        xgi = F.mul(x_i, g_i)
        # The shape looks like (B, A, 1, F)
        xgii = F.expand_dims(xgi, -2)

        # The shape looks like (B, A, A, F) = (B, A, 1, F) * (B, A, A, F)
        xgij = F.mul(F.expand_dims(x_i, -2), g_ij)

        # The shape looks like (B, A, A, F)
        xgii = self.norm(xgii + t)
        xgij = self.norm(xgij + t)

        # The shape looks like (B, A, 1, F)
        query = self.x2q(xgii)
        # The shape looks like (B, A, A, F)
        key = self.x2k(xgij)
        # The shape looks like (B, A, A, F)
        value = self.x2v(xgij)

        return query, key, value


class MultiheadAttention(Cell):
    r"""Multi-head attention.

    Args:
        dim_feature (int):  Diension of feature space (F).

        n_heads (int):      Number of heads (h). Default: 8

        dim_tensor (int):   Dimension of input tensor (D). Default: 4

    Note:

        X:  Dimension to be aggregated

        F:  Dimension of Feature space

        h:  Number of heads for multi-head attention

        f:  Dimensions per head (F = f * h)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim_feature: int,
                 n_heads: int = 8,
                 dim_tensor: int = 4,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        dim_tensor = int(dim_tensor)
        if dim_tensor < 2:
            raise ValueError('dim_tensor must be larger than 1')

        self.n_heads = int(n_heads)
        dim_feature = int(dim_feature)

        self.size_per_head = dim_feature // self.n_heads
        self.scores_mul = 1.0 / msnp.sqrt(float(self.size_per_head))

        # The shape looks like (h, f)
        self.reshape_tail = (self.n_heads, self.size_per_head)

        self.output = nn.Dense(dim_feature, dim_feature, weight_init='xavier_uniform', has_bias=False)

        self.softmax = ops.Softmax()
        self.bmm = ops.BatchMatMul()
        self.bmmt = ops.BatchMatMul(transpose_b=True)

        self.softmax_with_mask = SoftmaxWithMask()

    def construct(self,
                  query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  mask: Tensor = None,
                  cutoff: Tensor = None
                  ):
        """Compute multi-head attention.

        Args:
            query (Tensor):     Tensor with shape (..., Q, F). Data type is float.
            key (Tensor):       Tensor with shape (..., X, F). Data type is float.
            value (Tensor):     Tensor with shape (..., X, F). Data type is float.
            mask (Tensor):      Tensor with shape (..., X). Data type is bool.
            cutoff (Tensor):    Tensor with shape (..., X). Data type is float.

        Returns:
            output (Tensor):    Tensor with shape (..., F). Data type is float.

        """
        # pylint: disable=invalid-name

        if self.n_heads > 1:
            # The shape looks like (..., h, Q, f) <- (..., Q, h, f) <- (..., Q, F)
            query_ = F.reshape(query, query.shape[:-1] + self.reshape_tail).swapaxes(-2, -3)
            # The shape looks like (..., h, X, f) <- (..., X, h, f) <- (..., X, F)
            key_ = F.reshape(key, key.shape[:-1] + self.reshape_tail).swapaxes(-2, -3)
            # The shape looks like (..., h, X, f) <- (..., X, h, f) <- (..., X, F)
            value_ = F.reshape(value, value.shape[:-1] + self.reshape_tail).swapaxes(-2, -3)

            # The shape looks like (..., h, Q, X) = (..., h, Q, f) x (..., h, X, f)^T
            attention_scores = self.bmmt(query_, key_)
            # The shape looks like (..., h, Q, X) / sqrt(f)
            attention_scores = F.mul(attention_scores, self.scores_mul)

            if mask is None:
                attention_probs = self.softmax(attention_scores)
            else:
                attention_probs = self.softmax_with_mask(attention_scores, mask.expand_dims(-2).expand_dims(-2))

                if cutoff is not None:
                    # The shape looks like (..., h, Q, X) <- (..., 1, 1, X) <- (..., X)
                    excut = cutoff.expand_dims(-2).expand_dims(-2)
                    # The shape looks like (..., h, Q, X) = (..., h, X, X) * (..., 1, 1, X)
                    attention_probs = F.mul(attention_probs, excut)

            # The shape looks like (..., h, Q, f) = (..., h, Q, X) x (..., h, X, f)
            context = self.bmm(attention_probs, value_)
            # The shape looks like (..., Q, h, f)
            context = context.swapaxes(-2, -3)
            # The shape looks like (..., Q, F)
            context = F.reshape(context, query.shape)

        else:
            # The shape looks like (..., Q, X) = (..., Q, F) x (..., Q, F)^T / sqrt(F)
            attention_scores = self.bmmt(query, key) * self.scores_mul

            if mask is None:
                # The shape looks like (..., Q, X)
                attention_probs = self.softmax(attention_scores)
            else:
                # The shape looks like (..., 1, X)
                attention_probs = self.softmax_with_mask(attention_scores, F.expand_dims(mask, -2))

                if cutoff is not None:
                    # The shape looks like (..., 1, X) * (..., 1, X)
                    attention_probs = attention_probs * F.expand_dims(cutoff, -2)

            # The shape looks like (..., Q, F) = (..., Q, X) x (..., X, F)
            context = self.bmm(attention_probs, value)

        # The shape looks like (..., Q, F)
        return self.output(context)


class Pondering(Cell):
    r"""Pondering network for adapetive computation time.

    Args:
        n_in (int):         Dimension of input Tensor

        n_hidden (int):     Number of hidden layers. Default: 0

        bias_const (float): Initial value for bias. Default: 1

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 n_in: int,
                 n_hidden: int = 0,
                 bias_const: float = 1.,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        n_in = int(n_in)
        n_hidden = int(n_hidden)

        if n_hidden == 0:
            self.dense = nn.Dense(n_in, 1, has_bias=True, weight_init='xavier_uniform', bias_init=Constant(
                bias_const), activation='sigmoid')
        elif n_hidden > 0:
            nets = []
            for _ in range(n_hidden):
                nets.append(
                    nn.Dense(n_in, n_in, weight_init='xavier_uniform', activation='relu'))
            nets.append(nn.Dense(n_in, 1, weight_init='xavier_uniform',
                                 bias_init=Constant(bias_const), activation='sigmoid'))
            self.dense = nn.SequentialCell(nets)
        else:
            raise ValueError("n_hidden cannot be negative!")

    def construct(self, x: Tensor):
        """Calculate pondering network.

        Args:
            x (Tensor): Tensor with shape (B, A, X). Data type is float.

        Returns:
            y (Tensor): Tensor with shape (B, A, 1). Data type is float.

        """
        y = self.dense(x)
        return y.squeeze(-1)


class ACTWeight(Cell):
    r"""Adapetive computation time modified from:
        https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/UTransformer.py

    Args:
        n_in (int):         Dimension of input Tensor

        n_hidden (int):     Number of hidden layers. Default: 0

        bias_const (float): Initial value for bias. Default: 1

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, threshold: float = 0.9, **kwargs):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.threshold = threshold

    def construct(self, prob: Tensor, halting_prob: Tensor):
        """Calculate Adapetive computation time.

        Args:
            prob (Tensor):          Tensor with shape (B, A, 1). Data type is float.
            halting_prob (Tensor):  Tensor with shape (B, A, 1). Data type is float.

        Returns:
            w (Tensor):     Tensor with shape (B, A, 1). Data type is float.
            dp (Tensor):    Tensor with shape (B, A, 1). Data type is float.
            dn (Tensor):    Tensor with shape (B, A, 1). Data type is float.

        """

        # Mask for inputs which have not halted last cy
        running = F.cast(halting_prob < 1.0, prob.dtype)

        # Add the halting probability for this step to the halting
        # probabilities for those input which haven't halted yet
        add_prob = prob * running
        new_prob = halting_prob + add_prob
        mask_run = F.cast(new_prob <= self.threshold, prob.dtype)
        mask_halt = F.cast(new_prob > self.threshold, prob.dtype)

        # Mask of inputs which haven't halted, and didn't halt this step
        still_running = mask_run * running
        running_prob = halting_prob + prob * still_running

        # Mask of inputs which halted at this step
        new_halted = mask_halt * running

        # Compute remainders for the inputs which halted at this step
        remainders = new_halted * (1.0 - running_prob)

        # Add the remainders to those inputs which halted at this step
        dp = add_prob + remainders

        # Increment n_updates for all inputs which are still running
        dn = running

        # Compute the weight to be applied to the new state and output
        # 0 when the input has already halted
        # prob when the input hasn't halted yet
        # the remainders when it halted this step
        update_weights = prob * still_running + new_halted * remainders
        w = F.expand_dims(update_weights, -1)

        return w, dp, dn

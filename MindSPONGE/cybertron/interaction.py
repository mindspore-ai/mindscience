# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
Interaction layers
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

import mindsponge.function as func

from .block import Dense, MLP
from .block import PreActDense
from .block import SeqPreActResidual
from .base import Aggregate
from .base import PositionalEmbedding
from .base import MultiheadAttention
from .base import Pondering, ACTWeight
from .base import FeedForward
from .activation import get_activation

__all__ = [
    "Interaction",
    "SchNetInteraction",
    "PhysNetModule",
    "NeuralInteractionUnit",
]


class Interaction(Cell):
    r"""Interaction layer network

    Args:

        dim_feature (int):          Feature dimension.

        activation (Cell):          Activation function. Default: None

        use_distance (bool):        Whether to use distance between atoms. Default: True

        use_bond (bool):            Whether to use bond information. Default: False


    """

    def __init__(self,
                 dim_feature: int,
                 activation: Cell = None,
                 use_distances: bool = True,
                 use_bonds: bool = False,
                 ):
        super().__init__()

        self.reg_key = 'interaction'
        self.name = 'Interaction'
        self.dim_feature = func.get_integer(dim_feature)
        self.activation = get_activation(activation)
        self.use_bonds = use_bonds
        self.use_distances = use_distances

        self.gather_neighbours = func.gather_vectors

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        """print information of interaction layer"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print('-'*80)
        return self

    def _output_block(self, x: Tensor) -> Tensor:
        """output block network"""
        return x

    def construct(self,
                  x: Tensor,
                  f_ij: Tensor,
                  b_ij: Tensor,
                  c_ij: Tensor,
                  neighbours: Tensor,
                  mask: Tensor = None,
                  e: Tensor = None,
                  f_ii: Tensor = None,
                  b_ii: Tensor = None,
                  c_ii: Tensor = None,
                  atom_mask: Tensor = None
                  ):

        """Compute interaction layer.

        Args:
            x (Tensor):             Tensor of shape (B, A, F). Data type is float
                                    Representation of each atom.
            f_ij (Tensor):          Tensor of shape (B, A, N, F). Data type is float
                                    Edge vector of distance.
            b_ij (Tensor):          Tensor of shape (B, A, N, F). Data type is float
                                    Edge vector of bond connection.
            c_ij (Tensor):          Tensor of shape (B, A, N). Data type is float
                                    Cutoff for distance.
            neighbours (Tensor):    Tensor of shape (B, A, N). Data type is int
                                    Neighbour index.
            mask (Tensor):          Tensor of shape (B, A, N). Data type is bool
                                    Mask of neighbour index.
            e (Tensor):             Tensor of shape (B, A, F). Data type is float
                                    Embdding vector for each atom
            f_ii (Tensor):          Tensor of shape (B, A, 1, F). Data type is float
                                    Edge vector of distance for atom itself.
            b_ii (Tensor):          Tensor of shape (B, A, 1, F). Data type is float
                                    Edge vector of bond connection for atom itself.
            c_ii (Tensor):          Tensor of shape (B, A). Data type is float
                                    Cutoff for atom itself.
            atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool
                                    Mask for each atom

        Returns:
            y: (Tensor)             Tensor of shape (B, A, F). Data type is float

        Symbols:

            B:  Batch size.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates, usually is 3.
            F:  Feature dimension of representation.

        """

        #pylint: disable=unused-argument
        return x


class SchNetInteraction(Interaction):
    r"""Interaction layer of SchNet.

    Args:

        dim_feature (int):          Feature dimension.

        dim_filter (int):           Dimension of filter network.

        dis_filter (Cell):          Filter network for distance

        activation (Cell):          Activation function. Default: 'ssp'

        normalize_filter (bool):    Whether to nomalize filter network. Default: False

    """

    def __init__(self,
                 dim_feature: int,
                 dim_filter: int,
                 dis_filter: Cell,
                 activation: Cell = 'ssp',
                 normalize_filter: bool = False,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            activation=activation,
            use_distances=True,
            use_bonds=False,
        )

        self.dim_filter = func.get_integer(dim_filter)

        self.name = 'SchNet Interaction Layer'
        self.atomwise_bc = Dense(self.dim_feature, self.dim_filter)
        self.atomwise_ac = MLP(self.dim_filter, self.dim_feature, [self.dim_feature],
                               activation=self.activation, use_last_activation=False)

        self.dis_filter = dis_filter
        self.agg = Aggregate(axis=-2, mean=normalize_filter)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Dimension for filter network: ' + str(self.dim_filter))
        print('-'*80)
        return self

    def construct(self,
                  x: Tensor,
                  f_ij: Tensor,
                  b_ij: Tensor,
                  c_ij: Tensor,
                  neighbours: Tensor,
                  mask: Tensor = None,
                  e: Tensor = None,
                  f_ii: Tensor = None,
                  b_ii: Tensor = None,
                  c_ii: Tensor = None,
                  atom_mask: Tensor = None
                  ):

        #pylint: disable=invalid-name

        ax = self.atomwise_bc(x)
        x_ij = self.gather_neighbours(ax, neighbours)

        # CFconv: pass expanded interactomic distances through filter block
        W_ij = self.dis_filter(f_ij) * F.expand_dims(c_ij, -1)
        y = x_ij * W_ij

        # atom-wise multiplication, aggregating and Dense layer
        y = self.agg(y, mask)
        v = self.atomwise_ac(y)

        x_new = x + v

        return x_new


class PhysNetModule(Interaction):
    r"""PhysNet Module (Interaction layer)

    Args:

        dim_feature (int):          Feature dimension.

        dis_filter (Cell):          Filter network for distance

        activation (Cell):          Activation function. Default: 'swish'

        n_inter_residual (int):     Number of inter residual blocks. Default: 3

        n_outer_residual (int):     Number of outer residual blocks. Default: 2

    """
    def __init__(self,
                 dim_feature: int,
                 dis_filter: Cell = None,
                 activation: Cell = 'swish',
                 n_inter_residual: int = 3,
                 n_outer_residual: int = 2,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            activation=activation,
            use_distances=True,
            use_bonds=False,
        )

        self.name = 'PhysNet Module Layer'

        self.xi_dense = Dense(
            self.dim_feature, self.dim_feature, activation=self.activation)
        self.xij_dense = Dense(
            self.dim_feature, self.dim_feature, activation=self.activation)
        self.dis_filter = dis_filter

        self.gating_vector = Parameter(initializer(
            Normal(1.0), [self.dim_feature]), name="gating_vector")

        self.n_inter_residual = func.get_integer(n_inter_residual)
        self.n_outer_residual = func.get_integer(n_outer_residual)

        self.inter_residual = SeqPreActResidual(self.dim_feature, activation=self.activation,
                                                n_res=self.n_inter_residual)
        self.inter_dense = PreActDense(self.dim_feature, self.dim_feature, activation=self.activation)
        self.outer_residual = SeqPreActResidual(self.dim_feature, activation=self.activation,
                                                n_res=self.n_outer_residual)

        self.reducesum = P.ReduceSum()

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + self.dim_feature)
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Number of layers at inter residual: ' +
              self.n_inter_residual)
        print(ret+gap+' Number of layers at outer residual: ' +
              self.n_outer_residual)
        print('-'*80)
        return self

    def _attention_mask(self, f_ij, c_ij) -> Tensor:
        """attention mask"""
        x = f_ij * F.expand_dims(c_ij, -1)
        return self.dis_filter(x)

    def _interaction_block(self, x, f_ij, c_ij, neighbours, mask) -> Tensor:
        """interaction block"""

        #pylint: disable=invalid-name
        xi = self.activation(x)
        xij = self.gather_neighbours(xi, neighbours)

        ux = self.gating_vector * x

        dxi = self.xi_dense(xi)
        dxij = self.xij_dense(xij)
        Ggij = self._attention_mask(f_ij, c_ij)

        side = Ggij * dxij
        if mask is not None:
            side = side * F.expand_dims(mask, -1)
        v = dxi + self.reducesum(side, -2)

        v1 = self.inter_residual(v)
        v1 = self.inter_dense(v1)
        return ux + v1

    def construct(self,
                  x: Tensor,
                  f_ij: Tensor,
                  b_ij: Tensor,
                  c_ij: Tensor,
                  neighbours: Tensor,
                  mask: Tensor = None,
                  e: Tensor = None,
                  f_ii: Tensor = None,
                  b_ii: Tensor = None,
                  c_ii: Tensor = None,
                  atom_mask: Tensor = None
                  ):

        x1 = self._interaction_block(x, f_ij, c_ij, neighbours, mask)
        xnew = self.outer_residual(x1)

        return xnew


class NeuralInteractionUnit(Interaction):
    r"""Neural interaction unit for MolCT.

    Args:

        dim_feature (int):          Feature dimension.

        n_heads (int):              Number of head for multi-head attention. Default: 8

        max_cycles (int):           Maximum cycles for adaptive computation time (ACT). Default: 10

        activation (Cell):          Activation function. Default: 'swish'

        dis_filter (Cell):          Filter network for distance

        bond_filter (Cell):         Filter network for bond connection

        fixed_cycles (bool):        Whether to fixed number of cyles to do ACT. Default: False

        use_feed_forward (bool):    Whether to use feed forward network. Default: False

        act_threshold (float):      Threshold value for ACT. Default: 0.9


    """

    def __init__(self,
                 dim_feature: int,
                 n_heads: int = 8,
                 max_cycles: int = 10,
                 activation: Cell = 'swish',
                 dis_filter: Cell = None,
                 bond_filter: Cell = None,
                 fixed_cycles: bool = False,
                 use_feed_forward: bool = False,
                 act_threshold: float = 0.9,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            activation=activation,
            use_distances=(dis_filter is not None),
            use_bonds=(bond_filter is not None),
        )
        if dim_feature % n_heads != 0:
            raise ValueError('The term "dim_feature" cannot be divisible ' +
                             'by the term "n_heads" in AirNetIneteraction! ')

        self.name = 'Neural Interaction Unit'

        self.n_heads = func.get_integer(n_heads)
        self.max_cycles = func.get_integer(max_cycles)

        self.fixed_cycles = fixed_cycles

        if self.fixed_cycles:
            self.time_embedding = [0 for _ in range(self.max_cycles)]
        else:
            self.time_embedding = self._get_time_signal(
                self.max_cycles, self.dim_feature)

        self.dis_filter = dis_filter
        self.bond_filter = bond_filter

        self.positional_embedding = PositionalEmbedding(
            self.dim_feature, self.use_distances, self.use_bonds)
        self.multi_head_attention = MultiheadAttention(
            self.dim_feature, self.n_heads, dim_tensor=4)

        self.use_feed_forward = use_feed_forward
        self.feed_forward = None
        if self.use_feed_forward:
            self.feed_forward = FeedForward(self.dim_feature, self.activation)

        self.act_threshold = act_threshold
        self.act_epsilon = 1.0 - act_threshold

        self.pondering = None
        self.act_weight = None
        self.do_act = False
        if self.max_cycles > 1:
            self.do_act = True
            self.pondering = Pondering(self.dim_feature*3, bias_const=3)
            self.act_weight = ACTWeight(self.act_threshold)

        self.concat = P.Concat(-1)
        self.zeros_like = P.ZerosLike()
        self.zeros = P.Zeros()

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_feature))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Encoding distance: ' +
              ("No" if self.dis_filter is None else "Yes"))
        print(ret+gap+' Encoding bond: ' +
              "No" if self.bond_filter is None else "Yes")
        print(ret+gap+' Number of heads in multi-haed attention: '+str(self.n_heads))
        print(ret+gap+' Use feed forward network: ' +
              ('Yes' if self.use_feed_forward else 'No'))
        if self.max_cycles > 1:
            print(
                ret+gap+' Adaptive computation time (ACT) with maximum cycles: '+str(self.max_cycles))
            print(ret+gap+' Cycle mode: ' +
                  ('Fixed' if self.fixed_cycles else 'Fixible'))
            print(ret+gap+' Threshold for ACT: '+str(self.act_threshold))
        print('-'*80)
        return self

    def _encoder(self,
                 x: Tensor,
                 neighbours: Tensor,
                 g_ii: Tensor = 1,
                 g_ij: Tensor = 1,
                 b_ii: Tensor = 0,
                 b_ij: Tensor = 0,
                 t: Tensor = 0,
                 cutoff: Tensor = None,
                 mask: Tensor = None) -> Tensor:

        """encoder for transformer"""

        xij = self.gather_neighbours(x, neighbours)
        query, key, value = self.positional_embedding(
            x, xij, g_ii, g_ij, b_ii, b_ij, t)
        v = self.multi_head_attention(
            query, key, value, mask=mask, cutoff=cutoff)
        v = v.squeeze(-2)

        y = x + v

        if self.use_feed_forward:
            y = self.feed_forward(y)

        return y

    def _get_time_signal(self, length, channels, min_timescale=1.0, max_timescale=1.0e4) -> Tensor:
        """
        Generates a [1, length, channels] timing signal consisting of sinusoids
        Adapted from:
        https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/common_layer.py
        """
        position = msnp.arange(length, dtype=ms.float32)
        num_timescales = channels // 2
        log_timescale_increment = msnp.log(
            max_timescale / min_timescale, dtype=ms.float32) / (num_timescales - 1)
        inv_timescales = min_timescale * \
            msnp.exp(msnp.arange(num_timescales, dtype=ms.float32)
                     * -log_timescale_increment)
        scaled_time = F.expand_dims(position, 1) * \
            F.expand_dims(inv_timescales, 0)

        signal = msnp.concatenate([msnp.sin(scaled_time, dtype=ms.float32), msnp.cos(
            scaled_time, dtype=ms.float32)], axis=1)
        signal = msnp.pad(signal, [[0, 0], [0, channels % 2]],
                          'constant', constant_values=[0.0, 0.0])

        return signal

    def construct(self,
                  x: Tensor,
                  f_ij: Tensor,
                  b_ij: Tensor,
                  c_ij: Tensor,
                  neighbours: Tensor,
                  mask: Tensor = None,
                  e: Tensor = None,
                  f_ii: Tensor = None,
                  b_ii: Tensor = None,
                  c_ii: Tensor = None,
                  atom_mask: Tensor = None
                  ):

        if self.dis_filter is not None:
            g_ii = self.dis_filter(f_ii)
            g_ij = self.dis_filter(f_ij)
        else:
            g_ii = f_ii
            g_ij = f_ij

        if self.bond_filter is not None:
            b_ii = self.bond_filter(b_ii)
            b_ij = self.bond_filter(b_ij)

        cutoff = self.concat((c_ii, c_ij))
        mask = self.concat((atom_mask, mask))

        if self.do_act:
            xx = x
            x0 = self.zeros_like(x)

            halting_prob = self.zeros((x.shape[0], x.shape[1]), ms.float32)
            n_updates = self.zeros((x.shape[0], x.shape[1]), ms.float32)

            broad_zeros = self.zeros_like(e)

            if self.fixed_cycles:
                for cycle in range(self.max_cycles):
                    t = self.time_embedding[cycle]
                    vt = broad_zeros + t

                    xp = self.concat((xx, e, vt))
                    p = self.pondering(xp)
                    w, dp, dn = self.act_weight(p, halting_prob)
                    halting_prob = halting_prob + dp
                    n_updates = n_updates + dn

                    xx = self._encoder(xx, neighbours, g_ii,
                                       g_ij, b_ii, b_ij, t, cutoff, mask)

                    cycle = cycle + 1

                    x0 = xx * w + x0 * (1.0 - w)
            else:
                cycle = self.zeros((), ms.int32)
                while((halting_prob < self.act_threshold).any() and (cycle < self.max_cycles)):
                    t = self.time_embedding[cycle]
                    vt = broad_zeros + t
                    xp = self.concat((xx, e, vt))
                    p = self.pondering(xp)
                    w, dp, dn = self.act_weight(p, halting_prob)
                    halting_prob = halting_prob + dp
                    n_updates = n_updates + dn

                    xx = self._encoder(xx, neighbours, g_ii,
                                       g_ij, b_ii, b_ij, t, cutoff, mask)

                    cycle = cycle + 1

                    x0 = xx * w + x0 * (1.0 - w)
        else:
            t = self.time_embedding[0]
            x0 = self._encoder(x, neighbours, g_ii, g_ij,
                               b_ii, b_ij, t, cutoff, mask)

        return x0

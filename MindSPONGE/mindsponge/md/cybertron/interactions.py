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
"""interactions"""

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from .blocks import Dense, MLP
from .blocks import PreActDense
from .blocks import SeqPreActResidual
from .neighbors import GatherNeighbors
from .base import Aggregate, CFconv
from .base import PositionalEmbedding
from .base import MultiheadAttention
from .base import Pondering, ACTWeight
from .base import FeedForward
from .activations import get_activation

__all__ = [
    "Interaction",
    "SchNetInteraction",
    "PhysNetModule",
    "NeuralInteractionUnit",
]


class Interaction(nn.Cell):
    """Interaction"""
    def __init__(self,
                 gather_dim,
                 fixed_neigh,
                 activation=None,
                 use_distances=True,
                 use_bonds=False
                 ):
        super().__init__()

        self.name = 'Interaction'
        self.fixed_neigh = fixed_neigh
        self.use_bonds = use_bonds
        self.use_distances = use_distances
        self.activation = activation
        self.gather_neighbors = GatherNeighbors(gather_dim, fixed_neigh)

    def set_fixed_neighbors(self, flag=True):
        self.fixed_neigh = flag
        self.gather_neighbors.fixed_neigh = flag

    def _output_block(self, x):
        return x


class SchNetInteraction(Interaction):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        dim_feature (int): number of input atomic vector dimensions.
        dim_filter (int): dimensions of filter network.
        cfconv_module (nn.Cell): the algorithm to calcaulte continuous-filter
            convoluations.
        cutoff_network (nn.Cell, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(self, dim_feature, num_rbf, dim_filter, activation='swish', fixed_neigh=False,
                 normalize_filter=False,):
        super().__init__(
            gather_dim=dim_filter,
            fixed_neigh=fixed_neigh,
            activation=activation,
            use_bonds=False,
        )

        self.dim_filter = dim_filter
        self.activation = activation

        self.name = 'SchNet Interaction Layer'
        self.atomwise_bc = Dense(dim_feature, dim_filter)
        self.atomwise_ac = MLP(
            dim_filter, dim_feature, [
                dim_feature,], activation=activation, use_last_activation=False)

        self.cfconv = CFconv(num_rbf, dim_filter, activation)
        self.agg = Aggregate(axis=-2, mean=normalize_filter)

    def print_info(self):
        print('---------with activation function: ' + str(self.activation))
        if self.use_distances and self.use_bonds:
            print('---------with edge type: distances and bonds')
        else:
            print('---------with edge type: ' +
                  ('distance' if self.use_distances else 'bonds'))

        print('---------with dimension for filter network: ' + str(self.dim_filter))

    def construct(self, x, f_ij, c_ij, neighbors, mask=None):
        """Compute convolution block.

        Args:
            x (ms.Tensor[float]): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            rbf (ms.Tensor[float]): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (ms.Tensor[int]): indices of neighbors of (N_b, N_a, N_nbh) shape.
            mask (ms.Tensor[bool]): mask to filter out non-existing neighbors
                introduced via padding.

        Returns:
            ms.Tensor: block output with (N_b, N_a, n_out) shape.

        """

        ax = self.atomwise_bc(x)
        xij = self.gather_neighbors(ax, neighbors)

        # CFconv: pass expanded interactomic distances through filter block
        y = self.cfconv(xij, f_ij, c_ij)
        # element-wise multiplication, aggregating and Dense layer
        y = self.agg(y, mask)

        v = self.atomwise_ac(y)

        x_new = x + v

        return x_new


class PhysNetModule(Interaction):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        dim_feature (int): number of input atomic vector dimensions.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(self, num_rbf, dim_feature, activation='swish', fixed_neigh=False, n_inter_residual=3,
                 n_outer_residual=2,):
        super().__init__(
            gather_dim=dim_feature,
            fixed_neigh=fixed_neigh,
            activation=activation,
            use_bonds=False,
        )

        self.name = 'PhysNet Module Layer'

        self.xi_dense = Dense(dim_feature, dim_feature, activation=activation)
        self.xij_dense = Dense(dim_feature, dim_feature, activation=activation)
        self.fij_dense = Dense(
            num_rbf,
            dim_feature,
            has_bias=False,
            activation=None)

        self.gating_vector = Parameter(
            initializer(Normal(1.0), [dim_feature,]), name="gating_vector")

        self.inter_residual = SeqPreActResidual(
            dim_feature, activation=activation, n_res=n_inter_residual)
        self.inter_dense = PreActDense(
            dim_feature, dim_feature, activation=activation)
        self.outer_residual = SeqPreActResidual(
            dim_feature, activation=activation, n_res=n_outer_residual)

        self.activation = get_activation(activation)

        self.reducesum = P.ReduceSum()

    def print_info(self):
        print('---------with activation function: ' + str(self.activation))
        if self.use_distances and self.use_bonds:
            print('---------with edge type: distances and bonds')
        else:
            print('---------with edge type: ' +
                  ('distance' if self.use_distances else 'bonds'))

    def _interaction_block(self, x, f_ij, c_ij, neighbors, mask):
        """_interaction_block"""

        xi = self.activation(x)
        xij = self.gather_neighbors(xi, neighbors)

        ux = self.gating_vector * x

        dxi = self.xi_dense(xi)
        dxij = self.xij_dense(xij)
        g_gij = self.fij_dense(f_ij * F.expand_dims(c_ij, -1))

        side = g_gij * dxij
        if mask is not None:
            side = side * F.expand_dims(mask, -1)
        v = dxi + self.reducesum(side, -2)

        v1 = self.inter_residual(v)
        v1 = self.inter_dense(v1)
        return ux + v1

    def construct(self, x, f_ij, c_ij, neighbors, mask=None):
        """Compute convolution block.

        Args:
            x (ms.Tensor[float]): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            rbf (ms.Tensor[float]): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (ms.Tensor[int]): indices of neighbors of (N_b, N_a, N_nbh) shape.
            mask (ms.Tensor[bool]): mask to filter out non-existing neighbors
                introduced via padding.

        Returns:
            ms.Tensor: block output with (N_b, N_a, n_out) shape.

        """

        x1 = self._interaction_block(x, f_ij, c_ij, neighbors, mask)
        xnew = self.outer_residual(x1)

        return xnew


class NeuralInteractionUnit(Interaction):
    r"""Continuous-filter convolution block used in SchNet module.

    Args:
        dim_feature (int): dimensions of feature space.
        cfconv_module (nn.Cell): the algorithm to calcaulte continuous-filter
            convoluations.
        cutoff_network (nn.Cell, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
    """

    def __init__(self, dim_feature, num_rbf, n_heads=8, activation='swish', max_cycles=10, time_embedding=0,
                 use_pondering=True, fixed_cycles=False, use_distances=True, use_dis_filter=True, use_bonds=False,
                 use_bond_filter=False, act_threshold=0.9, fixed_neigh=False, use_feed_forward=False,):
        super().__init__(
            gather_dim=dim_feature,
            fixed_neigh=fixed_neigh,
            activation=activation,
            use_distances=use_distances,
            use_bonds=use_bonds
        )
        if dim_feature % n_heads != 0:
            raise ValueError('The term "dim_feature" cannot be divisible ' +
                             'by the term "n_heads" in AirNetIneteraction! ')

        self.name = 'Neural Interaction Unit'

        self.n_heads = n_heads
        self.max_cycles = max_cycles
        self.dim_feature = dim_feature
        self.num_rbf = num_rbf
        self.time_embedding = time_embedding

        if fixed_cycles:
            self.flexable_cycels = False
        else:
            self.flexable_cycels = True

        self.use_dis_filter = use_dis_filter
        if self.use_dis_filter:
            self.dis_filter = Dense(
                num_rbf,
                dim_feature,
                has_bias=True,
                activation=None)
        else:
            self.dis_filter = None

        self.bond_filter = None
        self.use_bond_filter = use_bond_filter
        if self.use_bond_filter:
            self.bond_filter = Dense(
                dim_feature,
                dim_feature,
                has_bias=False,
                activation=None)

        self.positional_embedding = PositionalEmbedding(
            dim_feature, self.use_distances, self.use_bonds)
        self.multi_head_attention = MultiheadAttention(
            dim_feature, n_heads, dim_tensor=4)

        self.use_feed_forward = use_feed_forward
        self.feed_forward = None
        if self.use_feed_forward:
            self.feed_forward = FeedForward(dim_feature, activation)

        self.act_threshold = act_threshold
        self.act_epsilon = 1.0 - act_threshold

        self.use_pondering = use_pondering
        self.pondering = None
        self.act_weight = None
        if self.max_cycles > 1:
            if self.use_pondering:
                self.pondering = Pondering(dim_feature * 3, bias_const=3)
                self.act_weight = ACTWeight(self.act_threshold)
            else:
                if self.flexable_cycels:
                    raise ValueError(
                        'The term "fixed_cycles" must be True ' +
                        'when the pondering network is None in AirNetIneteraction! ')
        self.fixed_weight = Tensor(1.0 / max_cycles, ms.float32)

        self.max = P.Maximum()
        self.min = P.Minimum()
        self.concat = P.Concat(-1)
        self.reducesum = P.ReduceSum()
        self.squeeze = P.Squeeze(-2)
        self.ones_like = P.OnesLike()
        self.zeros_like = P.ZerosLike()
        self.zeros = P.Zeros()

    def print_info(self):
        """print info"""
        print('---------with activation function: ' + str(self.activation))
        if self.use_distances and self.use_bonds:
            print('---------with edge type: distances and bonds')
        else:
            print('---------with edge type: ' +
                  ('distance' if self.use_distances else 'bonds'))

        if self.use_distances:
            print('---------with filter for distances: ' +
                  ('yes' if self.use_dis_filter else 'no'))

        if self.use_bonds:
            print('---------with filter for bonds: ' +
                  ('yes' if self.use_bond_filter else 'no'))

        print('---------with multi-haeds: ' + str(self.n_heads))
        print('---------with feed forward: ' +
              ('yes' if self.use_feed_forward else 'no'))
        if self.max_cycles > 1:
            print('---------using adaptive computation time with threshold: ' +
                  str(self.act_threshold))

    def _transformer_encoder(self, x, neighbors, g_ii=1, g_ij=1, b_ii=0, b_ij=0, c_ij=None, t=0, mask=None):
        """_transformer_encoder"""

        xij = self.gather_neighbors(x, neighbors)
        q, k, v = self.positional_embedding(
            x, xij, g_ii, g_ij, b_ii, b_ij, c_ij, t)
        v = self.multi_head_attention(q, k, v, mask=mask, cutoff=c_ij)
        v = self.squeeze(v)

        if self.use_feed_forward:
            return self.feed_forward(x + v)
        return x + v

    def construct(self, x, e, f_ii, f_ij, b_ii, b_ij, c_ij, neighbors, mask=None):
        """Compute convolution block.

        Args:
            x (ms.Tensor[float]): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (ms.Tensor[float]): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (ms.Tensor[int]): indices of neighbors of (N_b, N_a, N_nbh) shape.
            mask (ms.Tensor[bool]): mask to filter out non-existing neighbors
                introduced via padding.

        Returns:
            ms.Tensor: block output with (N_b, N_a, n_out) shape.

        """

        if self.use_distances and self.use_dis_filter:
            g_ii = self.dis_filter(f_ii)
            g_ij = self.dis_filter(f_ij)
        else:
            g_ii = f_ii
            g_ij = f_ij

        if self.use_bond_filter:
            b_ii = self.bond_filter(b_ii)
            b_ij = self.bond_filter(b_ij)

        if self.max_cycles == 1:
            t = self.time_embedding[0]
            x0 = self._transformer_encoder(
                x, neighbors, g_ii, g_ij, b_ii, b_ij, c_ij, t, mask)

        else:
            xx = x
            x0 = self.zeros_like(x)

            halting_prob = self.zeros((x.shape[0], x.shape[1]), ms.float32)
            n_updates = self.zeros((x.shape[0], x.shape[1]), ms.float32)

            broad_zeros = self.zeros_like(e)

            if self.flexable_cycels:
                cycle = self.zeros((), ms.int32)
                while((halting_prob < self.act_threshold).any() and (cycle < self.max_cycles)):
                    t = self.time_embedding[cycle]
                    vt = broad_zeros + t

                    xp = self.concat((xx, e, vt))
                    p = self.pondering(xp)
                    w, dp, dn = self.act_weight(p, halting_prob)
                    halting_prob = halting_prob + dp
                    n_updates = n_updates + dn

                    xx = self._transformer_encoder(
                        xx, neighbors, g_ii, g_ij, b_ii, b_ij, c_ij, t, mask)

                    cycle = cycle + 1

                    x0 = xx * w + x0 * (1.0 - w)
            else:
                for cycle in range(self.max_cycles):
                    t = self.time_embedding[cycle]
                    vt = broad_zeros + t

                    xp = self.concat((xx, e, vt))
                    p = self.pondering(xp)
                    w, dp, dn = self.act_weight(p, halting_prob)
                    halting_prob = halting_prob + dp
                    n_updates = n_updates + dn

                    xx = self._transformer_encoder(
                        xx, neighbors, g_ii, g_ij, b_ii, b_ij, c_ij, t, mask)

                    cycle = cycle + 1

                    x0 = xx * w + x0 * (1.0 - w)

        return x0

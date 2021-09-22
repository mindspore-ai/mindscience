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
"""cybertron.base"""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Constant

from .units import units
from .blocks import MLP, Dense, Residual
from .cutoff import SmoothCutoff

__all__ = [
    "GraphNorm",
    "Filter",
    "ResFilter",
    "CFconv",
    "Aggregate",
    "SmoothReciprocal",
    "SoftmaxWithMask",
    "PositionalEmbedding",
    "MultiheadAttention",
    "FeedForward",
    "Pondering",
    "ACTWeight",
    "Num2Mask",
    "Number2FullConnectNeighbors",
    "Types2FullConnectNeighbors",
]


class GraphNorm(nn.Cell):
    """graph norm"""
    def __init__(self,
                 dim_feature,
                 node_axis=-2,
                 alpha_init='one',
                 beta_init='zero',
                 gamma_init='one'
                 ):
        super().__init__()
        self.alpha = Parameter(
            initializer(
                alpha_init,
                dim_feature),
            name="alpha")
        self.beta = Parameter(initializer(beta_init, dim_feature), name="beta")
        self.gamma = Parameter(
            initializer(
                gamma_init,
                dim_feature),
            name="gamma")

        self.axis = node_axis

        self.reduce_mean = P.ReduceMean(keep_dims=True)

        self.sqrt = P.Sqrt()

    def construct(self, nodes):
        """construct"""
        mu = self.reduce_mean(nodes, self.axis)

        nodes2 = nodes * nodes
        mu2 = self.reduce_mean(nodes2, self.axis)

        a = self.alpha
        sigma2 = mu2 + (a * a - 2 * a) * mu * mu
        sigma = self.sqrt(sigma2)

        y = self.gamma * (nodes - a * mu) / sigma + self.beta

        return y


class Filter(nn.Cell):
    """filter"""
    def __init__(self,
                 num_rbf,
                 dim_filter,
                 activation,
                 n_hidden=1,
                 ):
        super().__init__()

        if n_hidden > 0:
            hidden_layers = [dim_filter for _ in range(n_hidden)]
            self.dense_layers = MLP(
                num_rbf,
                dim_filter,
                hidden_layers,
                activation=activation)
        else:
            self.dense_layers = Dense(
                num_rbf, dim_filter, activation=activation)

    def construct(self, rbf):
        """construct"""
        return self.dense_layers(rbf)


class ResFilter(nn.Cell):
    """resgilter"""
    def __init__(self,
                 num_rbf,
                 dim_filter,
                 activation,
                 n_hidden=1,
                 ):
        super().__init__()

        self.linear = Dense(num_rbf, dim_filter, activation=None)
        self.residual = Residual(
            dim_filter,
            activation=activation,
            n_hidden=n_hidden)

    def construct(self, x):
        """construct"""
        lx = self.linear(x)
        return self.residual(lx)


class CFconv(nn.Cell):
    """CFcony"""
    def __init__(self, num_rbf, dim_filter, activation,):
        super().__init__()
        # filter block used in interaction block
        self.filter = Filter(num_rbf, dim_filter, activation)

    def construct(self, x, f_ij, c_ij=None):
        w = self.filter(f_ij)
        if c_ij is not None:
            w = w * F.expand_dims(c_ij, -1)

        return x * w


class Aggregate(nn.Cell):
    """Pooling layer based on sum or average with optional masking.

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.

    """

    def __init__(self, axis, mean=False):
        super().__init__()
        self.average = mean
        self.axis = axis
        self.reduce_sum = P.ReduceSum()
        self.maximum = P.Maximum()

    def construct(self, inputs, mask=None):
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            inputs = inputs * F.expand_dims(mask, -1)
        # compute sum of input along axis

        y = self.reduce_sum(inputs, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                n = self.reduce_sum(mask, self.axis)
                n = self.maximum(n, other=F.ones_like(n))
            else:
                n = inputs.shape[self.axis]

            y = y / n
        return y


class SmoothReciprocal(nn.Cell):
    """SmoothReciprocal"""
    def __init__(self,
                 dmax=units.length(1, 'nm'),
                 cutoff_network=None
                 ):
        super().__init__()

        if cutoff_network is None:
            self.cutoff_network = SmoothCutoff(dmax, return_mask=False)
        else:
            self.cutoff_network = cutoff_network(dmax, return_mask=False)

        self.sqrt = P.Sqrt()

    def construct(self, rij, mask):
        """construct"""
        phi2rij = self.cutoff_network(rij * 2, mask)

        r_near = phi2rij * (1.0 / self.sqrt(rij * rij + 1.0))
        r_far = F.select(rij > 0, (1.0 - phi2rij) *
                         (1.0 / rij), F.zeros_like(rij))

        reciprocal = r_near + r_far
        if mask is not None:
            reciprocal = reciprocal * mask

        return reciprocal


class SoftmaxWithMask(nn.Cell):
    """SoftmaxWithMask"""
    def __init__(self, axis=-1):
        super().__init__()
        self.softmax = P.Softmax(axis)

        self.large_neg = -5e4

    def construct(self, x, mask):
        large_neg = F.ones_like(x) * self.large_neg
        xm = F.select(mask, x, large_neg)

        return self.softmax(xm)


class PositionalEmbedding(nn.Cell):
    """PositionalEmbedding"""
    def __init__(
            self,
            dim,
            use_distances=True,
            use_bonds=False,
            use_public_layer_norm=True):
        super().__init__()

        if not (use_bonds or use_distances):
            raise ValueError(
                '"use_bonds" and "use_distances" cannot be both "False" when initializing "PositionalEmbedding"!')

        self.use_distances = use_distances
        self.use_bonds = use_bonds

        if use_public_layer_norm:
            self.norm = nn.LayerNorm((dim,), -1, -1)
            self.norm_q = self.norm
            self.norm_k = self.norm
            self.norm_v = self.norm
        else:
            self.norm_q = nn.LayerNorm((dim,), -1, -1)
            self.norm_k = nn.LayerNorm((dim,), -1, -1)
            self.norm_v = nn.LayerNorm((dim,), -1, -1)

        self.x2q = Dense(dim, dim, has_bias=False)
        self.x2k = Dense(dim, dim, has_bias=False)
        self.x2v = Dense(dim, dim, has_bias=False)

        self.mul = P.Mul()
        self.concat = P.Concat(-2)

    def construct(
            self,
            xi,
            xij,
            g_ii=1,
            g_ij=1,
            b_ii=0,
            b_ij=0,
            c_ij=None,
            t=0):
        r"""Get query, key and query from atom types and positions

        Args:
            xi   (Mindspore.Tensor [B, A, F]):
            g_ii (Mindspore.Tensor [B, A, F]):
            xij  (Mindspore.Tensor [B, A, N, F]):
            g_ij (Mindspore.Tensor [B, A, N, F]):
            t    (Mindspore.Tensor [F]):

        Marks:
            B:  Batch size
            A:  Number of atoms
            N:  Number of neighbor atoms
            N': Number of neighbor atoms and itself (N' = N + 1)
            F:  Dimensions of feature space

        Returns:
            query  (Mindspore.Tensor [B, A, 1, F]):
            key    (Mindspore.Tensor [B, A, N', F]):
            value  (Mindspore.Tensor [B, A, N', F]):

        """

        if self.use_distances:
            # [B, A, F] * [B, A, F] + [B, A, F] = [B, A, F]
            a_ii = xi * g_ii
            # [B, A, N, F] * [B, A, N, F] + [B, A, N, F] = [B, A, N, F]
            a_ij = xij * g_ij
        else:
            a_ii = xi
            a_ij = xij

        if self.use_bonds:
            e_ii = a_ii + b_ii
            e_ij = a_ij + b_ij
        else:
            e_ii = a_ii
            e_ij = a_ij

        # [B, A, 1, F]
        e_ii = F.expand_dims(e_ii, -2)
        # [B, A, N', F] + [B, A, N', F]
        e_ij = self.concat((e_ii, e_ij))

        xq = self.norm_q(e_ii + t)
        xk = self.norm_k(e_ij + t)
        xv = self.norm_v(e_ij + t)
        # [B, A, 1, F]
        query = self.x2q(xq)
        # [B, A, N', F]
        key = self.x2k(xk)
        # [B, A, N', F]
        value = self.x2v(xv)

        if c_ij is not None:
            # [B, A, N', F] * [B, A, N', 1]
            key = key * F.expand_dims(c_ij, -1)
            value = value * F.expand_dims(c_ij, -1)

        return query, key, value


class MultiheadAttention(nn.Cell):
    r"""Compute multi-head attention.

    Args:
        dim_feature (int): Diension of feature space (F)
        n_heads     (int): Number of heads (h)
        dim_tensor  (int): Dimension of input tensor (D)

    Signs:
        X:  Dimension to be aggregated
        F:  Dimension of Feature space
        h:  Number of heads for multi-head attention
        f:  Dimensions per head (F = f * h)

    """

    def __init__(self, dim_feature, n_heads=8, dim_tensor=4):
        super().__init__()

        # D
        if dim_tensor < 2:
            raise ValueError('dim_tensor must be larger than 1')

        # h
        self.n_heads = n_heads

        # f = F / h
        self.size_per_head = dim_feature // n_heads
        # 1.0 / sqrt(f)
        scores_mul = 1.0 / np.sqrt(float(self.size_per_head))
        self.scores_mul = ms.Tensor(scores_mul, ms.float32)

        # shape = (h, f)
        self.reshape_tail = (self.n_heads, self.size_per_head)

        self.output = Dense(dim_feature, dim_feature, has_bias=False)

        self.mul = P.Mul()
        self.div = P.Div()
        self.softmax = P.Softmax()
        self.bmm = P.BatchMatMul()
        self.bmmt = P.BatchMatMul(transpose_b=True)
        self.reducesum = P.ReduceSum(keep_dims=True)

        # [0,1,...,D-1]
        ranges = list(range(dim_tensor + 1))
        tmpid = ranges[-2]
        ranges[-2] = ranges[-3]
        ranges[-3] = tmpid
        # [0,1,...,D-2,D-3,D-1]
        self.trans_shape = tuple(ranges)
        self.transpose = P.Transpose()

        self.softmax_with_mask = SoftmaxWithMask()

    def construct(self, query, key, value, mask=None, cutoff=None):
        r"""Compute multi-head attention.

        Args:
            query  (Mindspore.Tensor [..., 1, F] or [..., X, F]):
            key    (Mindspore.Tensor [..., X, F]):
            value  (Mindspore.Tensor [..., X, F]):
            mask   (Mindspore.Tensor [..., X]):
            cutoff (Mindspore.Tensor [..., X]):

        Returns:
            Mindspore.Tensor [..., F]: multi-head attention output.

        """
        if self.n_heads > 1:
            q_reshape = query.shape[:-1] + self.reshape_tail
            k_reshape = key.shape[:-1] + self.reshape_tail
            v_reshape = value.shape[:-1] + self.reshape_tail

            # [..., 1, h, f] or [..., X, h, f]
            q = F.reshape(query, q_reshape)
            # [..., h, 1, f] or [..., h, X, f]
            q = self.transpose(q, self.trans_shape)

            # [..., X, h, f]
            k = F.reshape(key, k_reshape)
            # [..., h, X, f]
            k = self.transpose(k, self.trans_shape)

            # [..., X, h, f]
            v = F.reshape(value, v_reshape)
            # [..., h, X, f]
            v = self.transpose(v, self.trans_shape)

            # [..., h, 1, f] x [..., h, X, f]^T = [..., h, 1, X]
            # or
            # [..., h, X, f] x [..., h, X, f]^T = [..., h, X, X]
            attention_scores = self.bmmt(q, k)
            # ([..., h, 1, X] or [..., h, X, X]) / sqrt(f)
            attention_scores = self.mul(attention_scores, self.scores_mul)

            if mask is None:
                # [..., h, 1, X] or [..., h, X, X]
                attention_probs = self.softmax(attention_scores)
            else:
                # [..., X] -> [..., 1, 1, X]
                exmask = F.expand_dims(F.expand_dims(mask, -2), -2)
                # [..., 1, 1, X] -> ([..., h, 1, X] or [..., h, X, X])
                mhmask = (exmask * F.ones_like(attention_scores)) > 0
                # [..., h, 1, X] or [..., h, X, X]
                attention_probs = self.softmax_with_mask(
                    attention_scores, mhmask)

                if cutoff is not None:
                    # [..., X] -> [..., 1, 1, X]
                    excut = F.expand_dims(F.expand_dims(cutoff, -2), -2)
                    attention_probs = self.mul(attention_probs, excut)

            context = self.bmm(attention_probs, v)
            # [..., 1, h, f] or [..., X, h, f]
            context = self.transpose(context, self.trans_shape)
            # [..., 1, F] or [..., X, F]
            context = F.reshape(context, query.shape)

        else:
            attention_scores = self.bmmt(query, key) * self.scores_mul

            if mask is None:
                # [..., 1, X] or [..., X, X]
                attention_probs = self.softmax(attention_scores)
            else:
                # [..., X] -> [..., 1, X]
                mask = F.expand_dims(mask, -2)
                # [..., 1, X]
                attention_probs = self.softmax_with_mask(
                    attention_scores, mask)

                if cutoff is not None:
                    # [..., 1, X] * [..., 1, X]
                    attention_probs = attention_probs * \
                        F.expand_dims(cutoff, -2)

            # [..., 1, X] x [..., X, F] = [..., 1, F]
            # or
            # [..., X, X] x [..., X, F] = [..., X, F]
            context = self.bmm(attention_probs, value)

        # [..., 1, F] or [..., X, F]
        return self.output(context)


class FeedForward(nn.Cell):
    def __init__(self, dim, activation, n_hidden=1):
        super().__init__()

        self.norm = nn.LayerNorm((dim,), -1, -1)
        self.residual = Residual(dim, activation=activation, n_hidden=n_hidden)

    def construct(self, x):
        nx = self.norm(x)
        return self.residual(nx)


class Pondering(nn.Cell):
    """Pondering"""
    def __init__(self, n_in, n_hidden=0, bias_const=1.):
        super().__init__()

        if n_hidden == 0:
            self.dense = nn.Dense(
                n_in,
                1,
                has_bias=True,
                weight_init='xavier_uniform',
                bias_init=Constant(bias_const),
                activation='sigmoid',
            )
        elif n_hidden > 0:
            nets = []
            for _ in range(n_hidden):
                nets.append(nn.Dense(n_in, n_in, weight_init='xavier_uniform', activation='relu'))
            nets.append(nn.Dense(n_in, 1, bias_init=Constant(bias_const), activation='sigmoid'))
            self.dense = nn.SequentialCell(nets)
        else:
            raise ValueError("n_hidden cannot be negative!")

        self.squeeze = P.Squeeze(-1)

    def construct(self, x):
        y = self.dense(x)
        return self.squeeze(y)

# Modified from:
# https://github.com/andreamad8/Universal-Transformer-Pytorch/blob/master/models/UTransformer.py


class ACTWeight(nn.Cell):
    """ACTWeight"""
    def __init__(self, threshold=0.9):
        super().__init__()
        self.threshold = threshold

        self.zeros_like = P.ZerosLike()
        self.ones_like = P.OnesLike()

    def construct(self, prob, halting_prob):
        """construct"""

        # Mask for inputs which have not halted last cy
        running = F.cast(halting_prob < 1.0, ms.float32)

        # Add the halting probability for this step to the halting
        # probabilities for those input which haven't halted yet
        add_prob = prob * running
        new_prob = halting_prob + add_prob
        mask_run = F.cast(new_prob <= self.threshold, ms.float32)
        mask_halt = F.cast(new_prob > self.threshold, ms.float32)

        # Mask of inputs which haven't halted, and didn't halt this step
        still_running = mask_run * running
        running_prob = halting_prob + prob * still_running

        # Mask of inputs which halted at this step
        new_halted = mask_halt * running

        # Compute remainders for the inputs which halted at this step
        remainders = new_halted * (1.0 - running_prob)

        # Add the remainders to those inputs which halted at this step
        # halting_prob = new_prob + remainders
        dp = add_prob + remainders

        # Increment n_updates for all inputs which are still running
        # n_updates = n_updates + running
        dn = running

        # Compute the weight to be applied to the new state and output
        # 0 when the input has already halted
        # prob when the input hasn't halted yet
        # the remainders when it halted this step
        update_weights = prob * still_running + new_halted * remainders
        w = F.expand_dims(update_weights, -1)

        return w, dp, dn


class Num2Mask(nn.Cell):
    """Num2Mask"""
    def __init__(self, dim):
        super().__init__()
        self.range = nn.Range(dim)
        ones = P.Ones()
        self.ones = ones((dim), ms.int32)

    def construct(self, num):
        nmax = num * self.ones
        idx = F.ones_like(num) * self.range()
        return idx < nmax


class Number2FullConnectNeighbors(nn.Cell):
    """Number2FullConnectNeighbors"""
    def __init__(self, tot_atoms):
        super().__init__()
        # tot_atoms: A
        # tot_neigh: N =  A - 1
        tot_neigh = tot_atoms - 1
        arange = nn.Range(tot_atoms)
        nrange = nn.Range(tot_neigh)

        self.ones = P.Ones()
        self.aones = self.ones((tot_atoms), ms.int32)
        self.nones = self.ones((tot_neigh), ms.int32)

        # neighbors for no connection (A*N)
        # [[0,0,...,0],
        #  [1,1,...,1],
        #  ...........,
        #  [N,N,...,N]]
        self.nnc = F.expand_dims(arange(), -1) * self.nones
        # copy of the index range (A*N)
        # [[0,1,...,N-1],
        #  [0,1,...,N-1],
        #  ...........,
        #  [0,1,...,N-1]]
        crange = self.ones((tot_atoms, 1), ms.int32) * nrange()
        # neighbors for full connection (A*N)
        # [[1,2,3,...,N],
        #  [0,2,3,...,N],
        #  [0,1,3,....N],
        #  .............,
        #  [0,1,2,...,N-1]]
        self.nfc = crange + F.cast(self.nnc <= crange, ms.int32)

        crange1 = crange + 1
        # the matrix for index range (A*N)
        # [[1,2,3,...,N],
        #  [1,2,3,...,N],
        #  [2,2,3,....N],
        #  [3,3,3,....N],
        #  .............,
        #  [N,N,N,...,N]]
        self.mat_idx = F.select(crange1 > self.nnc, crange1, self.nnc)

    def get_full_neighbors(self):
        """get_full_neighbors"""
        return F.expand_dims(self.nfc, 0)

    def construct(self, num_atoms):
        """construct"""
        # broadcast atom numbers to [B*A*N]
        # a_i: number of atoms in each molecule
        exnum = num_atoms * self.aones
        exnum = F.expand_dims(exnum, -1) * self.nones

        # [B,1,1]
        exones = self.ones((num_atoms.shape[0], 1, 1), ms.int32)
        # broadcast to [B*A*N]: [B,1,1] * [1,A,N]
        exnfc = exones * F.expand_dims(self.nfc, 0)
        exnnc = exones * F.expand_dims(self.nnc, 0)
        exmat = exones * F.expand_dims(self.mat_idx, 0)

        mask = exmat < exnum

        neighbors = F.select(mask, exnfc, exnnc)

        return neighbors, mask


class Types2FullConnectNeighbors(nn.Cell):
    """Types2FullConnectNeighbors"""
    def __init__(self, tot_atoms):
        super().__init__()
        # tot_atoms: A
        # tot_neigh: N =  A - 1
        tot_neigh = tot_atoms - 1
        arange = nn.Range(tot_atoms)
        nrange = nn.Range(tot_neigh)

        self.ones = P.Ones()
        self.aones = self.ones((tot_atoms), ms.int32)
        self.nones = self.ones((tot_neigh), ms.int32)
        self.eaones = F.expand_dims(self.aones, -1)

        # neighbors for no connection (A*N)
        # [[0,0,...,0],
        #  [1,1,...,1],
        #  ...........,
        #  [N,N,...,N]]
        self.nnc = F.expand_dims(arange(), -1) * self.nones

        # copy of the index range (A*N)
        # [[0,1,...,N-1],
        #  [0,1,...,N-1],
        #  ...........,
        #  [0,1,...,N-1]]
        exrange = self.ones((tot_atoms, 1), ms.int32) * nrange()

        # neighbors for full connection (A*N)
        # [[1,2,3,...,N],
        #  [0,2,3,...,N],
        #  [0,1,3,....N],
        #  .............,
        #  [0,1,2,...,N-1]]
        self.nfc = exrange + F.cast(self.nnc <= exrange, ms.int32)

        self.ar0 = nn.Range(0, tot_neigh)()
        self.ar1 = nn.Range(1, tot_atoms)()

    def get_full_neighbors(self):
        """get_full_neighbors"""
        return F.expand_dims(self.nfc, 0)

    def construct(self, atom_types):
        """construct"""
        # [B,1,1]
        exones = self.ones((atom_types.shape[0], 1, 1), ms.int32)
        # broadcast to [B*A*N]: [B,1,1] * [1,A,N]
        exnfc = exones * F.expand_dims(self.nfc, 0)
        exnnc = exones * F.expand_dims(self.nnc, 0)

        tmask = F.select(
            atom_types > 0,
            F.ones_like(atom_types),
            F.ones_like(atom_types) * -1)
        tmask = F.cast(tmask, ms.float32)
        extmask = F.expand_dims(tmask, -1) * self.nones

        mask0 = F.gather(tmask, self.ar0, -1)
        mask0 = F.expand_dims(mask0, -2) * self.eaones
        mask1 = F.gather(tmask, self.ar1, -1)
        mask1 = F.expand_dims(mask1, -2) * self.eaones

        mtmp = F.select(exnfc > exnnc, mask1, mask0)
        mask = F.select(extmask > 0, mtmp, F.ones_like(mtmp) * -1)
        mask = mask > 0

        idx = F.select(mask, exnfc, exnnc)

        return idx, mask

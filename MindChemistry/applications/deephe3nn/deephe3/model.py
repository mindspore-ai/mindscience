# Copyright 2024 Huawei Technologies Co., Ltd
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
model
"""
# pylint: disable=W0102

import warnings
import os
import math
from mindchemistry.e3.nn.gate import Gate
from mindchemistry.e3.o3.irreps import Irreps
from mindchemistry.e3.o3.sub import Linear, FullyConnectedTensorProduct, LinearBias
from mindchemistry.e3.o3.spherical_harmonics import SphericalHarmonics
from mindchemistry.so2_conv import SO3Rotation, SO2Convolution
from mindchemistry.so2_conv.init_edge_rot_mat import init_edge_rot_mat
from mindchemistry.graph.graph import LiftNodeToEdge, Aggregate
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.initializer import Uniform
from .utils import tp_path_exists, GaussianBasis
from .e3modules import SeparateWeightTensorProduct, E3ElementWise, E3LayerNorm, SkipConnection, SelfTp, SortIrreps

epsilon = 1e-8


def get_gate_nonlin(irreps_in1,
                    irreps_in2,
                    irreps_out,
                    act={
                        1: ops.tanh,
                        -1: ops.tanh
                    },
                    act_gates={
                        1: ops.sigmoid,
                        -1: ops.tanh
                    }):
    """
    get gate nonlinearity after tensor product
    irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
    irreps_out is desired irreps after gate nonlin
    notice that nonlin.irreps_out might not be exactly equal to irreps_out
    """
    irreps_scalars = Irreps([
        (mul, ir) for mul, ir in irreps_out
        if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()

    irreps_gated = Irreps([
        (mul, ir) for mul, ir in irreps_out
        if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()

    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            ir = "0o"
            warnings.warn('Using odd representations as gates')
    else:
        ir = None

    irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

    gate_nonlin = Gate(
        irreps_scalars,
        [act.get(ir.p, None) for _, ir in irreps_scalars],  # scalar
        irreps_gates,
        [act_gates.get(ir.p, None) for _, ir in irreps_gates],  # gates (scalars)
        irreps_gated,  # gated tensors
        ncon_dtype=ms.float16)

    return gate_nonlin


class EquiConv(nn.Cell):
    """
    Equivariant Convolutional Networks
    """

    def __init__(self,
                 fc_len_in,
                 irreps_in1,
                 irreps_in2,
                 irreps_out,
                 norm='',
                 nonlin=True,
                 act={
                     1: ops.tanh,
                     -1: ops.tanh
                 },
                 act_gates={
                     1: ops.sigmoid,
                     -1: ops.tanh
                 },
                 escn=False):
        super(EquiConv, self).__init__()

        self.escn = escn
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)

        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out,
                                          act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps([
                (mul, ir) for mul, ir in irreps_out
                if tp_path_exists(irreps_in1, irreps_in2, ir)
            ])

        if nonlin:
            self.cfconv = E3ElementWise(self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = E3ElementWise(irreps_tp_out)
            self.irreps_out = irreps_tp_out

        # fully connected net to create tensor product weights
        linear_act = Silu()

        sqrt128 = math.sqrt(128)
        sqrt64 = math.sqrt(64)


        if sqrt64 != 0:
            weightinit2 = Uniform(scale=1 / sqrt64)
            weightinit3 = Uniform(scale=1 / sqrt64)
            biasinit2 = Uniform(scale=1 / sqrt64)
            biasinit3 = Uniform(scale=1 / sqrt64)
        else:
            raise ValueError

        if sqrt128 != 0:
            biasinit1 = Uniform(scale=1 / sqrt128)
            weightinit1 = Uniform(scale=1 / sqrt128)
        else:
            raise ValueError

        self.fc = nn.SequentialCell(
            nn.Dense(fc_len_in,
                     64,
                     weight_init=weightinit1,
                     bias_init=biasinit1).to_float(ms.float16), linear_act,
            nn.Dense(64, 64, weight_init=weightinit2,
                     bias_init=biasinit2).to_float(ms.float16), linear_act,
            nn.Dense(64,
                     self.cfconv.len_weight,
                     weight_init=weightinit3,
                     bias_init=biasinit3).to_float(ms.float16))

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = E3LayerNorm(self.cfconv.irreps_in)
            else:
                raise ValueError(f'unknown norm: {norm}')

        if self.escn is True:
            self.so3_rotation = SO3Rotation(5, str(irreps_in1), str(irreps_tp_out))
            self.so2_conv = SO2Convolution(str(irreps_in1), str(irreps_tp_out))
        else:
            self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2,
                                                  irreps_tp_out)

    def construct(self,
                  fea_in1,
                  fea_in2,
                  fea_weight,
                  batch_edge,
                  ms_rotate_mat=None):
        """
        Equivariant Convolutional Networks construct process
        """
        if self.escn is True:
            wigner, wigner_inv = self.so3_rotation.set_wigner(ms_rotate_mat)
            embedding_rotate = self.so3_rotation.rotate(fea_in1, wigner)
            so2_conv_res = self.so2_conv(embedding_rotate, fea_weight)
            z = self.so3_rotation.rotate_inv(so2_conv_res, wigner_inv)
        else:
            z = self.tp(fea_in1, fea_in2)

        if self.nonlin is not None:
            z = self.nonlin(z)

        weight = self.fc(fea_weight)
        z = self.cfconv(z, weight)

        if self.norm is not None:
            z = self.norm(z, batch_edge)
        return z


class Silu(nn.Cell):
    """
    Silu activation class
    """

    def __init__(self):
        super(Silu, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        """
        Silu activation class construct process
        """
        return ops.mul(x, self.sigmoid(x))


class NodeUpdateBlock(nn.Cell):
    """
    Block class to update the Node information
    """

    def __init__(self,
                 num_species,
                 fc_len_in,
                 irreps_sh,
                 irreps_in_node,
                 irreps_out_node,
                 irreps_in_edge,
                 act,
                 act_gates,
                 use_selftp=False,
                 use_sc=True,
                 concat=True,
                 only_ij=False,
                 nonlin=False,
                 norm='e3LayerNorm',
                 if_sort_irreps=False,
                 escn=False):
        super(NodeUpdateBlock, self).__init__()

        self.escn = escn
        irreps_in_node = Irreps(irreps_in_node)
        irreps_sh = Irreps(irreps_sh)
        irreps_out_node = Irreps(irreps_out_node)
        irreps_in_edge = Irreps(irreps_in_edge)

        if concat:
            irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
            if if_sort_irreps:
                self.sort_node = SortIrreps(irreps_in1)
                irreps_in1 = self.sort_node.irreps_out
        else:
            irreps_in1 = irreps_in_node
        irreps_in2 = irreps_sh

        self.lin_pre_node = LinearBias(irreps_in=irreps_in_node,
                                       irreps_out=irreps_in_node,
                                       has_bias=True,
                                       ncon_dtype=ms.float16)

        self.nonlin_node = None
        if nonlin:
            self.nonlin_node = get_gate_nonlin(irreps_in1, irreps_in2,
                                               irreps_out_node, act, act_gates)
            irreps_conv_out = self.nonlin_node.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_node
            conv_nonlin = True

        self.conv_node = EquiConv(fc_len_in,
                                  irreps_in1,
                                  irreps_in2,
                                  irreps_conv_out,
                                  nonlin=conv_nonlin,
                                  act=act,
                                  act_gates=act_gates,
                                  escn=self.escn)

        self.lin_post_node = LinearBias(irreps_in=self.conv_node.irreps_out,
                                        irreps_out=self.conv_node.irreps_out,
                                        has_bias=True,
                                        ncon_dtype=ms.float16)

        if nonlin:
            self.irreps_out = self.nonlin_node.irreps_out
        else:
            self.irreps_out = self.conv_node.irreps_out

        self.sc_node = None
        if use_sc:
            self.sc_node = FullyConnectedTensorProduct(
                irreps_in_node,
                f'{num_species}x0e',
                self.conv_node.irreps_out,
                ncon_dtype=ms.float16)

        self.norm_node = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm_node = E3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')

        self.skip_connect_node = SkipConnection(irreps_in_node,
                                                self.irreps_out)

        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.irreps_in_node = irreps_in_node
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps

        self.node2edge_i = LiftNodeToEdge(0)
        self.node2edge_j = LiftNodeToEdge(1)

        self.aggregrate = Aggregate()

    def construct(self,
                  node_fea,
                  node_one_hot,
                  edge_sh,
                  edge_fea,
                  edge_length_embedded,
                  edge_index,
                  batch,
                  batch_input_x,
                  mask_dim2,
                  ms_rotate_mat=None):
        """
        Block class to update the Node information construct part
        """
        node_fea_old = node_fea
        node_self_connection = None

        if self.use_sc:
            node_self_connection = self.sc_node(node_fea, node_one_hot)
        node_fea = self.lin_pre_node(node_fea)

        fea_in = None
        if self.concat:
            ##### for dynamic shape ###
            vi = self.node2edge_i(node_fea, edge_index)
            vj = self.node2edge_j(node_fea, edge_index)

            vi = ops.mul(vi, mask_dim2)
            vj = ops.mul(vj, mask_dim2)

            fea_in = ops.cat([vi, vj, edge_fea], axis=-1)

            if self.if_sort_irreps:
                fea_in = self.sort_node.forward(fea_in)
            if self.escn:
                edge_update = self.conv_node(fea_in, edge_sh,
                                             edge_length_embedded,
                                             batch[edge_index[0]],
                                             ms_rotate_mat)
            else:
                edge_update = self.conv_node(fea_in, edge_sh,
                                             edge_length_embedded,
                                             batch[edge_index[0]])
        else:
            if self.escn:
                edge_update = self.conv_node(node_fea[edge_index[1]], edge_sh,
                                             edge_length_embedded,
                                             batch[edge_index[0]],
                                             ms_rotate_mat)
            else:
                edge_update = self.conv_node(node_fea[edge_index[1]], edge_sh,
                                             edge_length_embedded,
                                             batch[edge_index[0]])

        node_fea = self.aggregrate.scatter_sum(
            edge_update.astype(ms.float16), edge_index[0],
            batch_input_x.astype(ms.float16)).astype(ms.float32)

        node_fea = self.lin_post_node(node_fea)

        if self.use_sc:
            node_fea = node_fea + node_self_connection
        if self.nonlin_node is not None:
            node_fea = self.nonlin_node(node_fea)
        if self.norm_node is not None:
            ##### for dynamic shape ###
            node_fea = self.norm_node(node_fea, batch)

        node_fea = self.skip_connect_node(node_fea_old, node_fea)
        if self.self_tp is not None:
            node_fea = self.self_tp(node_fea)

        return node_fea


class EdgeUpdateBlock(nn.Cell):
    """
    Block class to update the edge information
    """

    def __init__(self,
                 num_species,
                 fc_len_in,
                 irreps_sh,
                 irreps_in_node,
                 irreps_in_edge,
                 irreps_out_edge,
                 act,
                 act_gates,
                 use_selftp=False,
                 use_sc=True,
                 init_edge=False,
                 nonlin=False,
                 norm='e3LayerNorm',
                 if_sort_irreps=False,
                 escn=False):
        super(EdgeUpdateBlock, self).__init__()

        self.escn = escn
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort_node = SortIrreps(irreps_in1)
            irreps_in1 = self.sort_node.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre_edge = LinearBias(irreps_in=irreps_in_edge,
                                       irreps_out=irreps_in_edge,
                                       has_bias=True,
                                       ncon_dtype=ms.float16)

        self.nonlin_edge = None
        self.lin_post_edge = None
        if nonlin:
            self.nonlin_edge = get_gate_nonlin(irreps_in1, irreps_in2,
                                               irreps_out_edge, act, act_gates)
            irreps_conv_out = self.nonlin_edge.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True

        self.conv_edge = EquiConv(fc_len_in,
                                  irreps_in1,
                                  irreps_in2,
                                  irreps_conv_out,
                                  nonlin=conv_nonlin,
                                  act=act,
                                  act_gates=act_gates,
                                  escn=self.escn)

        self.lin_post_edge = LinearBias(irreps_in=self.conv_edge.irreps_out,
                                        irreps_out=self.conv_edge.irreps_out,
                                        has_bias=True,
                                        ncon_dtype=ms.float16)

        if use_sc:
            self.sc_edge = FullyConnectedTensorProduct(
                irreps_in_edge,
                f'{num_species ** 2}x0e',
                self.conv_edge.irreps_out,
                ncon_dtype=ms.float16)

        if nonlin:
            self.irreps_out = self.nonlin_edge.irreps_out
        else:
            self.irreps_out = self.conv_edge.irreps_out

        self.norm_edge = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm_edge = E3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')

        self.skip_connect_edge = SkipConnection(
            irreps_in_edge, self.irreps_out)  # ! consider init_edge

        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge

        self.node2edge_i = LiftNodeToEdge(0)
        self.node2edge_j = LiftNodeToEdge(1)

    def construct(self,
                  node_fea,
                  edge_one_hot,
                  edge_sh,
                  edge_fea,
                  edge_length_embedded,
                  edge_index,
                  batch,
                  mask_length,
                  mask_dim1,
                  mask_dim2,
                  mask_dim3,
                  ms_rotate_mat=None):
        """
        Block class to update the edge information construct class
        """
        edge_self_connection = None
        edge_fea_old = None

        if not self.init_edge:
            edge_fea_old = edge_fea
            if self.use_sc:
                edge_self_connection = self.sc_edge(edge_fea, edge_one_hot)

        edge_fea = self.lin_pre_edge(edge_fea)

        ##### for dynamic shape ###
        vi = self.node2edge_i(node_fea, edge_index)
        vj = self.node2edge_j(node_fea, edge_index)

        vi = ops.mul(vi, mask_dim2)
        vj = ops.mul(vj, mask_dim2)

        fea_in = ops.cat((vi, vj, edge_fea), axis=-1)

        if self.if_sort_irreps:
            fea_in = self.sort_node.forward(fea_in)
        if self.escn:
            edge_fea = self.conv_edge(fea_in, edge_sh, edge_length_embedded,
                                      batch[edge_index[0]], ms_rotate_mat)
        else:
            edge_fea = self.conv_edge(fea_in, edge_sh, edge_length_embedded,
                                      batch[edge_index[0]])

        edge_fea = self.lin_post_edge(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection

        if self.nonlin_edge is not None:
            edge_fea = self.nonlin_edge(edge_fea)
        if self.norm_edge is not None:
            ##### for dynamic shape ###
            edge_fea = self.norm_edge(edge_fea, batch[edge_index[0]],
                                      mask_length, mask_dim1, mask_dim3)

        if not self.init_edge:
            edge_fea = self.skip_connect_edge(edge_fea_old, edge_fea)
        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea


class Net(nn.Cell):
    """
    Main network class
    """

    def __init__(self,
                 num_species,
                 irreps_embed_node,
                 irreps_edge_init,
                 irreps_sh,
                 irreps_mid_node,
                 irreps_post_node,
                 irreps_out_node,
                 irreps_mid_edge,
                 irreps_post_edge,
                 irreps_out_edge,
                 num_block,
                 r_max,
                 use_sc=True,
                 no_parity=False,
                 use_sbf=True,
                 selftp=False,
                 edge_upd=True,
                 only_ij=False,
                 num_basis=128,
                 act={
                     1: ops.tanh,
                     -1: ops.tanh
                 },
                 act_gates={
                     1: ops.sigmoid,
                     -1: ops.tanh
                 },
                 if_sort_irreps=False,
                 escn=False):

        super(Net, self).__init__()

        self.escn = escn
        self.num_species = num_species
        self.only_ij = only_ij

        irreps_embed_node = Irreps(irreps_embed_node)
        self.embedding = Linear(irreps_in=f"{num_species}x0e",
                                irreps_out=irreps_embed_node,
                                ncon_dtype=ms.float16)

        self.basis = GaussianBasis(start=0.0,
                                   stop=r_max,
                                   n_gaussians=num_basis,
                                   trainable=False)

        # distance expansion to initialize edge feature
        irreps_edge_init = Irreps(irreps_edge_init)

        self.distance_expansion = GaussianBasis(
            start=0.0,
            stop=6.0,
            n_gaussians=irreps_edge_init.dim,
            trainable=False)

        self.sh = SphericalHarmonics(
            irreps_out=irreps_sh,
            normalize=True,
            normalization='component',
        )

        self.use_sbf = use_sbf

        if no_parity:
            irreps_sh = Irreps([(mul, (ir.l, 1))
                                for mul, ir in Irreps(irreps_sh)])
        self.irreps_sh = irreps_sh

        irreps_node_prev = self.embedding.irreps_out
        irreps_edge_prev = irreps_edge_init

        self.node_update_blocks = nn.CellList([])
        self.edge_update_blocks = nn.CellList([])

        for index_block in range(num_block):
            if index_block == num_block - 1:
                silu = Silu()
                act = {1: silu, -1: ops.tanh}
                node_update_block = NodeUpdateBlock(
                    num_species,
                    num_basis,
                    irreps_sh,
                    irreps_node_prev,
                    irreps_post_node,
                    irreps_edge_prev,
                    act,
                    act_gates,
                    use_selftp=selftp,
                    use_sc=use_sc,
                    only_ij=only_ij,
                    if_sort_irreps=if_sort_irreps,
                    escn=self.escn)
                edge_update_block = EdgeUpdateBlock(
                    num_species,
                    num_basis,
                    irreps_sh,
                    node_update_block.irreps_out,
                    irreps_edge_prev,
                    irreps_post_edge,
                    act,
                    act_gates,
                    use_selftp=selftp,
                    use_sc=use_sc,
                    if_sort_irreps=if_sort_irreps,
                    escn=self.escn)
            else:
                node_update_block = NodeUpdateBlock(
                    num_species,
                    num_basis,
                    irreps_sh,
                    irreps_node_prev,
                    irreps_mid_node,
                    irreps_edge_prev,
                    act,
                    act_gates,
                    use_selftp=False,
                    use_sc=use_sc,
                    only_ij=only_ij,
                    if_sort_irreps=if_sort_irreps,
                    escn=self.escn)
                edge_update_block = None
                if edge_upd:
                    edge_update_block = EdgeUpdateBlock(
                        num_species,
                        num_basis,
                        irreps_sh,
                        node_update_block.irreps_out,
                        irreps_edge_prev,
                        irreps_mid_edge,
                        act,
                        act_gates,
                        use_selftp=False,
                        use_sc=use_sc,
                        if_sort_irreps=if_sort_irreps,
                        escn=self.escn)

            irreps_node_prev = node_update_block.irreps_out
            if edge_update_block is not None:
                irreps_edge_prev = edge_update_block.irreps_out

            self.node_update_blocks.append(node_update_block)
            self.edge_update_blocks.append(edge_update_block)

        irreps_out_edge = Irreps(irreps_out_edge)

        self.irreps_out_node = irreps_out_node
        self.irreps_out_edge = irreps_out_edge

        self.lin_node = LinearBias(irreps_in=irreps_node_prev,
                                   irreps_out=irreps_out_node,
                                   has_bias=True,
                                   ncon_dtype=ms.float16)
        self.lin_edge = LinearBias(irreps_in=irreps_edge_prev,
                                   irreps_out=irreps_out_edge,
                                   has_bias=True,
                                   ncon_dtype=ms.float16)
    def __repr__(self):
        info = '===== DeepH-E3 model structure: ====='
        if self.use_sbf:
            info += f'\nusing spherical bessel basis: {self.irreps_sh}'
        else:
            info += f'\nusing spherical harmonics: {self.irreps_sh}'
        for index, (nupd, eupd) in enumerate(
                zip(self.node_update_blocks, self.edge_update_blocks)):
            info += f'\n=== layer {index} ==='
            info += f'\nnode update: ({nupd.irreps_in_node} -> {nupd.irreps_out})'
            if eupd is not None:
                info += f'\nedge update: ({eupd.irreps_in_edge} -> {eupd.irreps_out})'
        info += '\n=== output ==='
        info += f'\noutput node: ({self.irreps_out_node})'
        info += f'\noutput edge: ({self.irreps_out_edge})'

        return info
    def construct(self, data_x, data_edge_index, data_edge_attr,
                  data_mask_length, batch_input_x, mask_dim1, mask_dim2,
                  mask_dim3):
        """
        Main network class construct process
        """
        node_one_hot = ops.one_hot(data_x, self.num_species, 1.0, 0.0, axis=-1)
        edge_one_hot = ops.one_hot(
            self.num_species * data_x[data_edge_index[0]] +
            data_x[data_edge_index[1]],
            self.num_species**2,
            1.0,
            0.0,
            axis=-1)

        node_fea = self.embedding(node_one_hot)

        edge_length = data_edge_attr[:, 0]
        edge_vec = ops.stack(
            (data_edge_attr[:, 2], data_edge_attr[:, 3], data_edge_attr[:, 1]),
            axis=-1)  # (y, z, x) order

        ms_rotate_mat = None
        if self.escn:
            ms_rotate_mat = init_edge_rot_mat(edge_vec)

        if self.use_sbf:
            edge_sh = self.sh(edge_length, edge_vec)
        else:
            edge_sh = self.sh(edge_vec)

        edge_length_embedded = self.basis(edge_length)

        ##### for dynamic shape ###
        edge_length_embedded = ops.mul(edge_length_embedded, mask_dim2)

        edge_fea = self.distance_expansion(edge_length)

        ##### for dynamic shape ###
        edge_fea = ops.mul(edge_fea, mask_dim2)

        index = 0
        for node_update_block, edge_update_block in zip(
                self.node_update_blocks, self.edge_update_blocks):
            data_batch = ops.zeros((ops.shape(data_x)[0]), ms.int64)
            if self.escn:
                node_fea = node_update_block(node_fea, node_one_hot, edge_sh,
                                             edge_fea, edge_length_embedded,
                                             data_edge_index, data_batch,
                                             batch_input_x, mask_dim2,
                                             ms_rotate_mat)
            else:
                node_fea = node_update_block(node_fea, node_one_hot, edge_sh,
                                             edge_fea, edge_length_embedded,
                                             data_edge_index, data_batch,
                                             batch_input_x, mask_dim2)
            if edge_update_block is not None:
                if self.escn:
                    edge_fea = edge_update_block(
                        node_fea, edge_one_hot, edge_sh, edge_fea,
                        edge_length_embedded, data_edge_index, data_batch,
                        data_mask_length, mask_dim1, mask_dim2, mask_dim3,
                        ms_rotate_mat)
                else:
                    edge_fea = edge_update_block(node_fea, edge_one_hot,
                                                 edge_sh, edge_fea,
                                                 edge_length_embedded,
                                                 data_edge_index, data_batch,
                                                 data_mask_length, mask_dim1,
                                                 mask_dim2, mask_dim3)
            index = index + 1
        edge_fea = self.lin_edge(edge_fea)

        return edge_fea

    def analyze_tp(self, path):
        """
        Main network class analyze_tp process
        """
        os.makedirs(path, exist_ok=True)
        for index, (nupd, eupd) in enumerate(
                zip(self.node_update_blocks, self.edge_update_blocks)):
            fig, _ = nupd.conv.tp.visualize()
            fig.savefig(os.path.join(path, f'node_update_{index}.png'))
            fig.clf()
            fig, _ = eupd.conv.tp.visualize()
            fig.savefig(os.path.join(path, f'edge_update_{index}.png'))
            fig.clf()

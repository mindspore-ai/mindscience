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
Embedding
"""

from typing import Union, Tuple

import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import ops
from mindspore.common.initializer import Initializer, Normal


from ...utils.units import GLOBAL_UNITS, Length, get_length
from .graph import GraphEmbedding
from ..cutoff import Cutoff
from ..filter import Filter

from ...configs import Config
from ...configs import Registry as R


@R.register('embedding.molecule')
# pylint: disable=E1102
class MolEmbedding(GraphEmbedding):
    r"""Embedding for molecule

    Args:
        dim_node (int): Dimension of node embedding vector.

        dim_edge (int): Dimension of edge embedding vector.

        emb_dis (bool): Whether to embed the distance.

        emb_bond (bool): Whether to embed the bond.

        cutoff (Union[Length, float, Tensor]): Cut-off radius. Default: ``None``.

        activation: Union[Cell, str]: Activation function. Default: ``None``.

        length_unit: Union[str, Units]: Length unit. Default: Global length unit

    """

    def __init__(self,
                 dim_node: int,
                 dim_edge: int = None,
                 emb_dis: bool = True,
                 emb_bond: bool = False,
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = 'smooth',
                 rbf_fn: Cell = 'log_gaussian',
                 num_basis: int = None,
                 atom_filter: Union[Filter, str] = None,
                 dis_filter: Union[Filter, str] = None,
                 bond_filter: Union[Filter, str] = 'residual',
                 interaction: Cell = None,
                 dis_self: Length = Length(0.05, 'nm'),
                 use_sub_cutoff: bool = False,
                 cutoff_buffer: Union[Length, float, Tensor] = Length(0.2, 'nm'),
                 num_atom_types: int = 64,
                 num_bond_types: int = 16,
                 initializer: Union[Initializer, str] = Normal(1.0),
                 activation: Cell = 'silu',
                 length_unit: str = GLOBAL_UNITS.length_unit,
                 **kwargs,
                 ):

        super().__init__(
            dim_node=dim_node,
            dim_edge=dim_node if dim_edge is None else dim_edge,
            emb_dis=emb_dis,
            emb_bond=emb_bond,
            activation=activation,
            length_unit=length_unit,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.num_atom_types = int(num_atom_types)
        self.initializer = initializer
        self.atom_embedding = None
        self.atom_embedding = nn.Embedding(vocab_size=self.num_atom_types,
                                           embedding_size=self.dim_node,
                                           use_one_hot=True,
                                           embedding_table=self.initializer)

        self.cutoff = get_length(cutoff, self.units)
        self.cutoff_fn = R.build('cutoff', cutoff_fn, cutoff=self.cutoff)
        if self.cutoff_fn is not None:
            self.cutoff = self.cutoff_fn.cutoff
        if self.cutoff is not None:
            self.cutoff = ms.Tensor(self.cutoff, ms.float32)

        dis_self = get_length(dis_self, self.units)
        # The shape looks like (1)
        self.dis_self = ms.Tensor(dis_self, ms.float32).reshape((-1,))

        self.num_bond_types = int(num_bond_types)

        self.atom_filter = R.build('filter', atom_filter, dim_in=self.dim_node,
                                   dim_out=self.dim_node, activation=activation,
                                   ) if atom_filter else None

        self.use_sub_cutoff = use_sub_cutoff
        self.cutoff_buffer = get_length(cutoff_buffer, self.units)

        self.rbf_fn = None
        self.dis_filter = None
        if self.emb_dis:
            self.rbf_fn = R.build('rbf', rbf_fn,
                                  r_max=self.cutoff, num_basis=num_basis,
                                  length_unit=self.units.length_unit)

            self.dis_filter = R.build('filter', dis_filter,
                                      dim_in=self.num_basis,
                                      dim_out=self._dim_edge,
                                      activation=activation
                                      ) if dis_filter else None

        self.bond_embedding = None
        self.bond_filter = None
        if self.emb_bond:
            self.bond_embedding = nn.Embedding(
                self.num_bond_types, self._dim_edge,
                use_one_hot=True, embedding_table=initializer)
            self.bond_filter = R.build('filter', bond_filter,
                                       dim_in=self._dim_edge,
                                       dim_out=self._dim_edge,
                                       activation=activation)

        if self.emb_dis and self.emb_bond:
            self.interaction = interaction

    @property
    def num_basis(self) -> int:
        """number of radical basis function"""
        if self.rbf_fn is None:
            return 1
        return self.rbf_fn.num_basis

    @property
    def dim_edge(self) -> int:
        """dimension of edge vector"""
        if self.emb_dis and self.dis_filter is None:
            return self.num_basis
        return self._dim_edge

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of molecular model"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+f' Graph Embedding: {self.cls_name}')
        print('-'*80)
        print(ret+gap+f' Length unit: {self.units.length_unit}')
        print(ret+gap+f' Atom embedding size: {self.num_atom_types}')
        print(ret+gap+f' Cutoff distance: {self.cutoff} {self.units.length_unit}')
        print(ret+gap+f' Cutoff function: {self.cutoff_fn.cls_name}')
        print(ret+gap+f' Radical basis functions: {self.rbf_fn.cls_name}')
        self.rbf_fn.print_info(num_retraction=num_retraction +
                               num_gap, num_gap=num_gap, char=char)
        print(ret+gap+f' Embedding distance: {self.emb_dis}')
        print(ret+gap+f' Embedding Bond: {self.emb_bond}')
        print(ret+gap+f' Dimension of node embedding vector: {self.dim_node}')
        print(ret+gap+f' Dimension of edge embedding vector: {self.dim_edge}')
        print('-'*80)

    def get_rbf(self, distances: Tensor):
        """get radical basis function"""
        if self.rbf_fn is None:
            # The shape looks like (B, A, N, 1)
            return ops.expand_dims(distances, -1)
        # The shape looks like (B, A, N, F)
        return self.rbf_fn(distances)

    def construct(self,
                  atom_type: Tensor,
                  atom_mask: Tensor,
                  distance: Tensor,
                  dis_mask: Tensor,
                  bond: Tensor,
                  bond_mask: Tensor,
                  **kwargs,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # pylint: disable=unused-argument

        if self.emb_dis:
            batch_size = distance.shape[0]
            num_atoms = distance.shape[-2]
        else:
            batch_size = bond.shape[0]
            num_atoms = bond.shape[-2]

        node_emb = self.atom_embedding(atom_type)
        if self.atom_filter is not None:
            node_emb = self.atom_filter(node_emb)

        node_mask = atom_mask
        if batch_size > 1 and atom_type.shape[0] != batch_size:
            node_emb = ops.broadcast_to(node_emb, (batch_size,) + node_emb.shape[1:])
            if atom_mask is not None:
                node_mask = ops.broadcast_to(atom_mask, (batch_size,)+atom_mask.shape[1:])

        dis_emb = None
        dis_mask = None
        dis_cutoff = None
        if self.emb_dis:
            # The shape looks like (B, 1, A)(A, A)
            distance = ops.where(ops.eye(num_atoms, num_atoms, ms.bool_), self.dis_self, distance)

            # The shape looks like (B, 1, A)(B, A, A, K)
            dis_emb = self.get_rbf(distance)
            if self.dis_filter is not None:
                # The shape looks like (B, 1, A)(B, A, A, F)
                dis_emb = self.dis_filter(dis_emb)

            # The shape looks like (B, 1, A)(B, A, A)
            if self.cutoff_fn is None:
                dis_cutoff = ops.ones_like(distance)
            else:
                if self.use_sub_cutoff:
                    # The shape looks like (B, 1, A)
                    center_dis = ops.expand_dims(distance[..., 0, :], -2)
                    cutoff = self.cutoff + self.cutoff_buffer - center_dis
                    cutoff = ops.maximum(0, ops.minimum(cutoff, self.cutoff))
                    dis_cutoff, dis_mask = self.cutoff_fn(distance, dis_mask, cutoff)
                else:
                    dis_cutoff, dis_mask = self.cutoff_fn(distance, dis_mask)

        bond_emb = None
        bond_mask = None
        bond_cutoff = None
        if self.emb_bond:
            bond_emb = self.bond_embedding(bond)

            if bond_mask is not None:
                bond_emb = bond_emb * ops.expand_dims(bond_mask, -1)
                bond_cutoff = ops.cast(bond_mask > 0, bond_emb.dtype)

            if self.bond_filter is not None:
                bond_emb = self.bond_filter(bond_emb)

        edge_cutoff = dis_cutoff
        edge_mask = dis_mask
        if not self.emb_dis:
            edge_emb = bond_emb
            edge_mask = bond_mask
            edge_cutoff = bond_cutoff
        elif not self.emb_bond:
            edge_emb = dis_emb
        else:
            node_emb, edge_emb = self.interaction(node_emb, node_emb, bond_emb,
                                                  bond_mask, bond_cutoff)
        outputs = node_emb, node_mask, edge_emb, edge_mask, edge_cutoff
        return outputs

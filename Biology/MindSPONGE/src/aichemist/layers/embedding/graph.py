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

from mindspore import nn
from mindspore import Tensor
from mindspore.nn import Cell


from ...utils.units import Units, GLOBAL_UNITS

from ...configs import Registry as R


@R.register('embedding.graph')
class GraphEmbedding(nn.Cell):
    r"""Base class of graph embedding network

    Args:
        dim_node (int): Dimension of node embedding vector.

        dim_edge (int): Dimension of edge embedding vector.

        emb_dis (bool): Whether to embed the distance.

        emb_bond (bool): Whether to embed the bond.

        cutoff (Union[Length, float, Tensor]): Cut-off distance. Default: Length(1, 'nm')

        activation: Union[Cell, str]: Activation function. Default: ``None``.

        length_unit: Union[str, Units]: Length unit. Default: Global length unit

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim_node: int,
                 dim_edge: int,
                 emb_dis: bool = True,
                 emb_bond: bool = False,
                 activation: Union[Cell, str] = None,
                 length_unit: Union[str, Units] = GLOBAL_UNITS.length_unit,
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = kwargs

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.emb_dis = emb_dis
        self.emb_bond = emb_bond

        self._dim_node = int(dim_node)
        self._dim_edge = int(dim_edge)

        self.cutoff = None

        self.activation = R.build('activation', activation)

    @property
    def dim_node(self) -> int:
        r"""dimension of node embedding vectors"""
        return self._dim_node

    @property
    def dim_edge(self) -> int:
        r"""dimension of edge embedding vectors"""
        return self._dim_edge

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        """print the information of molecular model"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+f' Graph Embedding: {self.cls_name}')
        print('-'*80)
        print(ret+gap+f' Length unit: {self.units.length_unit}')
        print(ret+gap+f' Embedding distance: {self.emb_dis}')
        print(ret+gap+f' Embedding Bond: {self.emb_bond}')
        print(ret+gap+f' Dimension of node embedding vector: {self.dim_node}')
        print(ret+gap+f' Dimension of edge embedding vector: {self.dim_edge}')
        print('-'*80)

    def convert_length_from(self, unit: Union[str, Units]) -> float:
        """returns a scale factor that converts the length from a specified unit."""
        return self.units.convert_length_from(unit)

    def convert_length_to(self, unit: Union[str, Units]) -> float:
        """returns a scale factor that converts the length to a specified unit."""
        return self.units.convert_length_to(unit)

    def construct(self,
                  atom_type: Tensor,
                  atom_mask: Tensor,
                  distance: Tensor,
                  dis_mask: Tensor,
                  bond: Tensor,
                  bond_mask: Tensor,
                  **kwargs,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the properties of the molecules.

        Args:
            atom_type (Tensor): Tensor of shape (B, A). Data type is int.
                Index of atom types. Default: ``None``.
            atom_mask (Tensor): Tensor of shape (B, A). Data type is bool
                Mask for atom types
            distance (Tensor): Tensor of shape (B, A, A). Data type is float.
                Distances between central atom and its neighbouring atoms.
            dis_mask (Tensor): Tensor of shape (B, A, A). Data type is bool.
                Mask for neighbour list.
            bond (Tensor): Tensor of shape (B, A, A). Data type is int.
                Types index of bond connected with two atoms
            bond_mask (Tensor): Tensor of shape (B, A, A). Data type is bool.
                Mask for bonds

        Returns:
            node_emb (Tensor): Tensor of shape (B, A, E). Data type is float.
                Node embedding vector.
            node_mask (Tensor): Tensor of shape (B, A, E). Data type is float.
                Mask for Node embedding vector.
            edge_emb (Tensor): Tensor of shape (B, A, A, K). Data type is float.
                Edge embedding vector.
            edge_mask (Tensor): Tensor of shape (B, A, A, K). Data type is float.
                Mask for edge embedding vector.
            edge_cutoff (Tensor): Tensor of shape (B, A, A). Data type is float.
                Cutoff for edge.

        Note:
            B:  Batch size.
            A:  Number of atoms in system.
            E:  Dimension of node embedding vector
            K:  Dimension of edge embedding vector
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        raise NotImplementedError

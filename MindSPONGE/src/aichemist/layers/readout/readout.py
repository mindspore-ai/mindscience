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
Readout functions
"""

from typing import Union, Tuple

from mindspore import Tensor
from mindspore.nn import Cell

from ...configs import Registry as R


class Readout(Cell):
    r"""Readout function that merges and converts representation vectors into predicted properties.

    Args:
        dim_output (int): Dimension of outputs. Default: 1

        dim_node_rep (int): Dimension of node vectors. Default: ``None``.

        dim_edge_rep (int): Dimension of edge vectors. Default: ``None``.

        activation (Cell): Activation function, Default: ``None``.

        scale (float): Scale factor for outputs. Default: 1

        shift (float): Shift factor for outputs. Default: 0

        unit (str): Unit of output. Default: ``None``.

    Note:

        B: Batch size.

        A: Number of atoms.

        T: Number of atom types.

        Y: Output dimension.

    """

    def __init__(self,
                 dim_node_rep: int = None,
                 dim_edge_rep: int = None,
                 activation: Union[Cell, str] = None,
                 axis: int = -2,
                 ndim: int = 1,
                 shape: Tuple[int] = (1,),
                 **kwargs,
                 ):
        super().__init__()
        self._kwargs = kwargs

        self.dim_node_rep = int(dim_node_rep)
        self.dim_edge_rep = int(dim_edge_rep)

        self._ndim = ndim
        self._shape = shape
        self.axis = int(axis)
        self.shift_by_atoms = True

        self.activation = None
        if activation is not None:
            self.activation = R.build('activation', activation)

    @property
    def ndim(self) -> int:
        """rank (ndim) of output Tensor (without batch size)"""
        return self._ndim

    @property
    def shape(self) -> Tuple[int]:
        """shape of output Tensor (without batch size)"""
        return self._shape

    def set_dimension(self, dim_node_rep: int, dim_edge_rep: int):
        """check and set dimension of representation vectors"""
        if self.dim_node_rep is None:
            self.dim_node_rep = int(dim_node_rep)
        elif self.dim_node_rep != dim_node_rep:
            raise ValueError(f'The `dim_node_rep` ({self.dim_node_rep}) of Readout cannot match '
                             f'the dimension of node representation vector ({dim_node_rep}).')

        if self.dim_edge_rep is None:
            self.dim_edge_rep = int(dim_edge_rep)
        elif self.dim_edge_rep != dim_edge_rep:
            raise ValueError(f'The `dim_edge_rep` ({self.dim_edge_rep}) of Readout cannot match '
                             f'the dimension of edge representation vector ({dim_edge_rep}).')

        return self

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+f" Activation function: {self.activation}")
        print(ret+gap+f" Representation dimension: {self.dim_node_rep}")
        print(ret+gap+f" Shape of readout: {self.shape}")
        print(ret+gap+f" Rank (ndim) of readout: {self.ndim}")
        print('-'*80)
        return self

    def construct(self,
                  node_rep: Tensor,
                  edge_rep: Tensor,
                  node_emb: Tensor = None,
                  edge_emb: Tensor = None,
                  edge_cutoff: Tensor = None,
                  atom_type: Tensor = None,
                  atom_mask: Tensor = None,
                  distance: Tensor = None,
                  dis_mask: Tensor = None,
                  dis_vec: Tensor = None,
                  bond: Tensor = None,
                  bond_mask: Tensor = None,
                  **kwargs,
                  ) -> Tensor:
        r"""Compute readout function.

        Args:
            node_rep (Tensor): Tensor of shape `(B, A, F)`. Data type is float.
                Atomic (node) representation vector.
            edge_rep (Tensor): Tensor of shape `(B, A, N, G)`. Data type is float.
                Edge representation vector.
            node_emb (Tensor): Tensor of shape `(B, A, E)`. Data type is float.
                Atomic (node) embedding vector.
            edge_emb (Tensor): Tensor of shape `(B, A, N, K)`. Data type is float.
                Edge embedding vector.
            atom_type (Tensor): Tensor of shape `(B, A)`. Data type is int.
                Index of atom types. Default: ``None``.
            atom_mask (Tensor): Tensor of shape `(B, A)`. Data type is bool
                Mask for atom types
            distance (Tensor): Tensor of shape `(B, A, N)`. Data type is float.
                Distances between central atom and its neighbouring atoms.
            dis_mask (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Mask for neighbour list.
            dis_vec (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Vectors from central atom to its neighbouring atoms.
            bond_types (Tensor): Tensor of shape `(B, A, N)`. Data type is int.
                Types index of bond connected with two atoms
            bond_mask (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Mask for bonds

        Returns:
            output: (Tensor): Tensor of shape `(B, ...)`. Data type is float

        Note:
            B:  Batch size.
            A:  Number of atoms in system.
            F:  Feature dimension of node representation vector.
            G:  Feature dimension of edge representation vector.
            E:  Feature dimension of node embedding vector.
            K:  Feature dimension of edge embedding vector.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        raise NotImplementedError

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

from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.common.initializer import Initializer, Normal

from ...utils.units import GLOBAL_UNITS, Length
from ...configs import Config

from .molecule import MolEmbedding
from ..cutoff import Cutoff
from ..filter import Filter
from ...configs import Registry as R


@R.register('embedding.conformation')
class ConformationEmbedding(MolEmbedding):
    r"""Embedding for molecular conformation

    Args:
        dim_node (int): Dimension of node embedding vector.

        dim_edge (int): Dimension of edge embedding vector.

        emb_dis (bool): Whether to embed the distance.

        emb_bond (bool): Whether to embed the bond.

        cutoff (Union[Length, float, Tensor]): Cut-off distance. Default: ``None``.

        activation: Union[Cell, str]: Activation function. Default: ``None``.

        length_unit: Union[str, Units]: Length unit. Default: Global length unit

    """

    def __init__(self,
                 dim_feature: int,
                 emb_bond: bool = False,
                 cutoff: Length = Length(1, 'nm'),
                 cutoff_fn: Cutoff = None,
                 rbf_fn: Cell = None,
                 num_basis: int = None,
                 atom_filter: Union[Filter, str] = None,
                 dis_filter: Union[Filter, str] = 'residual',
                 bond_filter: Union[Filter, str] = 'residual',
                 interaction: Cell = None,
                 dis_self: Length = Length(0.05, 'nm'),
                 num_atom_types: int = 64,
                 num_bond_types: int = 16,
                 initializer: Union[Initializer, str] = Normal(1.0),
                 activation: Cell = 'swish',
                 length_unit: str = GLOBAL_UNITS.length_unit,
                 **kwargs,
                 ):

        super().__init__(
            dim_feature=dim_feature,
            emb_dis=True,
            emb_bond=emb_bond,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
            rbf_fn=rbf_fn,
            num_basis=num_basis,
            atom_filter=atom_filter,
            dis_filter=dis_filter,
            bond_filter=bond_filter,
            interaction=interaction,
            dis_self=dis_self,
            num_atom_types=num_atom_types,
            num_bond_types=num_bond_types,
            initializer=initializer,
            activation=activation,
            length_unit=length_unit,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def construct(self,
                  atom_type: Tensor,
                  atom_mask: Tensor,
                  distance: Tensor,
                  dis_mask: Tensor,
                  bond: Tensor,
                  bond_mask: Tensor,
                  **kwargs,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        return super().construct(atom_type=atom_type,
                                 atom_mask=atom_mask,
                                 distance=distance,
                                 dis_mask=dis_mask,
                                 bond=bond,
                                 bond_mask=bond_mask,
                                 **kwargs
                                 )

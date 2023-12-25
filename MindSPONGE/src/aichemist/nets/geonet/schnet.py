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
Deep molecular model
"""

from typing import Union, List

from mindspore.nn import Cell, CellList
from ...configs import Config

from .model import MolecularGNN
from ...layers.interaction import Interaction, SchNetInteraction
from ...configs import Registry as R


@R.register('net.schnet')
class SchNet(MolecularGNN):
    r"""SchNet Model.

    Reference:

        Schütt, K. T.; Sauceda, H. E.; Kindermans, P.-J.; Tkatchenko, A.; Müller, K.-R.
        Schnet - a Deep Learning Architecture for Molecules and Materials [J].
        The Journal of Chemical Physics, 2018, 148(24): 241722.

    Args:
        dim_feature (int):          Dimension of atomic representation. Default: 64

        dim_filter (int):           Dimension of filter network. Default: 64

        n_interactison (int):        Number of interaction layers. Default: 3

        activation (Cell):          Activation function. Default: 'ssp'

        cutoff (Length):            Cutoff distance. Default: Length(1, 'nm')

        cutoff_fn (Cell):           Cutoff function. Default: 'cosine'

        rbf (Cell):                 Radical baiss function. Default: 'gaussian'

        normalize_filter (bool):    Whether to normalize the filter network. Default: ``False``.

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: ``False``.

        use_graph_norm (bool):      Whether to use graph normalization. Default: ``False``.

        public_dis_filter (bool):   Whether to use public (shared) filter for distance. Default: ``False``.

        length_unit (bool):         Unit of position coordinates. Default: 'nm'

        hyper_param (dict):         Hyperparameter for molecular model. Default: ``None``.

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        K:  Number of basis functions in RBF.

        F:  Feature dimension of representation.

    """

    def __init__(self,
                 dim_feature: int = 64,
                 dim_filter: int = 64,
                 dim_edge_emb: int = None,
                 interaction: Union[Interaction, List[Interaction]] = None,
                 n_interaction: int = 3,
                 activation: Union[Cell, str] = 'ssp',
                 normalize_filter: bool = False,
                 coupled_interaction: bool = False,
                 **kwargs,
                 ):

        super().__init__(
            dim_node_rep=dim_feature,
            dim_edge_rep=dim_feature,
            interaction=interaction,
            n_interaction=n_interaction,
            activation=activation,
            coupled_interaction=coupled_interaction,
            dim_node_emb=dim_feature,
            dim_edge_emb=dim_edge_emb,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.dim_feature = int(dim_feature)
        self.dim_filter = int(dim_filter)
        self.normalize_filter = normalize_filter

        if self.interaction is None and self.dim_edge_emb is not None:
            self.build_interaction()

        self.default_embedding = self.get_default_embedding()

    def get_default_embedding(self) -> dict:
        """get default configure of embedding"""
        default_embedding = {}
        default_embedding['cls_name'] = 'molecule'
        default_embedding['dim_node'] = 64
        default_embedding['emb_dis'] = True
        default_embedding['emb_bond'] = False
        default_embedding['cutoff'] = 1
        default_embedding['cutoff_fn'] = 'cosine'
        default_embedding['rbf_fn'] = 'gaussian'
        default_embedding['dis_self'] = 0.05
        default_embedding['num_atom_types'] = 64
        default_embedding['num_bond_types'] = 16
        default_embedding['initializer'] = 'Normal'
        default_embedding['length_unit'] = 'nm'
        default_embedding['dim_node'] = self.dim_node_emb
        default_embedding['dim_edge'] = self.dim_edge_emb
        default_embedding['activation'] = self.activation
        return default_embedding

    def build_interaction(self):
        if self.dim_edge_emb is None:
            raise ValueError('Cannot build interaction without `dim_edge_emb`. '
                             'Please use `set_embedding_dimension` at first.')

        if self.coupled_interaction:
            self.interaction = CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_edge_emb=self.dim_edge_emb,
                        dim_filter=self.dim_filter,
                        activation=self.activation,
                        normalize_filter=self.normalize_filter,
                    )
                ]
                * self.n_interaction
            )
        else:
            self.interaction = CellList(
                [
                    SchNetInteraction(
                        dim_feature=self.dim_feature,
                        dim_edge_emb=self.dim_edge_emb,
                        dim_filter=self.dim_filter,
                        activation=self.activation,
                        normalize_filter=self.normalize_filter,
                    )
                    for _ in range(self.n_interaction)
                ]
            )

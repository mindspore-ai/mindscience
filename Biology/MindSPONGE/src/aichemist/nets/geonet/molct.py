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

import mindspore as ms
from mindspore.nn import Cell, CellList
from mindspore import Tensor, ops

from ...configs import Config

from .model import MolecularGNN
from ...layers.interaction import Interaction, NeuralInteractionUnit
from ...layers.filter import ResFilter
from ...configs import Registry as R


@R.register('net.MolCT')
class MolCT(MolecularGNN):
    r"""Molecular Configuration Transformer (MolCT) Model

    Reference:

        Zhang, J.; Zhou, Y.; Lei, Y.-K.; Yang, Y. I.; Gao, Y. Q.,
        Molecular CT: unifying geometry and representation learning for molecules at different scales [J/OL].
        arXiv preprint, 2020: arXiv:2012.11816 [2020-12-22]. https://arxiv.org/abs/2012.11816

    Args:
        dim_feature (int):          Dimension of atomic representation. Default: 128

        n_interaction (int):        Number of interaction layers. Default: 3

        n_heads (int):              Number of heads in multi-head attention. Default: 8

        max_cycles (int):           Maximum number of cycles of the adapative computation time (ACT).
                                    Default: 10

        activation (Cell):          Activation function. Default: 'silu'

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: ``False``.

        fixed_cycles (bool):        Whether to use the fixed cycle number to do ACT. Default: ``False``.

        use_feed_forward (bool):    Whether to use feed forward after multi-head attention. Default: ``False``.

        act_threshold (float):      Threshold of adapative computation time. Default: 0.9

    Note:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        K:  Number of basis functions in RBF.

        F:  Feature dimension of representation.

    """

    def __init__(self,
                 dim_feature: int = 128,
                 dim_edge_emb: int = None,
                 interaction: Union[Interaction, List[Interaction]] = None,
                 n_interaction: int = 3,
                 activation: Union[Cell, str] = 'silu',
                 open_act: bool = False,
                 n_heads: int = 8,
                 max_cycles: int = 10,
                 coupled_interaction: bool = False,
                 fixed_cycles: bool = False,
                 use_feed_forward: bool = False,
                 act_threshold: float = 0.9,
                 **kwargs
                 ):

        super().__init__(
            dim_node_rep=dim_feature,
            dim_edge_rep=dim_feature,
            n_interaction=n_interaction,
            interaction=interaction,
            activation=activation,
            coupled_interaction=coupled_interaction,
            dim_node_emb=dim_feature,
            dim_edge_emb=dim_edge_emb,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.open_act = open_act
        self.max_cycles = 1
        if self.open_act:
            self.max_cycles = int(max_cycles)

        self.n_heads = int(n_heads)
        self.use_feed_forward = use_feed_forward
        self.fixed_cycles = fixed_cycles
        self.act_threshold = ms.Tensor(act_threshold, ms.float32)

        self.dim_feature = int(dim_feature)

        self.filter_net = None
        if self.dim_edge_emb is not None:
            self.filter_net = ResFilter(self.dim_edge_emb, self.dim_feature, self.activation)
            if self.interaction is None:
                self.build_interaction()

        self.default_embedding = self.get_default_embedding()

    def get_default_embedding(self) -> dict:
        """get default configure of embedding"""
        default_embedding = {}
        default_embedding['cls_name'] = 'molecule'
        default_embedding['dim_node'] = 128
        default_embedding['emb_dis'] = True
        default_embedding['emb_bond'] = False
        default_embedding['cutoff'] = 1
        default_embedding['cutoff_fn'] = 'smooth'
        default_embedding['rbf_fn'] = 'log_gaussian'
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
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads,
                        max_cycles=self.max_cycles,
                        activation=self.activation,
                        fixed_cycles=self.fixed_cycles,
                        use_feed_forward=self.use_feed_forward,
                        act_threshold=self.act_threshold,
                    )
                ] * self.n_interaction
            )
        else:
            self.interaction = CellList(
                [
                    NeuralInteractionUnit(
                        dim_feature=self.dim_feature,
                        n_heads=self.n_heads,
                        max_cycles=self.max_cycles,
                        activation=self.activation,
                        fixed_cycles=self.fixed_cycles,
                        use_feed_forward=self.use_feed_forward,
                        act_threshold=self.act_threshold,
                    )
                    for _ in range(self.n_interaction)
                ]
            )

    def set_dimension(self, dim_node_emb: int, dim_edge_emb: int):
        """check and set dimension of embedding vectors"""
        super().set_dimension(dim_node_emb, dim_edge_emb)
        if self.filter_net is None:
            self.filter_net = ResFilter(self.dim_edge_emb, self.dim_feature, self.activation)
        return self

    def construct(self,
                  node_emb: Tensor,
                  node_mask: Tensor = None,
                  edge_emb: Tensor = None,
                  edge_mask: Tensor = None,
                  edge_cutoff: Tensor = None,
                  **kwargs
                  ):

        # The shape looks like (A, A)
        diagonal = ops.eye(edge_mask.shape[-1], edge_mask.shape[-1], ms.bool_)
        # The shape looks like (B, A, A)
        edge_mask |= diagonal

        node_vec = node_emb
        edge_vec = self.filter_net(edge_emb)
        for i in range(len(self.interaction)):
            node_vec, edge_vec = self.interaction[i](
                node_vec=node_vec,
                node_emb=node_emb,
                node_mask=node_mask,
                edge_vec=edge_vec,
                edge_emb=edge_emb,
                edge_mask=edge_mask,
                edge_cutoff=edge_cutoff,
            )

        return node_vec, edge_vec

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
GNN-based deep molecular model (DMM)
"""

from typing import Union, List
import mindspore as ms
from mindspore import ops
from mindspore import nn
from ...layers.interaction import Interaction
from ...configs import Registry as R


class MolecularGNN(nn.Cell):
    r"""Basic class for graph neural network (GNN) based deep molecular model

    Reference:

        Zhang, J.; Lei, Y.-K.; Zhang, Z.; Chang, J.; Li, M.; Han, X.; Yang, L.; Yang, Y. I.; Gao, Y. Q.
        A Perspective on Deep Learning for Molecular Modeling and Simulations [J].
        The Journal of Physical Chemistry A, 2020, 124(34): 6745-6763.

    Args:
        dim_feature (int):          Dimension of atomic representation. Default: 128

        n_interaction (int):        Number of interaction layers. Default: 3

        activation (Cell):          Activation function. Default: ``None``.

        coupled_interaction (bool): Whether to use coupled (shared) interaction layer. Default: ``False``.

        length_unit (bool):         Unit of position coordinates. Default: 'nm'

    Note:

        B:  Number of simulation walker

        A:  Number of atoms in system

        N:  Number of neighbour atoms

        D:  Dimension of position coordinates, usually is 3

        K:  Number of basis functions in RBF

        F:  Feature dimension of representation

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim_node_rep: int,
                 dim_edge_rep: int,
                 interaction: Union[Interaction, List[Interaction]] = None,
                 n_interaction: int = 3,
                 coupled_interaction: bool = False,
                 activation: Union[nn.Cell, str] = 'silu',
                 dim_node_emb: int = None,
                 dim_edge_emb: int = None,
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = kwargs

        self.dim_node_rep = dim_node_rep
        self.dim_edge_rep = dim_edge_rep
        self.dim_node_emb = dim_node_emb
        self.dim_edge_emb = dim_edge_emb
        self.n_interaction = n_interaction
        self.coupled_interaction = coupled_interaction

        self.activation = R.build('activation', activation)

        self.interaction: List[Interaction] = None
        if interaction is not None:
            if isinstance(interaction, Interaction):
                interaction = [interaction] * self.n_interaction
            if interaction(interaction, list):
                interaction = ms.CellList(interaction)
            if interaction(interaction, ms.CellList):
                self.n_interaction = len(interaction)
                self.interaction = interaction
            else:
                raise TypeError(f'Unsupport type: {interaction}')

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

    def set_dimension(self, dim_node_emb: int, dim_edge_emb: int):
        """check and set dimension of embedding vectors"""
        if self.dim_node_emb is None:
            self.dim_node_emb = int(dim_node_emb)
        elif self.dim_node_emb != dim_node_emb:
            raise ValueError(f'The dimension of node embedding of Embedding Cell ({dim_node_emb})'
                             f'cannot match that of Model Cell ({self.dim_node_emb}).')

        if self.dim_edge_emb is None:
            self.dim_edge_emb = int(dim_edge_emb)
        elif self.dim_edge_emb != dim_edge_emb:
            raise ValueError(f'The dimension of edge embedding of Embedding Cell ({dim_edge_emb})'
                             f'cannot match that of Model Cell ({self.dim_edge_emb}).')

        if self.interaction is None:
            self.build_interaction()

        return self

    def build_interaction(self):
        """build interaction layer"""
        return self

    def broadcast_to_interactions(self, value: ms.Tensor, name: str):
        """return the broad cast value as the interaction layers"""
        tensor = ms.Tensor(value)
        size = tensor.size
        if self.coupled_interaction:
            if size > 1:
                raise ValueError(f'The size of "{name}" must be 1 when "coupled_interaction" is "True"')
        else:
            if size not in (self.n_interaction, 1):
                raise ValueError(f'"The size of "{name}" ({size}) must be equal to '
                                 f'"n_interaction" ({self.n_interaction})!')
            tensor = ops.broadcast_to(tensor, (self.n_interaction,))
        return tensor

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = '-'):
        """print the information of molecular model"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+f' Deep molecular model: {self.cls_name}')
        print('-'*80)
        print(ret+gap+f' Dimension of node representation vector: {self.dim_node_rep}')
        print(ret+gap+f' Dimension of edge representation vector: {self.dim_edge_rep}')
        print(ret+gap+f' Dimension of node embedding vector: {self.dim_node_emb}')
        print(ret+gap+f' Dimension of edge embedding vector: {self.dim_edge_emb}')
        if self.coupled_interaction:
            print(ret+gap+f' Using coupled interaction with {self.n_interaction} layers:')
            print(ret+gap+gap+' '+self.interaction[0].cls_name)
            self.interaction[0].print_info(
                num_retraction=num_retraction+num_gap, num_gap=num_gap, char=char)
        else:
            print(ret+gap+f' Using {self.n_interaction} independent interaction layers:')
            for i, inter in enumerate(self.interaction):
                print(ret+gap+' '+str(i)+'. '+inter.cls_name)
                inter.print_info(num_retraction=num_retraction +
                                 num_gap, num_gap=num_gap, char=char)
        print('-'*80)

    def construct(self,
                  node_emb: ms.Tensor,
                  node_mask: ms.Tensor = None,
                  edge_emb: ms.Tensor = None,
                  edge_mask: ms.Tensor = None,
                  edge_cutoff: ms.Tensor = None,
                  **kwargs
                  ):
        """Compute the representation of atoms.

        Args:
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

        Returns:
            representation: (Tensor)    Tensor of shape (B, A, F). Data type is float

        Note:

            B:  Batch size.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates, usually is 3.
            F:  Feature dimension of representation.

        """
        # pylint: disable=unused-argument

        node_vec = node_emb
        edge_vec = edge_emb

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

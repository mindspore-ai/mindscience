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
GFN Layer
"""

from typing import Union
from mindspore import Tensor, ops, nn
from mindspore.nn import LayerNorm

from .readout import Readout
from ..residuals import MLP


# pylint: disable=W0613
class GFNLayer(nn.Cell):
    """_summary_

    Args:
        dim_node_rep: int = None,
        dim_edge_rep: int = None,
        node_activation: Union[nn.Cell, str] = None,
        edge_activation: Union[nn.Cell, str] = None,

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim_node_rep: int = None,
                 dim_edge_rep: int = None,
                 node_activation: Union[nn.Cell, str] = None,
                 edge_activation: Union[nn.Cell, str] = None,
                 ):
        super().__init__()
        self.edge_decoder = MLP(dim_edge_rep, [dim_edge_rep] + [1], activation=edge_activation)
        self.node_update = nn.Dense(dim_node_rep+dim_edge_rep, dim_node_rep,
                                    weight_init='xavier_uniform', activation=node_activation)
        self.edge_encoder = nn.Dense(dim_edge_rep, dim_edge_rep,
                                     weight_init='xavier_uniform', activation=edge_activation)
        self.node_layernorm = LayerNorm([dim_node_rep])
        self.edge_layernorm = LayerNorm([dim_edge_rep])
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self,
                  node_rep,
                  edge_rep,
                  edge_msg,
                  edge_dec,
                  atom_mask,
                  neighbour_mask,
                  **kwargs):
        """_summary_

        Args:
            node_rep (Tensor): node representation
            edge_rep (Tensor): edge representaton
            edge_msg (Tensor): edge messages
            edge_dec (Tensor): value of edge decoder
            atom_mask (Tensor): atom mask
            neighbour_mask (Tensor): neighbour mask

        Returns:
            node_rep (Tensor): calculated node representation
            edge_rep (Tensor): calculated edge representaton
            edge_msg (Tensor): calculated edge messages
            edge_dec (Tensor): calculated value of edge decoder
        """
        # update atom
        node_msg = self.reduce_sum(edge_msg, -2)
        node_edge = ops.concat((node_rep, node_msg), -1)
        node_layernorm = self.node_layernorm(self.node_update(node_edge) * ops.expand_dims(atom_mask, -1))
        node_rep += node_layernorm

        # update edge
        edge_layernorm = self.edge_layernorm(ops.expand_dims(node_rep, -2) + ops.expand_dims(node_rep, 1))
        edge_rep += edge_layernorm

        # update force
        edge_msg = self.edge_encoder(edge_rep) * ops.expand_dims(neighbour_mask, -1)
        edge_dec += self.edge_decoder(edge_msg) * ops.expand_dims(neighbour_mask, -1)
        outputs = node_rep, edge_rep, edge_msg, edge_dec
        return outputs


class GFNReadout(Readout):
    """
    Readout for Graph Field Network

    Args:
        dim_node_rep (int, optional):                    dimensionality of node representations. Defaults to None.
        dim_edge_rep (int, optional):                    dimensionality of edge representations. Defaults to None.
        node_activation (Union[nn.Cell, str], optional): node activation. Defaults to None.
        edge_activation (Union[nn.Cell, str], optional): edge activation. Defaults to None.
        iterations (int, optional):                      number of interaction layers. Defaults to None.
        shared_parms (bool, optional):                   If True, all of the gfn layers will have the same parameters.
                                                         Defaults to True.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dim_node_rep: int = None,
                 dim_edge_rep: int = None,
                 node_activation: Union[nn.Cell, str] = None,
                 edge_activation: Union[nn.Cell, str] = None,
                 iterations: int = None,
                 shared_parms=True,
                 **kwargs):
        super().__init__()

        self._ndim = 2
        self._shape = (1,)

        self.iterations = iterations
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.shared_parms = shared_parms

        if shared_parms:
            self.read_out = nn.CellList(
                [GFNLayer(dim_node_rep, dim_edge_rep, node_activation, edge_activation)] * iterations)
        else:
            self.read_out = nn.CellList(
                [GFNLayer(dim_node_rep, dim_edge_rep, node_activation, edge_activation) for i in range(iterations)])

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+f" Activation function: {self.activation}")
        print(ret+gap+f" Representation dimension: {self.dim_node_rep}")
        print(ret+gap+f" Readout iterations: {self.iterations}")
        print(ret+gap+f" Shape of readout: {self.shape}")
        print(ret+gap+f" Rank (ndim) of readout: {self.ndim}")
        print(ret+gap+f" Whether used shared parameters: {self.shared_parms}")
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
                  **kwargs):

        edge_msg = edge_emb
        edge_dec = 0

        for i in range(self.iterations):
            node_rep, edge_rep, edge_msg, edge_dec = self.read_out[i](
                node_rep, edge_rep, edge_msg, edge_dec, atom_mask, dis_mask)

        dxi = self.reduce_sum(ops.expand_dims(edge_cutoff, -1) * dis_vec * edge_dec, 2)

        return dxi[:, 0].expand_dims(1)

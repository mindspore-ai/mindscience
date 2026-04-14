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
Interaction layers
"""

from typing import Union, Tuple

from mindspore import Tensor
from mindspore import nn
from mindspore.ops import functional as F

from ...configs import Config

from .base import Interaction
from ..residuals import MLP
from ..aggregator import Aggregate
from ..filter import DenseFilter
from ...configs import Registry as R


@R.register('interaction.schnet')
class SchNetInteraction(Interaction):
    r"""Interaction layer of SchNet.

    Args:
        dim_feature (int):          Feature dimension.

        dim_filter (int):           Dimension of filter network.

        filter_net (nn.Cell):       Filter network for distance

        activation (nn.Cell):       Activation function. Default: 'ssp'

        normalize_filter (bool):    Whether to nomalize filter network. Default: ``False``.

    """

    def __init__(self,
                 dim_feature: int,
                 dim_edge_emb: int,
                 dim_filter: int,
                 activation: Union[nn.Cell, str] = 'ssp',
                 normalize_filter: bool = False,
                 **kwargs,
                 ):

        super().__init__(
            dim_node_rep=dim_feature,
            dim_edge_rep=dim_feature,
            dim_node_emb=dim_feature,
            dim_edge_emb=dim_edge_emb,
            activation=activation,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.dim_filter = int(dim_filter)
        if dim_filter is None:
            self.dim_filter = self.dim_edge_rep

        # The shape looks like (..., K) -> (..., W)
        self.filter_net = DenseFilter(dim_in=self.dim_edge_emb, dim_out=self.dim_filter,
                                      activation=activation)

        # The shape looks like (..., F) -> (..., W)
        self.atomwise_bc = nn.Dense(self.dim_node_emb, self.dim_filter, weight_init='xavier_uniform')
        # The shape looks like (..., W) -> (..., F)
        self.atomwise_ac = MLP(self.dim_filter, [self.dim_node_rep], activation=activation)

        self.agg = Aggregate(axis=-2, mean=normalize_filter)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + str(self.dim_node_rep))
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        return self

    def construct(self,
                  node_vec: Tensor,
                  node_emb: Tensor,
                  node_mask: Tensor,
                  edge_vec: Tensor,
                  edge_emb: Tensor,
                  edge_mask: Tensor = None,
                  edge_cutoff: Tensor = None,
                  **kwargs
                  ) -> Tuple[Tensor, Tensor]:

        # The shape looks like (B, A, W) <- (B, A, F)
        x_i = self.atomwise_bc(node_vec)

        # The shape looks like (B, A, A, W) <- (B, A, A, K)
        g_ij = self.filter_net(edge_vec)
        # The shape looks like (B, A, A, W) * (B, A, A, 1)
        w_ij = g_ij * F.expand_dims(edge_cutoff, -1)

        # The shape looks like (B, A, 1, W) * (B, A, A, W)
        y = F.expand_dims(x_i, -2) * w_ij

        # The shape looks like (B, A, W) <- (B, A, A, W)
        y = self.agg(y, edge_mask)
        # The shape looks like (B, A, F) <- (B, A, W)
        v = self.atomwise_ac(y)

        # The shape looks like (B, A, F) + (B, A, F)
        node_new = node_vec + v

        return node_new, edge_vec

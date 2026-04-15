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
from mindspore import Parameter
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from ...configs import Config

from .base import Interaction
from ..residuals import PreActDense
from ..residuals import SeqPreActResidual
from ...configs import Registry as R


@R.register('interaction.physnet')
class PhysNetModule(Interaction):
    r"""PhysNet Module (Interaction layer)

    Args:
        dim_feature (int):          Feature dimension.

        activation (nn.Cell):       Activation function. Default: 'ssp'

        n_inter_residual (int):     Number of inter residual blocks. Default: 3

        n_outer_residual (int):     Number of outer residual blocks. Default: 2

    """

    def __init__(self,
                 dim_feature: int,
                 dim_edge_emb: int = None,
                 n_inter_residual: int = 3,
                 n_outer_residual: int = 2,
                 activation: Union[nn.Cell, str] = 'ssp',
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

        dim_feature = self.dim_node_rep

        self.xi_dense = nn.Dense(dim_feature, dim_feature, weight_init='xavier_uniform', activation=self.activation)
        self.xij_dense = nn.Dense(dim_feature, dim_feature, weight_init='xavier_uniform', activation=self.activation)

        self.attention_mask = nn.Dense(self.dim_edge_emb, self.dim_edge_rep,
                                       weight_init='xavier_uniform',
                                       has_bias=False, activation=None)

        self.gating_vector = Parameter(initializer(Normal(1.0), [dim_feature]),
                                       name="gating_vector")

        self.n_inter_residual = int(n_inter_residual)
        self.n_outer_residual = int(n_outer_residual)

        self.inter_residual = SeqPreActResidual(dim_feature, activation=self.activation,
                                                n_res=self.n_inter_residual)
        self.inter_dense = PreActDense(dim_feature, dim_feature, activation=self.activation)
        self.outer_residual = SeqPreActResidual(dim_feature, activation=self.activation,
                                                n_res=self.n_outer_residual)

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Feature dimension: ' + self.dim_node_rep)
        print(ret+gap+' Activation function: ' + self.activation.cls_name)
        print(ret+gap+' Number of layers at inter residual: ' +
              self.n_inter_residual)
        print(ret+gap+' Number of layers at outer residual: ' +
              self.n_outer_residual)
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

        # The shape looks like (B, A, F)
        xi = self.activation(node_vec)
        ux = self.gating_vector * node_vec
        dxi = self.xi_dense(xi)

        # The shape looks like (B, A, 1, F)
        dxij = self.xij_dense(F.expand_dims(xi, -2))

        # The shape looks like (B, A, A, F) * (B, A, A, 1)
        attention_mask = self.attention_mask(edge_vec * F.expand_dims(edge_cutoff, -1))

        # The shape looks like (B, A, A, F) * (B, A, 1, F)
        side = attention_mask * dxij
        if edge_mask is not None:

            side = side * F.expand_dims(edge_mask, -1)
        v = dxi + F.reduce_sum(side, -2)

        v1 = self.inter_residual(v)
        v1 = self.inter_dense(v1)
        y = ux + v1

        node_new = self.outer_residual(y)

        return node_new, edge_vec

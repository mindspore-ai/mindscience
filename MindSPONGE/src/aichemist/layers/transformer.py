# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
Attention layers
"""

from mindspore import nn
from mindspore import ops
from .. import core


class Attention(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.mlp_q = nn.SequentialCell([
            nn.Dense(embed_dim, embed_dim, has_bias=False),
            self.activation,
        ])
        self.mlp_k = nn.SequentialCell([
            nn.Dense(embed_dim, embed_dim, has_bias=False),
            self.activation,
        ])
        self.mlp_v = nn.SequentialCell([
            nn.Dense(embed_dim, embed_dim, has_bias=False),
        ])

    def construct(self, queries, keys, values, mask, cross_msgs=None):
        """Compute cross attention.
        x_i attend to y_j:
        a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
        attention_x = sum_j a_{i->j} y_j
        Args:
        queries: NxD float tensor --> queries
        keys: MxD float tensor --> keys
        values: Mxd
        mask: NxM
        Returns:
        attention_x: Nxd float tensor.
        """
        queries = self.mlp_q(queries)
        keys = self.mlp_k(keys)
        values = self.mlp_v(values)
        if not cross_msgs:
            return queries * 0.
        a = mask * ops.dot(queries, keys.transpose(1, 0)) - 1000. * (1. - mask)
        a_x = ops.softmax(a, axis=1)  # i->j, NxM, a_x.sum(axis=1) = torch.ones(N)
        attention_x = ops.dot(a_x, values)  # (N,d)
        return attention_x

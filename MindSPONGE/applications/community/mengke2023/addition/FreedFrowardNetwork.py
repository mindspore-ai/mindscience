# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
FeedForwardNetwork代码迁移： pytorch -> mindspore
"""
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

import numpy as np

class FeedForwardNetwork(nn.Cell):
    """FeedForwardNetwork"""
    def __init__(self, embedding_dim: int, ffn_embedding_dim: int,
                 activation_dropout: float = 0.1,
                 max_tokens_per_msa: int = 2**14):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(p=activation_dropout)
        self.fc1 = nn.Dense(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Dense(ffn_embedding_dim, embedding_dim)

    def construct(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    feedforward = FeedForwardNetwork(2, 4)
    feedforward.fc1.weight = Parameter(Tensor([[-0.2400, -0.0018],
                                               [0.4799, 0.4315],
                                               [-0.1139, -0.2861],
                                               [-0.4738, 0.3101]]))
    feedforward.fc1.bias = Parameter(Tensor([-0.2531, -0.5497, -0.6989, -0.3321]))
    feedforward.fc2.weight = Parameter(Tensor([[-0.1086, 0.1633, 0.2901, 0.1494],
                                               [0.0249, 0.3146, -0.1992, -0.1573]]))
    feedforward.fc2.bias = Parameter(Tensor([0.4258, 0.2745]))

    X = np.arange(24).reshape(12, 2)
    X = Tensor(X).astype(mstype.float32)
    res = feedforward(X)
    print(res)

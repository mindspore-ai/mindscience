# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Roberta LM Head"""
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
from mindspore.nn import LayerNorm
# pylint: disable=relative-beyond-top-level
from .transformer_layer import gelu


class RobertaLMHead(nn.Cell):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Dense(embed_dim, embed_dim)
        self.layer_norm = LayerNorm((embed_dim,), begin_norm_axis=-1, begin_params_axis=-1)
        self.linear = nn.Dense(embed_dim, output_dim)
        self.linear.weight.set_data(weight)
        self.linear.bias.set_data(ops.Zeros()(output_dim, ms.float32))

    def construct(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.linear(x)
        return x

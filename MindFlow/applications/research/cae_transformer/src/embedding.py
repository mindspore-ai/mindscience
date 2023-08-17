# ============================================================================
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
"""embedding"""
import math
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import initializer, HeNormal


class PositionalEmbedding(nn.Cell):
    """Positional embedding module"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = ops.stop_gradient(ops.zeros((max_len, d_model), ms.float32))
        cast = ops.Cast()
        expand = ops.ExpandDims()
        position = expand(cast(ms.numpy.arange(0, max_len), ms.float32), 1)
        div_term = ops.exp((cast(ms.numpy.arange(0, d_model, 2), ms.float32) * -(math.log(10000.0) / d_model)))

        self.pe[:, 0::2] = ops.Sin()(position * div_term)
        self.pe[:, 1::2] = ops.Cos()(position * div_term)
        self.pe = expand(self.pe, 0)

    def construct(self, x):
        return self.pe[:, :x.shape[1]]


class TokenEmbedding(nn.Cell):
    """Token embedding module"""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                              kernel_size=3, padding=padding, pad_mode='pad')
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv1d):
                m.weight.set_data(initializer(HeNormal(mode='fan_in', nonlinearity='leaky_relu'),
                                              m.weight.shape, ms.float32))

    def construct(self, x):
        x = self.conv(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        return x


class DataEmbedding(nn.Cell):
    """Data embedding module"""
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.embed_type = embed_type
        self.freq = freq
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
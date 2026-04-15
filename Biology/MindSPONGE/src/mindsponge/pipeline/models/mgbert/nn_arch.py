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
"""Model of MG-BERT"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P


class CustomWithLossCell(nn.Cell):
    """Training"""

    def __init__(self, backbone, loss_fun):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fun

    def construct(self, t_x, t_adjoin_matrix, t_y, t_char_weight):
        """construct"""
        output = self._backbone([t_x, t_adjoin_matrix])
        t_y = ops.reshape(t_y, (t_y.shape[0] * t_y.shape[1],))
        t_char_weight = ops.reshape(t_char_weight, (t_char_weight.shape[0] * t_char_weight.shape[1],))
        return self._loss_fn(output, t_y, t_char_weight)


def dense(in_channel, out_channel, use_se=True, activation=None):
    """Custom dense"""
    if not use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = ms.Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=ms.float32)
    else:
        boundary = np.sqrt(6 / (out_channel + in_channel))
        weight_shape = (out_channel, in_channel)
        weight = ms.Tensor(np.random.uniform(-boundary, boundary, weight_shape), dtype=ms.float32)

    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0, activation=activation)


def embedding(in_channel, out_channel, use_se=1):
    """embedding"""
    if use_se == 0:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = ms.Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=ms.float32)
    elif use_se == 1:
        weight_shape = (in_channel, out_channel)
        weight = ms.Tensor(np.random.uniform(-0.07, 0.07, weight_shape), dtype=ms.float32)
    else:
        boundary = np.sqrt(6 / (out_channel + in_channel))
        weight_shape = (in_channel, out_channel)
        weight = ms.Tensor(np.random.uniform(-boundary, boundary, weight_shape), dtype=ms.float32)

    return nn.Embedding(in_channel, out_channel, embedding_table=weight)


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    q = ops.Cast()(q, ms.float16)
    k = ops.Cast()(k, ms.float16)
    matmul_qk = P.BatchMatMul(transpose_b=True)(q, k)
    dk = ops.Cast()(k.shape[-1], ms.float16)

    scaled_attention_logits = matmul_qk / ops.sqrt(dk)
    scaled_attention_logits = ops.Cast()(scaled_attention_logits, ms.float32)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    attention_weights = ms.nn.Softmax()(ops.Cast()(scaled_attention_logits, ms.float32))
    attention_weights = ops.Cast()(attention_weights, ms.float16)
    v = ops.Cast()(v, ms.float16)
    output = P.BatchMatMul()(attention_weights, v)
    output = ops.Cast()(output, ms.float32)
    attention_weights = ops.Cast()(attention_weights, ms.float32)

    return output, attention_weights


class MultiHeadAttention(nn.Cell):
    """MultiHead Attention"""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        if d_model % self.num_heads != 0:
            raise ValueError()

        self.depth = d_model // self.num_heads
        wq = dense(d_model, d_model, use_se=False)
        wk = dense(d_model, d_model)
        wv = dense(d_model, d_model)
        fc = dense(d_model, d_model)
        self.cell_list = nn.CellList()
        self.cell_list.append(wq)
        self.cell_list.append(wk)
        self.cell_list.append(wv)
        self.cell_list.append(fc)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = ops.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return ops.transpose(x, (0, 2, 1, 3))

    def construct(self, v, k, q, mask, adjoin_matrix):
        """MultiHead Attention construction"""
        batch_size = q.shape[0]

        q = self.cell_list[0](q)  # (batch_size, seq_len, d_model)
        k = self.cell_list[1](k)  # (batch_size, seq_len, d_model)
        v = self.cell_list[2](v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix)

        scaled_attention = ops.transpose(scaled_attention,
                                         (0, 2, 1, 3))  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = ops.reshape(scaled_attention,
                                       (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.cell_list[3](concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(nn.Cell):
    """Encoder layer"""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        layernorm1 = ms.nn.LayerNorm([d_model], epsilon=1e-6)
        layernorm2 = ms.nn.LayerNorm([d_model], epsilon=1e-6)
        dropout1 = ms.nn.Dropout(1 - rate)
        dropout2 = ms.nn.Dropout(1 - rate)
        dense1 = dense(d_model, dff, activation='gelu')
        dense2 = dense(dff, d_model)

        self.cell_list = nn.CellList()
        self.cell_list.append(layernorm1)
        self.cell_list.append(layernorm2)
        self.cell_list.append(dropout1)
        self.cell_list.append(dropout2)
        self.cell_list.append(dense1)
        self.cell_list.append(dense2)

    def construct(self, x, mask, adjoin_matrix):
        """construct"""
        attn_output, attention_weights = self.mha(x, x, x, mask, adjoin_matrix)
        attn_output = self.cell_list[2](attn_output)
        out1 = self.cell_list[0](x + attn_output)
        ffn_output = self.cell_list[4](out1)
        ffn_output = self.cell_list[5](ffn_output)
        ffn_output = self.cell_list[3](ffn_output)
        out2 = self.cell_list[1](out1 + ffn_output)

        return out2, attention_weights


class Encoder(nn.Cell):
    """Encoder"""

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = embedding(input_vocab_size, d_model, use_se=1)
        self.dropout = ms.nn.Dropout(1 - rate)
        self.cell_list = nn.CellList()
        for _ in range(self.num_layers):
            self.cell_list.append(EncoderLayer(d_model, num_heads, dff, rate))

    def construct(self, x, mask, adjoin_matrix):
        """Encoder construction"""
        adjoin_matrix = ops.ExpandDims()(adjoin_matrix, 1)
        x = self.embedding(x)

        x_emd = ops.sqrt(ops.Cast()(self.d_model, ms.float32))
        x_1 = x * x_emd
        x = self.dropout(x_1)

        for i in range(self.num_layers):
            x, _ = self.cell_list[i](x, mask, adjoin_matrix)
        return x


class BertModel(nn.Cell):
    """Bert Model"""

    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=17, dropout_rate=0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff, input_vocab_size=vocab_size, rate=dropout_rate)
        fc1 = dense(d_model, d_model, activation='gelu')
        self.layernorm3 = ms.nn.LayerNorm([d_model], epsilon=0.001)
        fc2 = dense(d_model, vocab_size)
        self.cell_list = nn.CellList()
        self.cell_list.append(fc1)
        self.cell_list.append(fc2)

    def construct(self, inputs):
        """construct"""
        x = inputs[0]
        adjoin_matrix = inputs[1]
        seq = ops.Cast()(ops.equal(x, 0), ms.float32)
        mask = ops.ExpandDims()(seq, 1)
        mask = ops.ExpandDims()(mask, 2)
        x = self.encoder(x, mask=mask, adjoin_matrix=adjoin_matrix)
        x = self.cell_list[0](x)
        x = self.layernorm3(x)
        x = self.cell_list[0](x)
        x = ops.reshape(x, (-1, x.shape[2]))
        return x


class PredictModel(nn.Cell):
    """Predict Model"""

    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=17, dropout_rate=0.1,
                 dense_dropout=0.1):
        super(PredictModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff, input_vocab_size=vocab_size, rate=dropout_rate)

        fc1 = dense(d_model, d_model, activation=ms.nn.LeakyReLU(0.1))
        self.dropout = ms.nn.Dropout(1 - dense_dropout)
        fc2 = dense(d_model, 1)
        self.cell_list = nn.CellList()
        self.cell_list.append(fc1)
        self.cell_list.append(fc2)

    def construct(self, inputs):
        """construct"""
        x = inputs[0]
        adjoin_matrix = inputs[1]
        seq = ops.Cast()(ops.equal(x, 0), ms.float32)
        mask = ops.ExpandDims()(seq, 1)
        mask = ops.ExpandDims()(mask, 2)
        x = self.encoder(x, mask=mask, adjoin_matrix=adjoin_matrix)
        x = x[:, 0, :]
        x = self.cell_list[0](x)
        x = self.dropout(x)
        x = self.cell_list[1](x)
        return x


class ClassificationModel(nn.Cell):
    """Classification Model"""

    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=17, dropout_rate=0.1,
                 dense_dropout=0.1):
        super(ClassificationModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff, input_vocab_size=vocab_size, rate=dropout_rate)

        fc1 = dense(d_model, d_model, activation=ms.nn.LeakyReLU(0.1))
        self.dropout = ms.nn.Dropout(1 - dense_dropout)
        fc2 = dense(d_model, 1)
        self.cell_list = nn.CellList()
        self.cell_list.append(fc1)
        self.cell_list.append(fc2)

    def construct(self, inputs):
        """construct"""
        x = inputs[0]
        adjoin_matrix = inputs[1]
        seq = ops.Cast()(ops.equal(x, 0), ms.float32)
        mask = ops.ExpandDims()(seq, 1)
        mask = ops.ExpandDims()(mask, 2)
        x = self.encoder(x, mask=mask, adjoin_matrix=adjoin_matrix)
        x = x[:, 0, :]
        x = self.cell_list[0](x)
        x = self.dropout(x)
        x = self.cell_list[1](x)
        x = ops.reshape(x, (x.shape[0] * x.shape[1],))
        return x


class MGBertModel(nn.Cell):
    """mgbert"""

    def __init__(self, config):
        super(MGBertModel, self).__init__()
        if config.model_task == 'small':
            model_config = config.small
        elif config.model_task == 'medium':
            model_config = config.medium
        elif config.model_task == 'large':
            model_config = config.large
        if config.task_name == 'classification':
            self.model = ClassificationModel(num_layers=model_config.num_layers, d_model=model_config.d_model,
                                             dff=model_config.d_model * 2,
                                             num_heads=model_config.num_heads,
                                             vocab_size=config.vocab_size, dense_dropout=0.15)
        elif config.task_name == 'regression':
            model_config = config.small
            self.model = PredictModel(num_layers=model_config.num_layers, d_model=model_config.d_model,
                                      dff=model_config.d_model * 2,
                                      num_heads=model_config.num_heads, vocab_size=config.vocab_size,
                                      dense_dropout=0.15)
        else:
            model_config = config.small
            self.model = BertModel(num_layers=model_config.num_layers, d_model=model_config.d_model,
                                   dff=model_config.d_model * 2,
                                   num_heads=model_config.num_heads,
                                   vocab_size=config.vocab_size)

    def construct(self, inputs):
        """construct"""
        x = self.model(inputs)
        return x

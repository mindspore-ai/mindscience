# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""basic"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .initializer import glorot_uniform


class Attention(nn.Cell):
    r"""
    This is an implementation of multihead attention in the paper `Attention is all you need
    <https://arxiv.org/pdf/1706.03762v5.pdf>`_. Given the query vector with source length,
    and the key with key length and the target length, the attention will be performed as
    the following.

    .. math::

        Attention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

    if query, key and value tensor is same, then it will be modified version of self
    attention.

    Args:
        num_head(int):     The number of the heads.
        hidden_size(int):   The hidden size of the input.
        gating(bool):       Indicator of if the attention is gated.
        q_data_dim(int):    The last dimension length of the query tensor.
        m_data_dim(int):    The last dimension length of the key and value tensor.
        output_dim(int):    The last dimension length of the output tensor.
        batch_size(int):    The batch size of parameters in attention, used in while
                            control flow. Default: None.

    Inputs:
        - **q_data** (Tensor) - The query tensor with shape (batch_size,
          query_seq_length, q_data_dim) with query_seq_length the query sequence length.
        - **m_data** (Tensor) - The key/value tensor with shape (batch_size,
          value_seq_length, m_data_dim) with value_seq_length the value sequence length.
        - **attention_mask** (Tensor) - The mask for attention matrix with shape
          (batch_size, num_head, query_seq_length, value_seq_length).
        - **index** (Tensor) - The index of while loop, only used in case of while
          control flow. Default: None.
        - **nonbatched_bias** (Tensor) - Non-batched bias for the attention matrix with
          shape(num_heads, query_seq_length, value_seq_length). Default: None.

    Outputs:
        Tensor, output tensor of the Attention layer with shape (batch_size,
          query_seq_length, hidden_size).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import Attention
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = Attention(num_head=4, hidden_size=64, gating=True, q_data_dim=64,
        ...                   m_data_dim=64, output_dim=64)
        >>> q_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
        >>> m_data = Tensor(np.ones((32, 256, 64)), mstype.float32)
        >>> attention_mask = Tensor(np.ones((32, 4, 128, 256)), mstype.float32)
        >>> attn_out= model(q_data, m_data, attention_mask)
        >>> print(attn_out.shape)
        (32, 128, 64)
    """

    def __init__(self, num_head, hidden_size, gating, q_data_dim, m_data_dim, output_dim,
                 batch_size=None):
        super(Attention, self).__init__()
        self.q_data_dim = q_data_dim
        self.m_data_dim = m_data_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.gating = gating
        self.hidden_size = hidden_size
        self.dim_per_head = self.hidden_size // self.num_head
        self.batch_size = batch_size
        self.matmul = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.batch_size = batch_size
        self._init_parameter()

    def construct(self, q_data, m_data, attention_mask, index=None, nonbatched_bias=None):
        '''construct'''
        if self.batch_size:
            linear_q_weight = P.Gather()(self.linear_q_weights, index, 0)
            linear_k_weight = P.Gather()(self.linear_k_weights, index, 0)
            linear_v_weight = P.Gather()(self.linear_v_weights, index, 0)
            linear_output_weight = P.Gather()(self.linear_output_weights, index, 0)
            o_bias = P.Gather()(self.o_biases, index, 0)
            linear_gating_weight = 0
            gating_bias = 0
            if self.gating:
                linear_gating_weight = P.Gather()(self.linear_gating_weights, index, 0)
                gating_bias = P.Gather()(self.gating_biases, index, 0)
        else:
            linear_q_weight = self.linear_q_weights
            linear_k_weight = self.linear_k_weights
            linear_v_weight = self.linear_v_weights
            linear_output_weight = self.linear_output_weights
            o_bias = self.o_biases
            linear_gating_weight = 0
            gating_bias = 0
            if self.gating:
                linear_gating_weight = self.linear_gating_weights
                gating_bias = self.gating_biases

        dim_b, dim_q, dim_a = q_data.shape
        _, dim_k, dim_c = m_data.shape
        dim_h = self.num_head

        q_data = P.Reshape()(q_data, (-1, dim_a))
        m_data = P.Reshape()(m_data, (-1, dim_c))

        q = self.matmul(q_data, linear_q_weight) * self.dim_per_head ** (-0.5)
        k = self.matmul(m_data, linear_k_weight)
        v = self.matmul(m_data, linear_v_weight)

        q = P.Reshape()(q, (dim_b, dim_q, dim_h, -1))
        k = P.Reshape()(k, (dim_b, dim_k, dim_h, -1))
        v = P.Reshape()(v, (dim_b, dim_k, dim_h, -1))

        tmp_q = P.Transpose()(q, (0, 2, 1, 3))
        tmp_k = P.Transpose()(k, (0, 2, 1, 3))
        logits = P.Add()(self.batch_matmul_trans_b(tmp_q, tmp_k), attention_mask)

        if nonbatched_bias is not None:
            bias = P.ExpandDims()(nonbatched_bias, 0)
            logits = P.Add()(logits, bias)
        weights = self.softmax(logits)
        tmp_v = P.Transpose()(v, (0, 2, 3, 1))

        weighted_avg = P.Transpose()(self.batch_matmul_trans_b(weights, tmp_v), (0, 2, 1, 3))

        if self.gating:
            gating_bias = P.ExpandDims()(P.ExpandDims()(gating_bias, 0), 0)
            gate_values = P.Add()(P.Reshape()(self.matmul(q_data, linear_gating_weight),
                                              (dim_b, dim_q, dim_h, -1)),
                                  gating_bias)
            gate_values = self.sigmoid(gate_values)
            weighted_avg = P.Reshape()(weighted_avg * gate_values, (dim_b * dim_q, -1))

        weighted_avg = P.Reshape()(weighted_avg, (dim_b * dim_q, -1))
        output = P.Add()(P.Reshape()(self.matmul(weighted_avg, linear_output_weight),
                                     (dim_b, dim_q, -1)),
                         P.ExpandDims()(o_bias, 0))
        return output

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.linear_q_weights = Parameter(Tensor(np.zeros([self.batch_size,
                                                               self.num_head * self.dim_per_head,
                                                               self.q_data_dim]), mstype.float32))
            self.linear_k_weights = Parameter(Tensor(np.zeros([self.batch_size,
                                                               self.num_head * self.dim_per_head,
                                                               self.m_data_dim]), mstype.float32))
            self.linear_v_weights = Parameter(Tensor(np.zeros([self.batch_size,
                                                               self.num_head * self.dim_per_head,
                                                               self.m_data_dim]), mstype.float32))
            self.linear_output_weights = Parameter(Tensor(np.zeros([self.batch_size,
                                                                    self.output_dim,
                                                                    self.num_head * \
                                                                        self.dim_per_head]),
                                                          mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros([self.batch_size, self.output_dim]),
                                             mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(Tensor(np.zeros([self.batch_size,
                                                                        self.num_head * \
                                                                            self.dim_per_head,
                                                                        self.q_data_dim]),
                                                              mstype.float32))
                self.gating_biases = Parameter(Tensor(np.zeros((self.batch_size,
                                                                self.num_head,
                                                                self.dim_per_head)),
                                                      mstype.float32), name="gating_b")
        else:
            self.linear_q_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.q_data_dim, self.dim_per_head * self.q_data_dim,
                               [self.num_head * self.dim_per_head, self.q_data_dim]),
                mstype.float32))
            self.linear_k_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.m_data_dim, self.dim_per_head * self.m_data_dim,
                               [self.num_head * self.dim_per_head, self.m_data_dim]),
                mstype.float32))
            self.linear_v_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.m_data_dim, self.dim_per_head * self.m_data_dim,
                               [self.num_head * self.dim_per_head, self.m_data_dim]),
                mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros([self.output_dim, self.num_head * self.dim_per_head]),
                       mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros([self.output_dim]), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros([self.num_head * self.dim_per_head, self.q_data_dim]),
                           mstype.float32))
                self.gating_biases = Parameter(Tensor(np.ones((self.num_head, self.dim_per_head)),
                                                      mstype.float32),
                                               name="gating_b")


class GlobalAttention(nn.Cell):
    r"""
    This is an implementation of global gated self attention in the paper `Highly accurate
    protein structure prediction with AlphaFold
    <https://www.nature.com/articles/s41586-021-03819-2.pdf>`_. For this attention, the
    shape of the query tensor, key tensor and the value tensor should be the same.

    Args:
        num_head(int):     The number of the heads.
        gating(bool):       Indicator of if the attention is gated.
        input_dim(int):     The last dimension length of the input tensor.
        output_dim(int):    The last dimension length of the output tensor.
        batch_size(int):    The batch size of parameters in attention, used in while control
                            flow. Default: None.

    Inputs:
        - **q_data** (Tensor) - The query tensor with shape (batch_size, seq_length,
          input_dim) with seq_length the sequence length.
        - **m_data** (Tensor) - The key/value tensor with shape (batch_size, seq_length,
          input_dim).
        - **q_mask** (Tensor) - A binary mask for q_data of shape (batch_size,
          seq_length, 1).
        - **bias** (Tensor) - Bias for the attention matrix. Default: None.
        - **index** (Tensor) - The index of while loop, only used in case of while control
          flow. Default: None.

    Outputs:
        Tensor, Output tensor of the GlobalAttention layer with shape (batch_size, seq_length, output_dim).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import GlobalAttention
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = GlobalAttention(num_head=4, input_dim=64, gating=True, output_dim=256)
        >>> q_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
        >>> m_data = Tensor(np.ones((32, 128, 64)), mstype.float32)
        >>> q_mask = Tensor(np.ones((32, 128, 1)), mstype.float32)
        >>> attn_out= model(q_data, m_data, q_mask)
        >>> print(attn_out.shape)
        (32, 128, 256)
    """

    def __init__(self, num_head, gating, input_dim, output_dim, batch_size=None):
        super(GlobalAttention, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.dim_per_head = self.input_dim // self.num_head
        self.output_dim = output_dim
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.gating = gating
        self.batch_size = batch_size
        self._init_parameter()

    def construct(self, q_data, m_data, q_mask, index=None):
        '''construct'''
        if self.batch_size:
            q_weights = P.Gather()(self.linear_q_weights, index, 0)
            k_weights = P.Gather()(self.linear_k_weights, index, 0)
            v_weights = P.Gather()(self.linear_v_weights, index, 0)
            output_weights = P.Gather()(self.linear_output_weights, index, 0)
            output_bias = P.Gather()(self.o_biases, index, 0)
            gating_weights = 0
            gating_bias = 0
            if self.gating:
                gating_weights = P.Gather()(self.linear_gating_weights, index, 0)
                gating_bias = P.Gather()(self.gating_biases, index, 0)
        else:
            q_weights = self.linear_q_weights
            k_weights = self.linear_k_weights
            v_weights = self.linear_v_weights
            output_weights = self.linear_output_weights
            output_bias = self.o_biases
            gating_weights = 0
            gating_bias = 0
            if self.gating:
                gating_weights = self.linear_gating_weights
                gating_bias = self.gating_biases

        b, _, _ = m_data.shape

        v_weights = P.BroadcastTo((b,
                                   self.dim_per_head * self.num_head,
                                   self.dim_per_head))(v_weights)
        v = self.batch_matmul(m_data, v_weights)

        mask_shape = q_mask.shape
        value_shape = q_data.shape
        broadcast_factor = 1.
        value_size = value_shape[1]
        mask_size = mask_shape[1]
        if mask_size == 1:
            broadcast_factor = broadcast_factor * value_size
        qa = P.ReduceSum()(q_mask * q_data, 1)
        qb = P.ReduceSum()(q_mask, 1) * broadcast_factor + 1e-10
        q_avg = P.RealDiv()(qa, qb)

        q = P.Reshape()(self.matmul(q_avg, q_weights),
                        (-1, self.num_head, self.dim_per_head)) * (self.dim_per_head ** (-0.5))

        k_weights = P.BroadcastTo((b,
                                   self.dim_per_head * self.num_head,
                                   self.dim_per_head))(k_weights)
        k = self.batch_matmul(m_data, k_weights)

        attention_mask = 1e9 * (P.Transpose()(q_mask, (0, 2, 1)) - 1.0)
        logits = P.Add()(self.batch_matmul_trans_b(q, k), attention_mask)

        weights = self.softmax(logits)
        weighted_avg = self.batch_matmul(weights, v)

        if self.gating:
            q_data_shape = P.Shape()(q_data)
            if len(q_data_shape) != 2:
                q_data = P.Reshape()(q_data, (-1, q_data_shape[-1]))
            out_shape = q_data_shape[:-1] + (-1,)
            gate_values = P.Reshape()(self.matmul_trans_b(q_data, gating_weights) + gating_bias,
                                      out_shape)

            gate_values = P.Reshape()(self.sigmoid(gate_values),
                                      (b, -1, self.num_head, self.dim_per_head))
            weighted_avg = P.Reshape()(P.ExpandDims()(weighted_avg, 1) * gate_values,
                                       (-1, self.num_head * self.dim_per_head))
            weighted_avg_shape = P.Shape()(weighted_avg)
            if len(weighted_avg_shape) != 2:
                weighted_avg = P.Reshape()(weighted_avg, (-1, weighted_avg_shape[-1]))
            output = P.Reshape()(P.Add()(self.matmul_trans_b(weighted_avg,
                                                             output_weights), output_bias),
                                 (b, -1, self.output_dim))
        else:
            weighted_avg = P.Reshape()(weighted_avg, (-1, self.num_head * self.dim_per_head))
            weighted_avg_shape = P.Shape()(weighted_avg)
            if len(weighted_avg_shape) != 2:
                weighted_avg = P.Reshape()(weighted_avg, (-1, weighted_avg_shape[-1]))
            out_shape = weighted_avg_shape[:-1] + (-1,)
            output = P.Reshape()(P.Add()(self.matmul_trans_b(weighted_avg, output_weights),
                                         output_bias), out_shape)
            output = P.ExpandDims()(output, -1)
        return output

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.linear_q_weights = Parameter(
                Tensor(np.zeros((self.batch_size,
                                 self.input_dim,
                                 self.num_head,
                                 self.dim_per_head)),
                       mstype.float32))
            self.linear_k_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim, self.dim_per_head)),
                       mstype.float32))
            self.linear_v_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim, self.dim_per_head)),
                       mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros((self.batch_size,
                                 self.output_dim,
                                 self.num_head * self.dim_per_head)),
                       mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.batch_size, self.output_dim)),
                                             mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros((self.batch_size,
                                     self.num_head * self.dim_per_head,
                                     self.input_dim)),
                           mstype.float32))
                self.gating_biases = Parameter(Tensor(np.zeros((self.batch_size, self.input_dim)),
                                                      mstype.float32))
        else:
            self.linear_q_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.input_dim,
                               self.dim_per_head * self.input_dim,
                               (self.input_dim, self.num_head*self.dim_per_head)),
                mstype.float32))
            self.linear_k_weights = Parameter(
                Tensor(glorot_uniform(self.input_dim,
                                      self.dim_per_head,
                                      (1, self.input_dim, self.dim_per_head)),
                       mstype.float32))
            self.linear_v_weights = Parameter(
                Tensor(glorot_uniform(self.input_dim,
                                      self.dim_per_head,
                                      (1, self.input_dim, self.dim_per_head)),
                       mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros((self.output_dim, self.num_head * self.dim_per_head)),
                       mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.output_dim)),
                                             mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros((self.num_head * self.dim_per_head, self.input_dim)),
                           mstype.float32))
                self.gating_biases = Parameter(Tensor(np.ones((self.input_dim)), mstype.float32))

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
"""MultiheadAttention operations"""
from typing import Optional
import math
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor, Parameter
from mindspore.ops.primitive import Primitive
from mindspore.nn.layer.activation import get_activation
from mindspore.common.initializer import Initializer, initializer, \
    XavierUniform, _calculate_fan_in_and_fan_out, _assignment, Constant
# pylint: disable=relative-beyond-top-level
from ...esm_if1.module.util import ms_transpose, utils_softmax
from .rotary_embedding import RotaryEmbedding


class XavierNormal(Initializer):
    """Xavier normalization"""

    def __init__(self, gain=1):
        super(XavierNormal, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)
        std = self.gain * math.sqrt(2.0 / (n_in + n_out))
        data = np.random.normal(0, std, arr.shape)
        _assignment(arr, data)


class Dense(nn.Cell):
    """
    preprocess input of each layer.
    """

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super(Dense, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        self.reshape = ops.Reshape()
        self.shape_op = ops.Shape()

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 2, and the first dim must be equal to 'out_channels', and the "
                                 f"second dim must be equal to 'in_channels'. But got 'weight_init': {weight_init}, "
                                 f"'out_channels': {out_channels}, 'in_channels': {in_channels}.")
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' must "
                                     f"be equal to 1, and the first dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add = ops.BiasAdd()

        self.matmul = ops.MatMul(transpose_b=True)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (nn.Cell, Primitive)):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, but got "
                            f"{type(activation).__name__}.")
        self.activation_flag = self.activation is not None
        self.get_dtype = ops.DType()

    def construct(self, x):
        """Dense construction"""
        x_shape = self.shape_op(x)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)
        return x


class MultiheadAttention(nn.Cell):
    """Multihead attention"""

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            use_rotary_embeddings=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        if self.self_attention:
            if not self.qkv_same_dim:
                raise ValueError("Self-attention requires query, key and " "value to be of the same size")

        self.k_proj = Dense(self.kdim, embed_dim, has_bias=bias)
        self.v_proj = Dense(self.vdim, embed_dim, has_bias=bias)
        self.q_proj = Dense(embed_dim, embed_dim, has_bias=bias)

        self.out_proj = Dense(embed_dim, embed_dim, has_bias=bias)

        if add_bias_kv:
            self.bias_k = ms.Parameter(ms.Tensor((1, 1, embed_dim)))
            self.bias_v = ms.Parameter(ms.Tensor((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.rot_emb = None
        if use_rotary_embeddings:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

        self.onnx_trace = False

        self.enable_torch_version = True
        self.unsqueeze = ops.ExpandDims()
        self.dropout_net = nn.Dropout(p=self.dropout)
        if self.training:
            self.dropout_net.set_train()

    def reset_parameters(self):
        """Reset parameters"""
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            self.k_proj.weight = initializer(XavierUniform(gain=1 / math.sqrt(2)),
                                             self.k_proj.weight.shape, self.k_proj.weight.dtype)
            self.v_proj.weight = initializer(XavierUniform(gain=1 / math.sqrt(2)),
                                             self.v_proj.weight.shape, self.v_proj.weight.dtype)
            self.q_proj.weight = initializer(XavierUniform(gain=1 / math.sqrt(2)),
                                             self.q_proj.weight.shape, self.q_proj.weight.dtype)
        else:
            self.k_proj.weight = initializer(XavierUniform(), self.k_proj.weight.shape,
                                             self.k_proj.weight.dtype)
            self.v_proj.weight = initializer(XavierUniform(), self.v_proj.weight.shape,
                                             self.v_proj.weight.dtype)
            self.q_proj.weight = initializer(XavierUniform(), self.q_proj.weight.shape,
                                             self.q_proj.weight.dtype)

        self.out_proj.weight = initializer(XavierUniform(), self.out_proj.weight.shape,
                                           self.out_proj.weight.dtype)
        if self.out_proj.bias is not None:
            self.out_proj.bias = initializer(Constant(0.0), self.out_proj.bias.shape)
        if self.bias_k is not None:
            self.bias_k = initializer(XavierNormal(), self.bias_k.shape,
                                      self.bias_k.dtype)
        if self.bias_v is not None:
            self.bias_v = initializer(XavierNormal(), self.bias_v.shape,
                                      self.bias_v.dtype)

    def construct(self, query, key: Optional[ms.Tensor], value: Optional[ms.Tensor],
                  key_padding_mask: Optional[ms.Tensor], need_weights: bool = True,
                  need_head_weights: bool = False):
        """Multihead attention construction"""
        if need_head_weights:
            need_weights = True
        tgt_len, bsz, embed_dim = query.shape

        k = self.k_proj(key)
        v = self.v_proj(key)
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        q = ms_transpose(q.view((tgt_len, bsz * self.num_heads, self.head_dim)), 0, 1)
        k = ms_transpose(k.view((-1, bsz * self.num_heads, self.head_dim)), 0, 1)
        v = ms_transpose(v.view((-1, bsz * self.num_heads, self.head_dim)), 0, 1)
        src_len = k.shape[1]
        if self.add_zero_attn:
            src_len += 1
            k = ops.Concat(1)([k, ops.Zeros()(((k.shape[0], 1) + k.shape[2:]), k.dtype)])
            k = ops.Concat(1)([k, ops.Zeros()(((k.shape[0], 1) + k.shape[2:]), k.dtype)])
            v = ops.Concat(1)([v, ops.Zeros()(((v.shape[0], 1) + v.shape[2:]), v.dtype)])
            key_padding_mask = ops.Concat(1)(
                [
                    key_padding_mask,
                    ops.Zeros()((key_padding_mask.shape[0], 1), key_padding_mask.dtype),
                ])
        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        k = ms_transpose(k, 1, 2)
        attn_weights = ops.BatchMatMul()(q, k)
        #     don't attend to padding symbols
        attn_weights = attn_weights.view((bsz, self.num_heads, tgt_len, src_len))
        key_padding_mask_ = self.unsqueeze(self.unsqueeze(key_padding_mask, 1), 2).astype(ms.bool_)
        attn_weights = ops.MaskedFill()(attn_weights, key_padding_mask_,
                                        ms.Tensor(-1e9, ms.float32))
        attn_weights = attn_weights.view((bsz * self.num_heads, tgt_len, src_len))

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.astype(attn_weights.dtype)
        attn_probs = self.dropout_net(attn_weights_float.astype(attn_weights.dtype))
        attn_probs = ops.Cast()(attn_probs, ms.float16)
        v = ops.Cast()(v, ms.float16)
        attn = ops.BatchMatMul()(attn_probs, v)
        attn = ops.Cast()(attn, ms.float32)

        attn = ms_transpose(attn, 0, 1).view((tgt_len, bsz, embed_dim))
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = ms_transpose(attn_weights_float.view((
                bsz, self.num_heads, tgt_len, src_len
            )), 1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(axis=0)
        return attn, attn_weights

# Copyright 2022 Huawei Technologies Co., Ltd
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
"""GVP operations, will be used in gvp_encoder.py"""
from typing import Dict, Optional, Tuple
import uuid
import math
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor, Parameter
from mindspore.ops.primitive import Primitive
from mindspore import _checkparam as Validator
from mindspore.nn.layer.activation import get_activation
from mindspore.common.initializer import Initializer, initializer,\
    XavierUniform, _calculate_fan_in_and_fan_out, _assignment, Constant
# pylint: disable=relative-beyond-top-level
from .message_passing import scatter_sum, MessagePassing
from .util import ms_transpose, _norm_no_nan, _split, tuple_cat, _merge, tuple_sum, tuple_index, utils_softmax


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


class GVP(nn.Cell):
    """GVP"""

    def __init__(self, in_dims, out_dims, h_dim=None, vector_gate=False,
                 activations=(ops.ReLU(), ops.Sigmoid()), tuple_io=True,
                 eps=1e-8):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.tuple_io = tuple_io
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = Dense(self.vi, self.h_dim, has_bias=False)
            self.ws = Dense(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = Dense(self.h_dim, self.vo, has_bias=False)
                if vector_gate:
                    self.wg = Dense(self.so, self.vo)
        else:
            self.ws = Dense(self.si, self.so)

        self.vector_gate = vector_gate
        self.scalar_act, self.vector_act = activations
        self.eps = eps

    def construct(self, x):
        """GVP construction"""

        if self.vi:
            s, v = x
            v = ms_transpose(v, (v.ndim - 1), (v.ndim - 2))
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2, eps=self.eps)
            concat_op = ops.Concat(axis=-1)
            s = self.ws(concat_op((s, vn)))
            if self.scalar_act:
                s = self.scalar_act(s)
            if self.vo:
                v = self.wv(vh)
                v = ms_transpose(v, (v.ndim - 1), (v.ndim - 2))
                if self.vector_gate:
                    unsqueeze = ops.ExpandDims()
                    g = unsqueeze(self.wg(s), -1)
                else:
                    g = _norm_no_nan(v, axis=-1, keepdims=True, eps=self.eps)
                if self.vector_act:
                    g = self.vector_act(g)
                    v = v * g
        else:
            if self.tuple_io:
                if x[1] is not None:
                    raise ValueError("'x[1]' should not be None")
                x = x[0]
            s = self.ws(x)
            if self.scalar_act:
                s = self.scalar_act(s)
            if self.vo:
                zeros = ops.Zeros()
                v = zeros(list(s.shape)[:-1] + [self.vo, 3])

        if self.vo:
            return (s, v)
        if self.tuple_io:
            return (s, None)
        return s


class _VDropout(nn.Cell):
    """Dropout"""

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate)
        self.ones = ops.Ones()
        self.unsqueeze = ops.ExpandDims()

    def construct(self, x):
        """Dropout construction"""

        if x is None:
            return None
        if not self.training:
            return x
        a = self.ones(x.shape[:-1], x.dtype)
        mask = self.dropout(a)
        mask = self.unsqueeze(mask, -1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Cell):
    """Dropout"""

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(1 - drop_rate)
        self.vdropout = _VDropout(1 - drop_rate)

    def construct(self, x):
        if isinstance(x, ms.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


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
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = Validator.check_bool(has_bias, "has_bias", self.cls_name)
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

        self.cast = ops.Cast()
        self.get_dtype = ops.DType()

    def construct(self, x):
        """Dense construction"""
        x = self.cast(x, ms.float16)

        x_shape = self.shape_op(x)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.cast(self.weight, x.dtype))
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, x.dtype))
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)

        x = self.cast(x, ms.float32)
        return x


class LayerNorm(nn.Cell):
    """Layer normalization"""

    def __init__(self, dims, tuple_io=True, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.tuple_io = tuple_io
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm([self.s])
        self.eps = eps

    def construct(self, x):
        """Layer normalization construction"""

        if not self.v:
            if self.tuple_io:
                return self.scalar_norm(x[0]), None
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False, eps=self.eps)
        nonzero_mask = (vn > 2 * self.eps)
        vn = (vn * nonzero_mask)
        nonzero_mask = ms.ops.Cast()(nonzero_mask, ms.float32)
        v_1 = ops.ReduceSum(keep_dims=True)(vn, axis=-2)
        v_2 = self.eps + ops.ReduceSum(keep_dims=True)(nonzero_mask, axis=-2)
        vn = v_1 / v_2
        sqrt = ops.Sqrt()
        vn = sqrt(vn + self.eps)
        v = nonzero_mask * (v / vn)
        return self.scalar_norm(s), v


class GVPConv(MessagePassing):
    """GVP Convolution"""

    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3,
                 vector_gate=False, module_list=None, aggr="mean", eps=1e-8,
                 activations=(ops.ReLU(), ops.Sigmoid())):
        super(GVPConv, self).__init__()
        self.eps = eps
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        self.aggr = aggr

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP((2 * self.si + self.se, 2 * self.vi + self.ve),
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims,
                        vector_gate=vector_gate, activations=activations)
                )
                for _ in range(n_layers - 2):
                    module_list.append(GVP(out_dims, out_dims,
                                           vector_gate=vector_gate))
                module_list.append(GVP(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.SequentialCell(*module_list)

    def construct(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(x_s, edge_index, s=x_s, v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
                                 edge_attr=edge_attr, aggr=self.aggr)
        output = _split(message, self.vo)
        return output

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        output = _merge(*message)
        return output


class GVPConvLayer(nn.Cell):
    """GVP Convolution layer"""

    def __init__(self, node_dims, edge_dims, vector_gate=False,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, attention_heads=0,
                 conv_activations=(ops.ReLU(), ops.Sigmoid()),
                 n_edge_gvps=0, layernorm=True, eps=1e-8):

        super(GVPConvLayer, self).__init__()
        if attention_heads == 0:
            self.conv = GVPConv(
                node_dims, node_dims, edge_dims, n_layers=n_message,
                vector_gate=vector_gate,
                aggr="add" if autoregressive else "mean",
                activations=conv_activations,
                eps=eps,
            )
        else:
            raise NotImplementedError
        if layernorm:
            self.norm = nn.CellList([LayerNorm(node_dims, eps=eps) for _ in range(2)])
        else:
            self.norm = nn.CellList([nn.Identity() for _ in range(2)])
        self.dropout = nn.CellList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP(node_dims, hid_dims, vector_gate=vector_gate))
            for _ in range(n_feedforward - 2):
                ff_func.append(GVP(hid_dims, hid_dims, vector_gate=vector_gate))
            ff_func.append(GVP(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.SequentialCell(*ff_func)

        self.edge_message_func = None
        if n_edge_gvps > 0:
            si, vi = node_dims
            se, ve = edge_dims
            module_list = [
                GVP((2 * si + se, 2 * vi + ve), edge_dims, vector_gate=vector_gate)
            ]
            for _ in range(n_edge_gvps - 2):
                module_list.append(GVP(edge_dims, edge_dims,
                                       vector_gate=vector_gate))
            if n_edge_gvps > 1:
                module_list.append(GVP(edge_dims, edge_dims,
                                       activations=(None, None)))
            self.edge_message_func = nn.SequentialCell(*module_list)
            if layernorm:
                self.edge_norm = LayerNorm(edge_dims, eps=eps)
            else:
                self.edge_norm = nn.Identity()
            self.edge_dropout = Dropout(drop_rate)

    def construct(self, x, edge_index, edge_attr,
                  autoregressive_x=None, node_mask=None):
        """GVP Convolution layer construction"""

        if self.edge_message_func:
            src, dst = edge_index
            if autoregressive_x is None:
                x_src = x[0][src], x[1][src]
            else:
                unsqueeze = ops.ExpandDims()
                mask = (src < dst)
                mask = unsqueeze(mask, -1)
                x_src = (
                    ms.numpy.where(mask, x[0][src], autoregressive_x[0][src]),
                    ms.numpy.where(unsqueeze(mask, -1), x[1][src],
                                   autoregressive_x[1][src])
                )
            x_dst = x[0][dst], x[1][dst]

            x_edge = (
                ops.Concat(axis=-1)([x_src[0], edge_attr[0], x_dst[0]]),
                ops.Concat(axis=-2)([x_src[1], edge_attr[1], x_dst[1]])
            )
            edge_attr_dh = self.edge_message_func(x_edge)
            edge_attr = self.edge_norm(tuple_sum(edge_attr,
                                                 self.edge_dropout(edge_attr_dh)))

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            unsqueeze = ops.ExpandDims()

            src = ops.OnesLike()(dst)
            index = ms.Tensor(dst, ms.int32)
            count = scatter_sum(src, index, dim_size=dh[0].shape[0])

            min_value = ms.Tensor(1, ms.float32)
            count = ops.clip_by_value(count, clip_value_min=min_value)
            count = unsqueeze(count, -1)

            dh = dh[0] / count, unsqueeze((dh[1] / count), -1)
        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)

        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_

        return x, edge_attr


class SinusoidalPositionalEmbedding(nn.Cell):
    """Sinusoidal positional embedding"""

    def __init__(self, embed_dim, padding_idx):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self._float_tensor = ms.Tensor(1, ms.float32)
        self.weights = None

    def construct(self, x):
        """Sinusoidal positional embedding construction"""

        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.shape[0]:
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.astype(self._float_tensor.dtype)

        positions = self.make_positions(x)
        positions = ops.Cast()(positions, ms.int32)
        output = ops.gather(self.weights, positions.view((-1)), 0).view((bsz, seq_len, -1))
        return ops.stop_gradient(output)


    def make_positions(self, x):
        mask = ops.NotEqual()(x, self.padding_idx)
        range_buf = ms.numpy.arange(x.shape[1]).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        floor = ops.Floor()
        mask = ops.Cast()(mask, ms.float32)
        return positions * floor(mask) + self.padding_idx * (1 - floor(mask))

    def get_embedding(self, num_embeddings):
        """Get sinusoidal positional embedding"""

        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = ops.Exp()(ms.numpy.arange(half_dim, dtype=ms.float32) * -emb)
        unsqueeze = ops.ExpandDims()
        emb = unsqueeze(ms.numpy.arange(num_embeddings, dtype=ms.float32), 1) * unsqueeze(emb, 0)
        concat = ops.Concat(1)
        emb = concat([ops.Sin()(emb), ops.Cos()(emb)]).view((num_embeddings, -1))
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = concat([emb, ops.Zeros()((num_embeddings, 1), ms.float32)])
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb


class FairseqIncrementalState:
    """Fair sequence incremental state"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def get_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
            key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
            self,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
            key: str,
            value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)


def with_incremental_state(cls):
    """Incremental state"""
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


@with_incremental_state
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

        self.onnx_trace = False

        self.enable_torch_version = True

    @staticmethod
    def apply_sparse_mask(attn_weights):
        return attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[ms.Tensor],
            prev_key_padding_mask: Optional[ms.Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[ms.Tensor]:
        """Append key padding masks"""

        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            prev_key_padding_mask = ops.Cast()(prev_key_padding_mask, ms.int32)
            key_padding_mask = ops.Cast()(key_padding_mask, ms.int32)
            new_key_padding_mask = ops.Concat(1)(
                [prev_key_padding_mask, key_padding_mask]
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = ops.Zeros()(
                (batch_size, src_len - prev_key_padding_mask.shape[1]), prev_key_padding_mask.dtype
            )
            prev_key_padding_mask = ops.Cast()(prev_key_padding_mask, ms.int32)
            filler = ops.Cast()(filler, ms.int32)
            new_key_padding_mask = ops.Concat(1)(
                [prev_key_padding_mask, filler]
            )
        elif key_padding_mask is not None:
            filler = ops.Zeros()(
                (batch_size, src_len - key_padding_mask.shape[1]),
                ms.int32,
            )

            key_padding_mask = ops.Cast()(key_padding_mask, ms.int32)
            if filler.shape == (1, 0):
                new_key_padding_mask = key_padding_mask
            else:
                new_key_padding_mask = ops.concat((filler, key_padding_mask), 1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

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
            self.out_proj.bias = initializer(Constant(0.0), self.out_proj.bias.shape, self.out_proj.bias.dtype)
        if self.bias_k is not None:
            self.bias_k = initializer(XavierNormal(), self.bias_k.shape,
                                      self.bias_k.dtype)
        if self.bias_v is not None:
            self.bias_v = initializer(XavierNormal(), self.bias_v.shape,
                                      self.bias_v.dtype)

    def construct(
            self,
            query,
            key: Optional[ms.Tensor],
            value: Optional[ms.Tensor],
            key_padding_mask: Optional[ms.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[ms.Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        """Multihead attention construction"""

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.shape
        if embed_dim != self.embed_dim or list(query.shape) != [tgt_len, bsz, embed_dim]:
            raise ValueError("embed_dim not equal to self.embed_dim, or query.shape not "
                             "equal to (tgt_len, bsz, embed_dim)")
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    if (not self.encoder_decoder_attention) or self.self_attention:
                        raise ValueError()
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                if value is not None:
                    raise ValueError()
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            if key is None or value is None:
                raise ValueError()
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            if self.bias_v is None:
                raise ValueError()
            k = ops.Concat()([k, ms.numpy.tile(self.bias_k, (1, bsz, 1))])
            v = ops.Concat()([v, ms.numpy.tile(self.bias_v, (1, bsz, 1))])
            if attn_mask is not None:
                attn_mask_zero = ops.Zeros()((attn_mask.shape[0], 1), attn_mask.dtype)
                attn_mask = ops.Concat(1)(
                    [attn_mask, attn_mask_zero]
                )
            if key_padding_mask is not None:
                key_padding_mask_zero = ops.Zeros()((key_padding_mask.shape[0], 1), key_padding_mask.dtype)
                key_padding_mask = ops.Concat(1)(
                    [
                        key_padding_mask,
                        key_padding_mask_zero
                    ]
                )

        q = ms_transpose(q.view((tgt_len, bsz * self.num_heads, self.head_dim)), 0, 1)
        if k is not None:
            k = ms_transpose(k.view((-1, bsz * self.num_heads, self.head_dim)), 0, 1)
        if v is not None:
            v = ms_transpose(v.view((-1, bsz * self.num_heads, self.head_dim)), 0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                o_prev_key = saved_state.get("prev_key", " ")
                if o_prev_key is None:
                    raise ValueError()
                prev_key = o_prev_key.view((bsz * self.num_heads, -1, self.head_dim))
                if static_kv:
                    k = prev_key
                else:
                    if k is None:
                        raise ValueError()
                    k = ops.Concat(1)([prev_key, k])
            if "prev_value" in saved_state:
                o_prev_value = saved_state.get("prev_value", " ")
                if o_prev_value is None:
                    raise ValueError()
                prev_value = o_prev_value.view((bsz * self.num_heads, -1, self.head_dim))
                if static_kv:
                    v = prev_value
                else:
                    if v is None:
                        raise ValueError()
                    v = ops.Concat(1)([prev_value, v])
            prev_key_padding_mask: Optional[ms.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state.get("prev_key_padding_mask", " ")
            if k is None or v is None:
                raise ValueError()
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.shape[1],
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view((bsz, self.num_heads, -1, self.head_dim))
            saved_state["prev_value"] = v.view((bsz, self.num_heads, -1, self.head_dim))
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            if incremental_state is None:
                raise ValueError()
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        if k is None:
            raise ValueError()
        src_len = k.shape[1]

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            if key_padding_mask.shape[0] != bsz or key_padding_mask.shape[1] != src_len:
                raise ValueError()

        if self.add_zero_attn:
            if v is None:
                raise ValueError()
            src_len += 1
            k = ops.Concat(1)([k, ops.Zeros()(((k.shape[0], 1) + k.shape[2:]), k.dtype)])
            k = ops.Concat(1)([k, ops.Zeros()(((k.shape[0], 1) + k.shape[2:]), k.dtype)])
            v = ops.Concat(1)([v, ops.Zeros()(((v.shape[0], 1) + v.shape[2:]), v.dtype)])
            if attn_mask is not None:
                attn_mask = ops.Concat(1)(
                    [attn_mask, ops.Zeros()((attn_mask.shape[0], 1), attn_mask.dtype)])
            if key_padding_mask is not None:
                key_padding_mask = ops.Concat(1)(
                    [
                        key_padding_mask,
                        ops.Zeros()((key_padding_mask.shape[0], 1), key_padding_mask.dtype),
                    ])

        q = ops.Cast()(q, ms.float16)
        k = ops.Cast()(ms_transpose(k, 1, 2), ms.float16)
        attn_weights = ops.BatchMatMul()(q, k)
        attn_weights = ops.Cast()(attn_weights, ms.float32)

        attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights)

        if list(attn_weights.shape) != [bsz * self.num_heads, tgt_len, src_len]:
            raise ValueError()
        unsqueeze = ops.ExpandDims()
        if attn_mask is not None:
            attn_mask = unsqueeze(attn_mask, 0)
            if self.onnx_trace:
                attn_mask = ms.numpy.tile(attn_mask, (attn_weights.shape[0], 1, 1))
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view((bsz, self.num_heads, tgt_len, src_len))
            key_padding_mask_ = unsqueeze(unsqueeze(key_padding_mask, 1), 2).astype(ms.bool_)
            attn_weights = ops.MaskedFill()(attn_weights, key_padding_mask_,
                                            ms.Tensor(-1e9, ms.float32))
            attn_weights = attn_weights.view((bsz * self.num_heads, tgt_len, src_len))

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.astype(attn_weights.dtype)

        dropout_net = nn.Dropout(p=self.dropout)
        if self.training:
            dropout_net.set_train()
        attn_probs = dropout_net(attn_weights_float.astype(attn_weights.dtype))

        if v is None:
            raise ValueError()

        attn_probs = ops.Cast()(attn_probs, ms.float16)
        v = ops.Cast()(v, ms.float16)
        attn = ops.BatchMatMul()(attn_probs, v)
        attn = ops.Cast()(attn, ms.float32)

        if list(attn.shape) != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError()
        if self.onnx_trace and attn.shape[1] == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.view((tgt_len, bsz, embed_dim))
        else:
            attn = ms_transpose(attn, 0, 1).view((tgt_len, bsz, embed_dim))
        attn = self.out_proj(attn)
        attn_weights: Optional[ms.Tensor] = None
        if need_weights:
            attn_weights = ms_transpose(attn_weights_float.view((
                bsz, self.num_heads, tgt_len, src_len
            )), 1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(axis=0)

        return attn, attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade state dict name"""

        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]]
    ) -> Dict[str, Optional[ms.Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        empty_result: Dict[str, Optional[ms.Tensor]] = {}
        return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[ms.Tensor]]],
            buffer: Dict[str, Optional[ms.Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)


def _set_input_buffer(selfattention: MultiheadAttention,
                      incremental_state: Dict[str, Dict[str, Optional[ms.Tensor]]],
                      buffer: Dict[str, Optional[ms.Tensor]],
                      ):
    """Set input buffer"""
    return selfattention.set_incremental_state(incremental_state, "attn_state", buffer)


def _get_input_buffer(
        selfattention: MultiheadAttention, incremental_state: Optional[Dict[str, Dict[str, Optional[ms.Tensor]]]]
) -> Dict[str, Optional[ms.Tensor]]:
    """Get input buffer"""
    result = selfattention.get_incremental_state(incremental_state, "attn_state")
    if result is not None:
        return result
    empty_result: Dict[str, Optional[ms.Tensor]] = {}
    return empty_result

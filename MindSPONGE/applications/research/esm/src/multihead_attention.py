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
"""Some functions used in transformer network"""

from typing import Dict, Optional, Tuple
import uuid
import math
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import Initializer, initializer,\
    XavierUniform, _calculate_fan_in_and_fan_out, _assignment
import mindspore.nn as nn
from mindspore import Tensor
from src.util import Dense


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


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    """Utils softmax"""
    if onnx_trace:
        return ops.Softmax(axis=dim)(ops.Cast()(x, ms.float32))
    x = x.astype(ms.float32)
    return ops.Softmax(axis=dim)(x)


def ms_transpose(x, index_a, index_b):
    """Transpose"""
    index = list(i for i in range(len(x.shape)))
    index[index_a] = index_b
    index[index_b] = index_a
    input_trans = x.transpose(index)
    return input_trans


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
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

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
            ms.common.initializer.Constant(value=0.0)(self.out_proj.bias)
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
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]


        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
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
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
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
                assert o_prev_key is not None
                prev_key = o_prev_key.view((bsz * self.num_heads, -1, self.head_dim))
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = ops.Concat(1)([prev_key, k])
            if "prev_value" in saved_state:
                o_prev_value = saved_state.get("prev_value", " ")
                assert o_prev_value is not None
                prev_value = o_prev_value.view((bsz * self.num_heads, -1, self.head_dim))
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = ops.Concat(1)([prev_value, v])
            prev_key_padding_mask: Optional[ms.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state.get("prev_key_padding_mask", " ")
            assert k is not None and v is not None
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
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.shape[1]

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.shape[0] == bsz
            assert key_padding_mask.shape[1] == src_len

        if self.add_zero_attn:
            assert v is not None
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

        assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]
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

        assert v is not None

        attn_probs = ops.Cast()(attn_probs, ms.float16)
        v = ops.Cast()(v, ms.float16)
        attn = ops.BatchMatMul()(attn_probs, v)
        attn = ops.Cast()(attn, ms.float32)

        assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
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

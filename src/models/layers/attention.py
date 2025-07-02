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
"""Attention module"""

from mindspore import ops, nn, Tensor
import mindspore.common.dtype as mstype

from .basic_block import DropPath


class Attention(nn.Cell):
    r"""Attention implementation base class

    Args:
        in_channels (int): The dimension of input vector.
        num_heads (int): The number of attention heads.
        compute_dtype (mindspore.dtype): Compute dtype. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.
        - **attn_mask** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, sequence\_len)` or
          :math:`(sequence\_len, sequence\_len)` or :math:`(batch\_size, num_heads, sequence\_len, sequence\_len)`.
        - **key_padding_mask** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len)` or
          :math:`(batch\_size, sequence\_len, sequence\_len)` or
          :math:`(batch\_size, num_heads, sequence\_len, sequence\_len)`.

    Outputs:
        - **output** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import Attention
        >>> model = Attention(in_channels=512, num_heads=4)
        >>> x = ops.rand((2, 32, 512))
        >>> q, k, v = model.get_qkv(x)
        >>> print(q.shape)
        (2, 4, 32, 128)
    """

    def __init__(self, in_channels, num_heads, compute_dtype=mstype.float32):
        super().__init__()
        self.num_heads = num_heads
        self.compute_dtype = compute_dtype
        self.softmax_func = nn.Softmax(axis=-1)
        self.matmul = ops.BatchMatMul()
        self.qkv = nn.Dense(
            in_channels, in_channels * 3, weight_init="XavierUniform"
        ).to_float(compute_dtype)

    def softmax(self, scores, compute_dtype=mstype.float32):
        if scores.dtype != mstype.float32:
            scores = scores.astype(mstype.float32)
        attn = self.softmax_func(scores)
        if compute_dtype == mstype.float32:
            return attn
        return attn.astype(compute_dtype)

    def _mask_scores(self, scores, attn_mask=None, key_padding_mask=None):
        """mask attention scores"""
        batch, _, _, node = scores.shape
        mask = ops.zeros_like(scores)
        if attn_mask is not None:
            attn_mask = attn_mask.astype(scores.dtype)
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.reshape(1, 1, node, node)
            elif len(attn_mask.shape) == 3:
                attn_mask = attn_mask.unsqueeze(1)
            else:
                pass
            mask += attn_mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.astype(scores.dtype)
            if len(key_padding_mask.shape) == 2:
                key_padding_mask = ops.broadcast_to(key_padding_mask.unsqueeze(1), (batch, node, node)).unsqueeze(1)
            elif len(key_padding_mask.shape) == 3:
                key_padding_mask = key_padding_mask.unsqueeze(1)
            else:
                pass
            mask += key_padding_mask
        scores += mask * Tensor(-1e10, scores.dtype)
        return scores

    def get_qkv(self, x):
        b, n, _ = x.shape
        qkv = (
            self.qkv(x).reshape(b, n, 3, self.num_heads, -
                                1).transpose((2, 0, 3, 1, 4))
        )
        return qkv[0], qkv[1], qkv[2]

    def _reshape_output(self, x):
        b, _, n, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def construct(self, x, attn_mask=None, key_padding_mask=None):
        """Attention network construction."""
        raise NotImplementedError


class MultiHeadAttention(Attention):
    r"""Multi Head Attention proposed in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        in_channels (int): The input channels.
        num_heads (int): The number of attention heads.
        drop_mode (str): Dropout method, ``dropout`` or ``droppath``. Default: ``dropout``.
        dropout_rate (float): The drop rate of dropout layer, greater than 0 and less equal than 1. Default: ``0.0``.
        compute_dtype (mindspore.dtype): Compute dtype. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.
        - **attn_mask** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, sequence\_len)` or
          :math:`(sequence\_len, sequence\_len)` or :math:`(batch\_size, num_heads, sequence\_len, sequence\_len)`.
        - **key_padding_mask** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len)` or
          :math:`(batch\_size, sequence\_len, sequence\_len)` or
          :math:`(batch\_size, num_heads, sequence\_len, sequence\_len)`.

    Outputs:
        - **output** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import MultiHeadAttention
        >>> model = MultiHeadAttention(in_channels=512, num_heads=4)
        >>> x = ops.rand((2, 32, 512))
        >>> mask_shape = (2, 4, 32, 32)
        >>> mask = ops.ones(mask_shape)
        >>> output = model(x, mask)
        >>> print(output.shape)
        (2, 32, 512)
    """

    def __init__(self, in_channels,
                 num_heads,
                 drop_mode="dropout",
                 dropout_rate=0.0,
                 compute_dtype=mstype.float32,
                 ):
        super().__init__(in_channels, num_heads, compute_dtype)
        assert (
            in_channels % num_heads == 0
        ), "hidden channels must be divisible by number of heads"
        self.scale = (in_channels // num_heads) ** -0.5
        self.proj = nn.Dense(
            in_channels, in_channels, weight_init="XavierUniform"
        ).to_float(compute_dtype)

        if drop_mode == "dropout":
            self.drop = nn.Dropout(p=dropout_rate)
            self.attn_drop = nn.Dropout(p=dropout_rate)
        else:
            self.drop = DropPath(dropout_rate=dropout_rate)
            self.attn_drop = DropPath(dropout_rate=dropout_rate)

    def construct(self, x, attn_mask=None, key_padding_mask=None):
        """construct"""
        query, key, value = self.get_qkv(x)
        scores = self.matmul(query, key.swapaxes(-1, -2)) * self.scale
        scores = self._mask_scores(scores, attn_mask, key_padding_mask)
        attn = self.softmax(scores, self.compute_dtype)
        attn = self.attn_drop(attn)
        output = self.matmul(attn, value)
        output = self._reshape_output(output)

        output = self.proj(output)
        output = self.drop(output)
        return output


class Mlp(nn.Cell):
    """Mlp"""

    def __init__(self, in_channels, dropout_rate=0.0, compute_dtype=mstype.float16):
        super().__init__()
        self.fc1 = nn.Dense(
            in_channels, in_channels * 4, weight_init="XavierUniform"
        ).to_float(compute_dtype)
        self.fc2 = nn.Dense(
            in_channels * 4, in_channels, weight_init="XavierUniform"
        ).to_float(compute_dtype)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def construct(self, x):
        """construct"""
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Cell):
    r"""
    `AttentionBlock` comprises an `MultiHeadAttention` and an `MLP` layer.

    Args:
        in_channels (int): The input channels.
        num_heads (int): The number of attention heads.
        drop_mode (str): Dropout method. Default: ``dropout``. Support ``dropout`` or ``droppath``.
        dropout_rate (float): The drop rate of dropout layer, greater than 0 and less equal than 1. Default: ``0.0``.
        compute_dtype (mindspore.dtype): Compute dtype. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.
        - **mask** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, sequence\_len)` or
          :math:`(sequence\_len, sequence\_len)` or :math:`(batch\_size, num_heads, sequence\_len, sequence\_len)`.

    Outputs:
        - **output** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import AttentionBlock
        >>> model = AttentionBlock(in_channels=256, num_heads=4)
        >>> x = ops.rand((4, 100, 256))
        >>> output = model(x)
        >>> print(output.shape)
        (4, 100, 256)
    """

    def __init__(self,
                 in_channels,
                 num_heads,
                 drop_mode="dropout",
                 dropout_rate=0.0,
                 compute_dtype=mstype.float32,
                 ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.attention_norm = nn.LayerNorm([in_channels], epsilon=1e-6).to_float(
            mstype.float32
        )
        self.ffn_norm = nn.LayerNorm([in_channels], epsilon=1e-6).to_float(
            mstype.float32
        )
        self.ffn = Mlp(
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype,
        )
        self.attention = MultiHeadAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            drop_mode=drop_mode,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype,
        )

    def construct(self, x, mask=None):
        """construct"""
        h = x
        x = self.attention_norm(x)
        x = self.attention(x, mask)
        x = x + h

        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

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
from typing import Optional
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
        - **attn_mask** (Tensor, optional) - Tensor with shape :math:`(sequence\_len, sequence\_len)` or
          or :math:`(batch\_size, 1, sequence\_len, sequence\_len)`. Default: ``None``.
        - **key_padding_mask** (Tensor, optional) - Tensor with shape :math:`(batch\_size, sequence\_len)`.
          Default: ``None``.

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

    def __init__(self, in_channels: int, num_heads: int, compute_dtype: mstype = mstype.float32):
        super().__init__()
        self.num_heads = num_heads
        self.compute_dtype = compute_dtype
        self.qkv = nn.Dense(
            in_channels, in_channels * 3, weight_init="XavierUniform"
        ).to_float(compute_dtype)

    @staticmethod
    def merge_mask(attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """merge mask"""
        if attn_mask is None and key_padding_mask is None:
            return None
        mask = Tensor(0, dtype=mstype.uint8)
        if attn_mask is not None:
            node = attn_mask.shape[-1]
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.reshape(1, 1, node, node)
            elif len(attn_mask.shape) == 4:
                pass
            else:
                raise Exception(f'attn_mask shape {attn_mask.shape} not support')
            mask = mask + attn_mask.astype(mstype.uint8)
        if key_padding_mask is not None:
            batch, node = key_padding_mask.shape[0], key_padding_mask.shape[-1]
            if len(key_padding_mask.shape) == 2:
                key_padding_mask = ops.broadcast_to(key_padding_mask.unsqueeze(1), (batch, node, node)).unsqueeze(1)
            else:
                raise Exception(f'key_padding_mask shape {attn_mask.shape} not support')
            mask = mask + key_padding_mask.astype(mstype.uint8)
        return mask

    @staticmethod
    def mask_scores(scores: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """mask attention scores"""
        if mask is None:
            return scores
        scores += mask * Tensor(-1e10, scores.dtype)
        return scores

    def get_qkv(self, x: Tensor) -> tuple[Tensor]:
        """get qkv value"""
        b, n, _ = x.shape
        qkv = (
            self.qkv(x).reshape(b, n, 3, self.num_heads, -
                                1).transpose((2, 0, 3, 1, 4))
        )
        return qkv[0], qkv[1], qkv[2]

    def _reshape_output(self, x: Tensor) -> Tensor:
        b, _, n, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, n, -1)

    def construct(self, x: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None):
        """Attention network construction."""
        raise NotImplementedError


class ScaledDot(nn.Cell):
    """Scaled dot attention"""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def construct(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
        scores = ops.matmul(query, key.swapaxes(-1, -2)) * self.scale
        scores = Attention.mask_scores(scores, mask)
        scores = scores.astype(mstype.float32)
        attn = ops.softmax(scores, axis=-1)
        attn = attn.astype(value.dtype)
        output = ops.matmul(attn, value)
        return output


class FlashAttn(nn.Cell):
    r"""FlashAttention proposed in `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_.

    Args:
        num_heads (int): The number of attention heads.
        scale (float): The attention scale.
        fa_dtype (mindspore.dtype, optional): FlashAttention compute dtype. Choose from `mstype.bfloat16`,
            `mstype.float16`. Default: ``mstype.bfloat16``, indicates ``mindspore.bfloat16``.

    Inputs:
        - **query** (Tensor) - Tensor with shape :math:`(batch\_size, num\_heads, sequence\_len, in\_channels)`.
        - **key** (Tensor) - Tensor with shape :math:`(batch\_size, num\_heads, sequence\_len, in\_channels)`.
        - **value** (Tensor) - Tensor with shape :math:`(batch\_size, num\_heads, sequence\_len, in\_channels)`.
        - **mask** (Tensor, optional) - Tensor with shape :math:`(sequence\_len, sequence\_len)` or
          or :math:`(batch\_size, 1, sequence\_len, sequence\_len)`. Default: ``None``.

    Outputs:
        - **output** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import FlashAttn
        >>> model = FlashAttn(num_heads=4, scale=0.25)
        >>> in_shape = (2, 16, 32, 16)
        >>> q, k, v = ops.rand(in_shape), ops.rand(in_shape), ops.rand(in_shape)
        >>> mask_shape = (32, 32)
        >>> mask = ops.randint(0, 2, mask_shape)
        >>> output = model(q, k, v, mask)
        >>> print(output.shape)
        (2, 16, 32, 16)
    """

    def __init__(self, num_heads: int, scale: float, fa_dtype=mstype.bfloat16):
        super().__init__()
        self.fa_dtype = fa_dtype
        self.num_heads = num_heads
        self.scale = scale

    def construct(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None):
        query, key, value = query.astype(self.fa_dtype), key.astype(self.fa_dtype), value.astype(self.fa_dtype)
        if mask is not None:
            mask = mask.astype(mstype.uint8)
        scores = ops.flash_attention_score(query, key, value, input_layout='BNSD', head_num=self.num_heads,
                                           attn_mask=mask, scalar_value=self.scale)
        return scores


class MultiHeadAttention(Attention):
    r"""Multi Head Attention proposed in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        in_channels (int): The input channels.
        num_heads (int): The number of attention heads.
        enable_flash_attn (bool): Whether use flash attention. FlashAttention only supports Ascend backend.
            FlashAttention proposed in `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_.
            Default: ``False``.
        fa_dtype (mindspore.dtype): FlashAttention compute dtype. Choose from `mstype.bfloat16`, `mstype.float16`.
            Default: ``mstype.bfloat16``, indicates ``mindspore.bfloat16``.
        drop_mode (str): Dropout method, ``dropout`` or ``droppath``. Default: ``dropout``.
        dropout_rate (float): The drop rate of dropout layer, greater than 0 and less equal than 1. Default: ``0.0``.
        compute_dtype (mindspore.dtype): Compute dtype. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.
        - **attn_mask** (Tensor, optional) - Tensor with shape :math:`(sequence\_len, sequence\_len)` or
          or :math:`(batch\_size, 1, sequence\_len, sequence\_len)`. Default: ``None``.
        - **key_padding_mask** (Tensor, optional) - Tensor with shape :math:`(batch\_size, sequence\_len)`.
          Default: ``None``.

    Outputs:
        - **output** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import MultiHeadAttention
        >>> model = MultiHeadAttention(in_channels=512, num_heads=4)
        >>> x = ops.rand((2, 32, 512))
        >>> mask_shape = (32, 32)
        >>> mask = ops.ones(mask_shape)
        >>> output = model(x, mask)
        >>> print(output.shape)
        (2, 32, 512)
    """

    def __init__(self, in_channels: int,
                 num_heads: int,
                 enable_flash_attn: bool = False,
                 fa_dtype: mstype = mstype.bfloat16,
                 drop_mode: str = "dropout",
                 dropout_rate: float = 0.0,
                 compute_dtype: mstype = mstype.float32,
                 ):
        super().__init__(in_channels, num_heads, compute_dtype)
        assert (
            in_channels % num_heads == 0
        ), "hidden channels must be divisible by number of heads"
        scale = (in_channels // num_heads) ** -0.5
        self.proj = nn.Dense(in_channels, in_channels).to_float(compute_dtype)
        if enable_flash_attn:
            print('use flash attention')
            self.attn = FlashAttn(num_heads=num_heads, scale=scale, fa_dtype=fa_dtype)
        else:
            self.attn = ScaledDot(scale=scale)
        if drop_mode == "dropout":
            self.drop = nn.Dropout(p=dropout_rate)
        else:
            self.drop = DropPath(dropout_rate=dropout_rate)

    def construct(self, x: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None):
        """construct"""
        query, key, value = self.get_qkv(x)
        mask = self.merge_mask(attn_mask, key_padding_mask)
        output = self.attn(query, key, value, mask)
        output = output.astype(mstype.float32)
        output = self._reshape_output(output)
        output = self.proj(output)
        output = self.drop(output)
        return output


class FeedForward(nn.Cell):
    """FeedForward"""
    def __init__(self, in_channels, dropout_rate=0.0, compute_dtype=mstype.float16):
        super().__init__()
        self.fc1 = nn.Dense(in_channels, in_channels * 4).to_float(compute_dtype)
        self.fc2 = nn.Dense(in_channels * 4, in_channels).to_float(compute_dtype)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def construct(self, x: Tensor):
        """construct"""
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Cell):
    r""" `TransformerBlock` comprises an `MultiHeadAttention` and an `FeedForward` layer.

    Args:
        in_channels (int): The input channels.
        num_heads (int): The number of attention heads.
        enable_flash_attn (bool): Whether use flash attention. FlashAttention only supports Ascend backend.
            FlashAttention proposed in `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness <https://arxiv.org/abs/2205.14135>`_.
            Default: ``False``.
        fa_dtype (mindspore.dtype): FlashAttention compute dtype. Choose from `mstype.bfloat16`, `mstype.float16`.
            Default: ``mstype.bfloat16``, indicates ``mindspore.bfloat16``.
        drop_mode (str): Dropout method. Default: ``dropout``. Support ``dropout`` or ``droppath``.
        dropout_rate (float): The drop rate of dropout layer, greater than 0 and less equal than 1. Default: ``0.0``.
        compute_dtype (mindspore.dtype): Compute dtype. Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.
        - **mask** (Tensor, optional) - Tensor with shape :math:`(sequence\_len, sequence\_len)` or
          :math:`(batch\_size, 1, sequence\_len, sequence\_len)`. Default: ``None``.

    Outputs:
        - **output** (Tensor) - Tensor with shape :math:`(batch\_size, sequence\_len, in\_channels)`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import TransformerBlock
        >>> model = TransformerBlock(in_channels=256, num_heads=4)
        >>> x = ops.rand((4, 100, 256))
        >>> output = model(x)
        >>> print(output.shape)
        (4, 100, 256)
    """

    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 enable_flash_attn: bool = False,
                 fa_dtype: mstype = mstype.bfloat16,
                 drop_mode: str = "dropout",
                 dropout_rate: float = 0.0,
                 compute_dtype: mstype = mstype.float32,
                 ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.attention_norm = nn.LayerNorm([in_channels], epsilon=1e-6).to_float(
            mstype.float32
        )
        self.ffn_norm = nn.LayerNorm([in_channels], epsilon=1e-6).to_float(
            mstype.float32
        )
        self.ffn = FeedForward(
            in_channels=in_channels,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype,
        )
        self.attention = MultiHeadAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            enable_flash_attn=enable_flash_attn,
            fa_dtype=fa_dtype,
            drop_mode=drop_mode,
            dropout_rate=dropout_rate,
            compute_dtype=compute_dtype,
        )

    def construct(self, x: Tensor, mask: Optional[Tensor] = None):
        """construct"""
        h = x
        x = self.attention_norm(x)
        x = self.attention(x, mask)
        x = x + h

        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

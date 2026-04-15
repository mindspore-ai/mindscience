# Copyright 2025 Huawei Technologies Co., Ltd
#
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

from typing import Literal, TypeAlias
import typing
import alphafold3.utils.attention.attention_base as base
import alphafold3.utils.attention.ms_attention as ms_attention

Implementation: TypeAlias = Literal["ms"]


def dot_product_attention(query, key, value, *, bias, mask, implementation,
                          logits_dtype=None, precision=None):
    """Performs scaled dot-product attention.

    Scaled dot-product attention from "Attention is all you need"
    https://arxiv.org/abs/1706.03762.

    Computes self- or cross-attention. The following is computed:
    softmax(qk_scale * query @ key^T + bias) @ value.

    Supports both multi-head and multi-query attention
    (https://arxiv.org/abs/1911.02150).

    Arguments:
      query: Query array of shape `[batch, seq_len_q, num_heads, head_dim]`.
      key: Key array of shape `[batch, seq_len_kv, num_heads, head_dim]`.
        `num_heads` can be 1 for multi-query attention.
      value: Value array of shape `[batch, seq_len_kv, num_heads, head_dim]`.
        `num_heads` can be 1 for multi-query attention.
      bias: Optional bias array, broadcastable to shape `[batch, num_heads,
        seq_len_q, seq_len_kv]`.
      mask: Optional boolean mask, broadcastable to `[batch, num_heads, seq_len_q,
        seq_len_kv]`. Attention weights are masked out if the corresponding mask
        value is `False`.
      implementation: if `None` (default), an implementation is automatically
        chosen. 'ms' will use standard MS and work on any platform.
      logits_dtype: Data type for attention logits (`query @ key^T`). If `None` is
        passed (the default), the accumulator type from the `query @ key^T` dot
        product will be used, which is FP32 for BF16/FP16/FP32 inputs. Note that
        this default increases the memory usage for BF16/FP16 inputs when using
        `implementation='ms'`.
      precision: The precision for the dot products. Either a single or a tuple
      of `DEFAULT` precision.

    Returns:
      An array with the same shape as `query`.
    """

    if implementation is not None:
        named_args = typing.get_args(Implementation)
        if implementation not in named_args:
            raise ValueError(
                f"Unsupported named implementation. Must be one of {named_args}."
            )

    logits_dtype = base.AUTO if logits_dtype is None else logits_dtype
    precision = "DEFAULT" if precision is None else precision

    args = (query, key, value)
    kwargs = dict(
        precision=precision,
        logits_dtype=logits_dtype,
        bias=bias,
        mask=mask,
    )

    return ms_attention.MsDotProductAttention()(*args, **kwargs)

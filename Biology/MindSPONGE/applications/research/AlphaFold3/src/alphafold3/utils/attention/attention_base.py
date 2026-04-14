# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend

import abc
import enum
import math
import dataclasses
import functools
from dataclasses import dataclass, KW_ONLY
from typing import Any
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from alphafold3.utils.common import precision as precision_lib


@dataclasses.dataclass(frozen=True)
class Mask:
    """An attention mask.

        `k_start` (inclusive) and `k_end` (exclusive) define range of enabled
        k-sequence values for each row of logits.

        For example, a local attention mask could be defined as follows:
        ```
        seq_len_q = seq_len_k = 4
        window_size = 2
        k_start = Tensor(np.maximum(0, np.arange(seq_len_q) + 1 - window_size))
        mask = Mask(k_start=k_start, is_causal=True)
        assert mask.as_array(seq_len_q, seq_len_k) == Tensor(np.array(
            [[1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]], dtype=bool))
        ```
    """
    bool_mask: ms.Tensor | None = None
    _: dataclasses.KW_ONLY
    q_start: ms.Tensor | None = None
    q_end: ms.Tensor | None = None
    k_start: ms.Tensor | None = None
    k_end: ms.Tensor | None = None
    is_causal: bool = False

    def tree_flatten(self):
        return (
            self.bool_mask,
            self.q_start,
            self.q_end,
            self.k_start,
            self.k_end,
        ), (self.is_causal,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (is_causal,) = aux
        bool_mask, q_start, q_end, k_start, k_end = children
        return cls(
            bool_mask,
            q_start=q_start,
            q_end=q_end,
            k_start=k_start,
            k_end=k_end,
            is_causal=is_causal,
        )

    def as_array(self, q_len_or_indices, k_len_or_indices):
        """Returns the mask as a boolean array."""
        q_indices = ops.arange(q_len_or_indices) if isinstance(
            q_len_or_indices, int) else q_len_or_indices
        q_indices = q_indices[..., None]

        k_indices = ops.arange(k_len_or_indices) if isinstance(
            k_len_or_indices, int) else k_len_or_indices
        k_indices = k_indices[..., None, :]

        mask = []
        if self.bool_mask is not None:
            mask.append(self.bool_mask)

        if self.q_start is not None:
            mask.append(q_indices >= self.q_start[..., None, :])

        if self.q_end is not None:
            mask.append(q_indices < self.q_end[..., None, :])

        if self.k_start is not None:
            mask.append(k_indices >= self.k_start[..., None])

        if self.k_end is not None:
            mask.append(k_indices < self.k_end[..., None])

        if self.is_causal:
            mask.append(q_indices >= k_indices)

        logical_and = functools.partial(functools.reduce, ops.logical_and)

        if mask:
            return logical_and(mask)
        else:
            return None

    def take(self, *attrs):
        """Returns a mask with attrs removed and the removed attrs."""
        default_mask = type(self)()
        replacements = {attr: getattr(default_mask, attr) for attr in attrs}
        values = (getattr(self, attr) for attr in attrs)
        return dataclasses.replace(self, **replacements), *values

    def __and__(self, other):
        """Returns the intersection of two masks."""
        if not isinstance(other, Mask):
            other = Mask(other)

        def combine(op):
            return lambda a, b: b if a is None else a if b is None else op(a, b)

        return Mask(
            bool_mask=combine(ops.logical_and)(
                self.bool_mask, other.bool_mask),
            q_end=combine(ops.minimum)(self.q_end, other.q_end),
            k_start=combine(ops.maximum)(self.k_start, other.k_start),
            k_end=combine(ops.minimum)(self.k_end, other.k_end),
            is_causal=self.is_causal or other.is_causal,
        )


CAUSAL_MASK = Mask(is_causal=True)


@enum.unique
class SoftmaxResidualMode(enum.Enum):
    """The mode of storing softmax residuals for the backwards pass.

    The stable softmax calculation performs two reductions calculating:
        - the maximum input value (`x_max`),
        - the sum of exponentiated values (`denom`).

    We can store these values as residuals to avoid the need to recompute them
    in the backwards pass.

    It is also possible to combine the two residuals into a single residual,
    `res = x_max + log(denom)`, as `exp(x - res) === exp(x - x_max - log(denom))
    === exp(x - x_max) / denom`. Combining the residuals reduces the memory usage
    of the residuals, but will reduce the accuracy of the backwards pass if
    `abs(x_max) >> log(denom)`.
    """

    SEPARATE = "separate"
    COMBINED = "combined"

    def conform(self, aux):
        match self, aux:
            case None, _:
                return None
            case SoftmaxResidualMode.SEPARATE, (_, _):
                return aux
            case SoftmaxResidualMode.SEPARATE, _:  # pytype: disable=redundant-match  # b/300135240
                raise ValueError("`aux` has been combined.")
            case SoftmaxResidualMode.COMBINED, (x_max, denom):
                return x_max + ops.log(denom)
            case SoftmaxResidualMode.COMBINED, _:  # pytype: disable=redundant-match  # b/300135240
                return aux


class DotProductAttention(abc.ABC):
    """Dot product attention function."""

    def __call__(self, query, key, value, *, precision, logits_dtype, bias, mask, q_indices=None, k_indices=None):
        """Performs scaled dot-product attention.

        Scaled dot-product attention from "Attention is all you need"
        https://arxiv.org/abs/1706.03762.

        Computes self- or cross-attention. The following is computed:
        softmax(qk_scale * query @ key^T + bias) @ value.

        Supports both multi-head and multi-query attention
        (https://arxiv.org/abs/1911.02150).

        Arguments:
        query: Query array of shape `[batch, seq_len_q, num_heads_q, head_dim]`.
            It must be a multiple of num_heads_kv.
            Here's an example of how q/kv heads are interleaved:
            For 8 key/value heads and 4 query heads:
            - key/value heads [0, 1] see query head 0
            - key/value heads [2, 3] see query head 1
            - key/value heads [4, 5] see query head 2
        key: Key array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`. It
            must be divisible by num_heads_q.
        value: Value array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`.
        precision: The precision for the dot products. Either a tuple `(
            query_key_dot_precision, weights_value_dot_precision)` or a single
            precision applied to both dot products.
        logits_dtype: Data type for attention logits (`query @ key^T`). If `AUTO`
            is passed (the default), the accumulator type from the `query @ key^T`
            dot product will be used.
        bias: Optional bias array, broadcastable to shape `[batch, num_heads,
            seq_len_q, seq_len_kv]`.
        mask: Optional boolean mask, broadcastable to `[batch, num_heads,
            seq_len_q, seq_len_kv]`. Attention weights are masked out if the
            corresponding mask value is `False`.
        q_indices: Optional indices for each token in query sequence.
        k_indices: Optional indices for each token in key/value sequence.

        Returns:
        An array with the same shape as `query`.
        """
        return self.fwd(
            query,
            key,
            value,
            precision=precision,
            logits_dtype=logits_dtype,
            bias=bias,
            mask=mask,
            q_indices=q_indices,
            k_indices=k_indices,
        )

    def fwd(self, query, key, value, *, precision, logits_dtype, bias, mask, q_indices, k_indices):
        """Performs attention."""
        if not isinstance(precision, tuple):
            precision = (precision, precision)

        q_k_dot_precision, weights_v_dot_precision = precision

        if not isinstance(q_k_dot_precision, precision_lib.DotPrecision):
            q_k_dot_precision = precision_lib.get_equivalent_dot_precision(
                query.dtype, key.dtype, q_k_dot_precision
            )

        if not isinstance(weights_v_dot_precision, precision_lib.DotPrecision):
            weights_v_dot_precision = precision_lib.get_equivalent_dot_precision(
                value.dtype, value.dtype, weights_v_dot_precision
            )

        if not isinstance(mask, Mask):
            mask = Mask(mask)

        return self._fwd(
            Tensor(query),
            Tensor(key),
            Tensor(value),
            q_k_dot_precision=q_k_dot_precision,
            logits_dtype=logits_dtype,
            logits_scale=1 / math.sqrt(query.shape[-1]),
            bias=bias,
            mask=mask,
            weights_v_dot_precision=weights_v_dot_precision,
            q_indices=q_indices,
            k_indices=k_indices,
        )

    @abc.abstractmethod
    def _fwd(self, q, k, v, *, q_k_dot_precision, logits_dtype, logits_scale, bias, mask,
             weights_v_dot_precision, q_indices, k_indices):
        """Performs attention."""
        ...

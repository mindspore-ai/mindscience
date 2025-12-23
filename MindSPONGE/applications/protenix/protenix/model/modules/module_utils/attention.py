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

"""MindSpore implementation of dot product attention."""
import dataclasses
import functools
import mindspore as ms
from mindspore import ops


def _softmax(x):
    """Computes softmax."""
    dtype = ms.float32
    x_max, _ = ops.max(x.astype(dtype), axis=-1, keepdims=True)
    unnormalized = ops.exp(x - x_max)
    denom = ops.sum(unnormalized, dim=-1, keepdim=True)
    return (unnormalized / denom).astype(x.dtype)


def cal_logits(q, k, use_bf16=False):
    """Calculate logits."""
    # ...qhd,...khd->...hqk
    dtype = q.dtype
    if use_bf16:
        q = q.astype(ms.bfloat16)
        k = k.astype(ms.bfloat16)
    q_trans = ops.transpose(q, (0, 2, 1, 3))  # ...qhd -> ...hqd
    k_trans = ops.transpose(k, (0, 2, 3, 1))  # ...khd -> ...hdk
    logits = ops.matmul(q_trans, k_trans)
    if use_bf16:
        logits = logits.astype(dtype)
    return logits


def cal_out(weights, v, use_bf16=False):
    """Calculate output."""
    # ...hqk,...khd->...qhd
    if use_bf16:
        weights = weights.astype(ms.bfloat16)
        v = v.astype(ms.bfloat16)
    v_trans = ops.transpose(v, (0, 2, 1, 3))  # ...khd -> ...hkd
    out_temp = ops.matmul(weights, v_trans)  # ...hqk,...hkd->...hqd
    out = ops.transpose(out_temp, (0, 2, 1, 3))
    return out


def attention(
        q, k, v, *, logits_scale,
        bias, mask
):
    """Compute attention."""
    logits = cal_logits(q, k)

    logits *= logits_scale

    if bias is not None:
        logits += bias

    if mask is not None:
        if not isinstance(mask, Mask):
            mask = Mask(mask)
        mask = mask.as_array(q.shape[-3], k.shape[-3])
        mask_value = -3.4028235e+37  # a small value close to min of bfloat16
        logits = ops.where(mask.bool(), logits, mask_value)

    weights = _softmax(logits)

    out = cal_out(weights, v)

    return out

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
    bool_mask: ms.Tensor = None
    _: dataclasses.KW_ONLY
    q_start: ms.Tensor = None
    q_end: ms.Tensor = None
    k_start: ms.Tensor = None
    k_end: ms.Tensor = None
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

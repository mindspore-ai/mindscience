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

import dataclasses
import mindspore as ms
from mindspore import ops
import alphafold3.utils.attention.attention_base as base


def _softmax(x):
    """Computes softmax."""
    dtype = ms.float32
    x_max, _ = ops.max(x.astype(dtype), axis=-1, keepdims=True)
    unnormalized = ops.exp(x - x_max)
    denom = ops.sum(unnormalized, dim=-1, keepdim=True)
    return (unnormalized / denom).astype(x.dtype)


def cal_logits(q, k, use_bf16=False):
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
    # ...hqk,...khd->...qhd
    if use_bf16:
        weights = weights.astype(ms.bfloat16)
        v = v.astype(ms.bfloat16)
    v_trans = ops.transpose(v, (0, 2, 1, 3))  # ...khd -> ...hkd
    out_temp = ops.matmul(weights, v_trans)  # ...hqk,...hkd->...hqd
    out = ops.transpose(out_temp, (0, 2, 1, 3))
    return out


def _attend(
        q, k, v, *, q_k_dot_precision, logits_dtype, logits_scale,
        bias, mask, weights_v_dot_precision, q_indices, k_indices,
):
    logits = cal_logits(q, k)

    logits *= logits_scale

    if bias is not None:
        logits += bias

    if mask is not None:
        q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
        k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
        mask = mask.as_array(q_len_or_indices, k_len_or_indices)

    if mask is not None:  # TBD in ms
        mask_value = -3.4028235e+37  # a small value close to min of bfloat16
        logits = ops.where(mask.bool(), logits, mask_value)

    weights = _softmax(logits)

    out = cal_out(weights, v)

    return out


@dataclasses.dataclass(frozen=True)
class MsDotProductAttention(base.DotProductAttention):
    """MS dot product attention function."""

    _: dataclasses.KW_ONLY

    def _fwd(
            self, q, k, v, *, q_k_dot_precision, logits_dtype, logits_scale,
            bias, mask, weights_v_dot_precision, q_indices, k_indices,
    ):

        return _attend(
            q, k, v, bias=bias, mask=mask, q_indices=q_indices, k_indices=k_indices,
            q_k_dot_precision=q_k_dot_precision, logits_dtype=logits_dtype, logits_scale=logits_scale,
            weights_v_dot_precision=weights_v_dot_precision,
        )

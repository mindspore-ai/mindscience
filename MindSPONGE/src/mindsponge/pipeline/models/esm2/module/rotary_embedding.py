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
"""Rotary Embedding"""

from typing import Tuple
import mindspore as ms
from mindspore import ops
from mindspore import nn
import mindspore.numpy as mnp


def rotate_half(x):
    res = ops.split(x, 32, 2)
    x1, x2 = res[0], res[1]
    return ops.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, : x.shape[-2], :]
    sin = sin[:, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Cell):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """
    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.inv_freq = 1.0 / (10000 ** (mnp.arange(0, dim, 2, ms.float32) / dim))
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def construct(self, q: ms.Tensor, k: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """rotary position embeddings"""
        x = k
        seq_dimension = -2
        t = mnp.arange(x.shape[seq_dimension], dtype=ms.float32)
        t = ops.reshape(t, (-1, 1))
        inv_freq = ops.reshape(self.inv_freq, (1, -1))
        freqs = t * inv_freq
        emb = ops.concat((freqs, freqs), axis=-1)
        cos_cached = ops.cos(emb)[None, :, :]
        sin_cached = ops.sin(emb)[None, :, :]
        return (
            apply_rotary_pos_emb(q, cos_cached, sin_cached),
            apply_rotary_pos_emb(k, cos_cached, sin_cached),
        )

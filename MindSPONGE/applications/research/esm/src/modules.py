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
"""Sinusoidal positional embedding"""

import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


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

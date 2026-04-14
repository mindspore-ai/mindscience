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
"""contact prediction head"""
from typing import Optional
import mindspore.nn as nn
import mindspore as ms
from mindspore import ops


def symmetrize(x):
    """Make layer symmetric in final two dimensions, used for contact prediction."""
    return x + x.transpose(0, 1, 3, 2)


def apc(x):
    """Perform average product correct, used for contact prediction."""
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)
    avg = a1 * a2
    avg = avg.div(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


class ContactPredictionHead(nn.Cell):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(self, in_features: int, prepend_bos: bool, append_eos: bool,
                 bias=True, eos_idx: Optional[int] = None):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        if append_eos and eos_idx is None:
            raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
        self.eos_idx = eos_idx
        self.regression = nn.Dense(in_features, 1, has_bias=bias)
        self.activation = nn.Sigmoid()

    def construct(self, tokens, attentions):
        """contact prediction head"""
        # remove eos token attentions
        if self.append_eos:
            eos_mask = tokens.ne(self.eos_idx)
            eos_mask = ops.Cast()(eos_mask, ms.float32)
            eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
            attentions = attentions * eos_mask[:, None, None, :, :]
            attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        if self.prepend_bos:
            attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.shape
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: B x C x T x T
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        attentions = self.regression(attentions).squeeze(3)
        return self.activation(attentions)

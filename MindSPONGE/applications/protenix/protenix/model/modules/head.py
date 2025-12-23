# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Distogram head."""

from typing import Final
from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops, mint
from protenix.model import base_config
from protenix.model.modules import base_modules as bm
from mindscience.e3nn.utils import Ncon


_CONTACT_THRESHOLD: Final[float] = 8.0
_CONTACT_EPSILON: Final[float] = 1e-3


def hook_fn(grad):
    print('distogram')
    print(grad)


hook = ops.HookBackward(hook_fn)


class DistogramHead(nn.Cell):
    """
    A DistogramHead class that computes a distogram from pair embeddings, predicting distances between residues.

    Args:
        config (Config): Configuration object containing parameters for the distogram head.
        global_config (GlobalConfig): Global configuration object.
        in_channel (int): Number of input channels for the linear layer.

    Inputs:
        - **batch** (dict) - Dictionary containing batch features.
        - **embeddings** (dict) - Dictionary containing pair embeddings.

    Outputs:
        - **bin_edges** (Tensor) - Tensor of bin edges for distance predictions.
        - **contact_probs** (Tensor) - Tensor of contact probabilities.

    Notes:
        - The distogram head computes distance probabilities using a linear transformation and softmax.
        - The Ncon class is used for tensor contraction operations.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        first_break: float = 2.3125
        last_break: float = 21.6875
        num_bins: int = 64

    def __init__(
        self, config, global_config, in_channel, dtype=ms.float32
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.linear = bm.CustomDense(
            in_channel, self.config.num_bins, weight_init=self.global_config.final_init, ndim=3, dtype=dtype)
        self.ncon = Ncon([[-1, -2, 1], [1]])

    def construct(self, pair_act):
        """construct"""
        left_half_logits = self.linear(pair_act)
        right_half_logits = left_half_logits

        logits = left_half_logits + mint.swapaxes(right_half_logits, -2, -3)
        probs = mint.softmax(logits, dim=-1)
        breaks = mint.linspace(
            self.config.first_break,
            self.config.last_break,
            self.config.num_bins - 1,
        )
        bin_tops = mint.concat(
            (breaks, (breaks[-1] + breaks[-1] - breaks[-2]).reshape(-1)))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)
        contact_probs = self.ncon(
            [probs.astype(ms.float32), is_contact_bin.astype(ms.float32)])

        return breaks, contact_probs, logits

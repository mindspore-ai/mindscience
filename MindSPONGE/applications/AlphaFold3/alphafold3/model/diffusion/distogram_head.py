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

"""Distogram head."""

from typing import Final
from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops
from alphafold3.model import base_config
from alphafold3.model.components import base_modules as bm
from mindscience.e3nn.utils import Ncon


_CONTACT_THRESHOLD: Final[float] = 8.0
_CONTACT_EPSILON: Final[float] = 1e-3


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
        """Initialize DistogramHead."""
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.linear = bm.CustomDense(
            in_channel, self.config.num_bins, weight_init=self.global_config.final_init, ndim=3, dtype=dtype)
        self.ncon = Ncon([[-1, -2, 1], [1]])

    def construct(self, batch, embeddings):
        """Compute distogram predictions."""
        pair_act = embeddings["pair"]
        seq_mask = batch.token_features.mask.astype(ms.bool_)
        pair_mask = seq_mask[:, None] * seq_mask[None, :]
        left_half_logits = self.linear(pair_act)
        right_half_logits = left_half_logits
        logits = left_half_logits + ops.swapaxes(right_half_logits, -2, -3)
        probs = ops.softmax(logits, axis=-1)
        breaks = ops.linspace(
            self.config.first_break,
            self.config.last_break,
            self.config.num_bins - 1,
        )
        bin_tops = ops.concat(
            (breaks, (breaks[-1] + breaks[-1] - breaks[-2]).reshape(-1)))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)
        contact_probs = self.ncon([probs.astype(ms.float32), is_contact_bin.astype(ms.float32)])
        contact_probs = pair_mask * contact_probs
        return {
            'bin_edges': breaks,
            'contact_probs': contact_probs,
        }

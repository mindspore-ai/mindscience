# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""modules for the Diffuser model."""

from dataclasses import dataclass
from typing import List

import mindspore as ms
from mindspore import nn, ops, mint
from protenix.model import base_config
from protenix.model.modules.module_utils.attention import attention
from protenix.model.modules import base_modules as bm
from protenix.openfold_local.utils.chunk_utils import chunk_layer


def get_shard_size(num_residues, shard_spec):
    shard_size = shard_spec[0][-1]
    for num_residues_upper_bound, num_residues_shard_size in shard_spec:
        shard_size = num_residues_shard_size
        if (
            num_residues_upper_bound is None
            or num_residues <= num_residues_upper_bound
        ):
            break
    return shard_size


class GridSelfAttention(nn.Cell):
    """
    Self-attention mechanism that operates either per-sequence or per-residue.

    Args:
        config (Config): Configuration object containing parameters for the attention mechanism.
        global_config (GlobalConfig): Global configuration object.
        transpose (bool): Whether to transpose the activation tensor during processing.
        normalized_shape (tuple): Shape of the input tensor for normalization.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **pair_mask** (Tensor) - Mask tensor indicating valid regions in the input.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the self-attention mechanism.
    """

    @dataclass
    class Config(base_config.BaseConfig):
        num_head: int = 4
        key_dim: int = 32

    def __init__(
            self, config, global_config, transpose, normalized_shape, dtype=ms.float32
    ):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.transpose = transpose
        num_channels = normalized_shape[-1]
        in_shape = normalized_shape[-1]
        if num_channels % self.config.num_head != 0:
            raise ValueError("num_channels must be divisible by num_head")
        self.qkv_dim = max(num_channels // self.config.num_head, 16)
        qkv_shape = self.config.num_head * self.qkv_dim
        self.q_projection = bm.CustomDense(
            in_shape, qkv_shape, use_bias=False, ndim=3, dtype=dtype)
        self.k_projection = bm.CustomDense(
            in_shape, qkv_shape, use_bias=False, ndim=3, dtype=dtype)
        self.v_projection = bm.CustomDense(
            in_shape, qkv_shape, use_bias=False, ndim=3, dtype=dtype)
        self.gating_query = bm.CustomDense(num_channels, self.config.num_head * self.qkv_dim, weight_init='zeros',
                                           use_bias=False, ndim=3, dtype=dtype)
        self.output_projection = bm.CustomDense(self.config.num_head * self.qkv_dim, num_channels,
                                                weight_init=self.global_config.final_init, ndim=3, dtype=dtype)
        self.act_norm = bm.LayerNorm(normalized_shape, dtype=ms.float32)
        self.pair_bias_projection = bm.CustomDense(num_channels, self.config.num_head, use_bias=False,
                                                   weight_init='he_normal', ndim=3,
                                                   dtype=dtype)  # linear, change to ones for test
        num_residues = normalized_shape[0]
        self.chunk_size = get_shard_size(
            num_residues, self.global_config.pair_attention_chunk_size
        )

    def _attention(self, act, mask, bias):
        """triangle attention"""
        q = self.q_projection(act)
        k = self.k_projection(act)
        v = self.v_projection(act)
        q = q.view(q.shape[:-1] + (self.config.num_head, -1))
        k = k.view(k.shape[:-1] + (self.config.num_head, -1))
        v = v.view(v.shape[:-1] + (self.config.num_head, -1))

        weighted_avg = attention(
            q,
            k,
            v,
            logits_scale=self.config.key_dim ** (-0.5),
            mask=mask,
            bias=bias,
        )

        gate_value = self.gating_query(act)
        weighted_avg = weighted_avg * mint.sigmoid(gate_value.astype(ms.float32)).astype(
            gate_value.dtype).view(gate_value.shape[:-1] + (self.config.num_head, -1))
        weighted_avg = weighted_avg.reshape(weighted_avg.shape[:-2] + (-1,))
        return self.output_projection(weighted_avg)

    def construct(self, act, pair_mask):
        """Builds a module.

        Arguments:
            act: [num_seq, num_res, channels] activations tensor
            pair_mask: [num_seq, num_res] mask of non-padded regions in the tensor.
                Only used in inducing points attention currently.

        Returns:
            Result of the self-attention operation.
        """
        if pair_mask is None:
            # [*, I, J]
            pair_mask = act.new_ones(
                act.shape[:-1],
            )
        if self.transpose:
            act = mint.swapaxes(act, -2, -3)
            pair_mask = mint.swapaxes(pair_mask, -1, -2)
        act = self.act_norm(act)

        non_batched_bias = self.pair_bias_projection(act)
        non_batched_bias = ops.transpose(non_batched_bias, (2, 0, 1))
        pair_mask = pair_mask[:, None, None, :].astype(ms.bool_)
        chunk_size = 128
        if chunk_size is None:
            act = self._attention(
                act, pair_mask, non_batched_bias.unsqueeze(0))
        else:
            act = self._chunk(act, pair_mask, non_batched_bias.unsqueeze(
                0), chunk_size)
        if self.transpose:
            act = mint.swapaxes(act, -2, -3)
        return act

    def _chunk(
        self,
        x: ms.Tensor,
        mask: ms.Tensor,
        biases: List[ms.Tensor],
        chunk_size: int,
        inplace_safe: bool = False,
    ) -> ms.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "act": x,
            "mask": mask,
            "bias": biases,
        }

        return chunk_layer(
            self._attention,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            out=x if inplace_safe else None,
        )

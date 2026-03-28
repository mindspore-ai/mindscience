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

import mindspore as ms
from mindspore import nn, Tensor, Parameter, mint, ops
from mindspore.mint.nn import functional as F

from vortex.model.utils import grab_first_if_tuple

class RMSNorm(nn.Cell):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.eps, self.hidden_size = config.eps, config.hidden_size
        self.scale = Parameter(mint.ones(self.hidden_size, dtype=config.params_dtype))
        self.use_flash_rmsnorm = config.get("use_flash_rmsnorm", False)

        if self.use_flash_rmsnorm:
            from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

            self.rmsnorm_func = rmsnorm_func

    def forward(self, x, layer_idx="100", extend=""):
        if self.use_flash_rmsnorm:
            return self.rmsnorm_func(x, self.scale, self.eps)
        else:
            norm = mint.linalg.norm(x, ord=2, dim=-1, keepdim=True)
            y = x / (norm * self.hidden_size ** (-1.0 / 2) + self.eps)
            out = (self.scale * y).astype(ms.bfloat16)
            return out


class ParallelGatedMLP(nn.Cell):
    def __init__(
        self,
        config,
        layer_idx,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act_type = config.get("mlp_activation", "gelu")
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError
        
        if self.layer_idx > 0 and config.get("evo2_style_activations", False):
            self.act = nn.Identity()

        self.multiple_of = multiple_of * config.model_parallel_size

        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        inner_size = config.get("inner_mlp_size", inner_size)

        self.l1 = nn.Dense(
            in_channels=config.hidden_size,
            out_channels=inner_size,
            has_bias=False
        )
        self.l2 = nn.Dense(
            in_channels=config.hidden_size,
            out_channels=inner_size,
            has_bias=False
        )
        self.l3 = nn.Dense(
            in_channels=inner_size,
            out_channels=config.hidden_size,
            has_bias=False
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        z1, z2 = grab_first_if_tuple(z1), grab_first_if_tuple(z2)
        y = self.l3(self.act(z1) * z2)
        return grab_first_if_tuple(y)


class VocabParallelEmbedding(nn.Embedding):
    "Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/embedding.py"

    def __init__(self, config, dtype):
        vocab_size, process_group, padding_idx = (
            config.vocab_size,
            config.get("process_group", None),
            config.get("padding_idx", None),
        )
        self.process_group = process_group
        if process_group is not None:
            world_size = mint.distributed.get_world_size(process_group)
            if vocab_size % world_size != 0:
                raise ValueError(f"vocab_size ({vocab_size}) must be divisible by " f"world_size ({world_size})")
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(
            vocab_size // world_size,
            embedding_size=config.hidden_size,
            padding_idx=padding_idx,
            dtype=dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.process_group is None:  # single process
            return super().construct(input).astype(ms.bfloat16)
        else:
            rank = mint.distributed.get_rank(self.process_group)
            vocab_size = self.num_embeddings
            vocab_start_index, vocab_end_index = (
                rank * vocab_size,
                (rank + 1) * vocab_size,
            )
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (input < vocab_start_index) | (input >= vocab_end_index)
            input = input - vocab_start_index
            input[input_ids_mask] = 0
            embeddings = self.forward(input)
            embeddings[input_ids_mask] = 0.0
            # Reduce to the global process group
            mint.distributed.all_reduce(embeddings, group=self.process_group)
            return embeddings

    def unembed(self, u: Tensor) -> Tensor:
        if self.process_group is None:
            return ops.matmul(u, self.embedding_table.T)
        else:
            raise NotImplementedError


class VocabParallelUnembedding(VocabParallelEmbedding):
    def forward(self, input: Tensor) -> Tensor:
        return self.unembed(input)

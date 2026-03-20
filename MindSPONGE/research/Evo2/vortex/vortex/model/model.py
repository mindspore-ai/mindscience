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

import math
import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter, mint
from mindspore import numpy as mnp
import mindspore.mint.nn.functional as F
from mindspore.device_context.ascend import is_available

from vortex.model.cache import (
    InferenceParams,
    HyenaCascadeFIRInferenceParams,
    HyenaCascadeIIRInferenceParams,
)
from vortex.model.engine import HyenaInferenceEngine
from vortex.model.layers import (
    ParallelGatedMLP,
    RMSNorm,
    VocabParallelEmbedding,
    VocabParallelUnembedding,
)
from vortex.model.utils import (
    Lambda,
    column_split,
    interleave,
    print_rank_0,
)

import logging
from tqdm import tqdm
from typing import Optional, Union

from vortex.model.attention import MHA

class RotaryEmbeddingMock(nn.Cell):
    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        super().__init__()
        self.base = float(base)
        self.dim = dim
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (mint.arange(0, self.dim, 2, dtype=ms.float32) / self.dim))
    
    def forward(
        self,
        qkv: Tensor,
        kv: Optional[Tensor] = None,
        seqlen_offset: Union[int, Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ):
        return qkv

class MHAMock(nn.Cell):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        cross_attn=False,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_alibi=False,
        window_size=(-1, -1),
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.rotary_emb_dim = rotary_emb_dim
        self.rotary_emb = RotaryEmbeddingMock(
            dim=rotary_emb_dim,
            base=rotary_emb_base,
            scale_base=rotary_emb_scale_base,
            interleaved=rotary_emb_interleaved,
            device=device,
        )

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        return x

class AttentionBlock(nn.Cell):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.pre_norm, self.post_norm = RMSNorm(config), RMSNorm(config)
        self.layer_idx = layer_idx
        self.print_activations = config.get("print_activations", False)
        self.proj_groups = config.get("proj_groups", 1)
        dtype = config.get("attn_block_dtype", ms.bfloat16)
        mlp_dtype = config.get("mlp_dtype", ms.bfloat16)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

        self.counter = 0
        self.inner_mha_cls = MHAMock(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_attention_heads // self.proj_groups,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            qkv_proj_bias=config.get("qkv_proj_bias", True),
            rotary_emb_base=config.get("rotary_emb_base", 1000000),
            causal=True,
            layer_idx=layer_idx,
            out_proj_bias=config.get("mha_out_proj_bias", True),
            use_flash_attn=self.config.use_flash_attn,
            dtype=dtype
        ).to_float(dtype)

        # check if using interpolated rotary pos emb from config, and swap the rope emb
        if config.get("use_interpolated_rotary_pos_emb", False):
            from vortex.model.positional_embeddings import swap_mha_rope
            
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={"scaling_factor": config.get("rotary_emb_scaling_factor", 1.0)},
            )

        if self.config.get("smeared_gqa", False):
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
        
        if self.inner_mha_cls.rotary_emb_dim > 0:
            self.inner_mha_cls.rotary_emb.register_buffer("inv_freq", self.inner_mha_cls.rotary_emb.inv_freq)

        self.mlp = ParallelGatedMLP(config, layer_idx).to_float(mlp_dtype)

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if (
            type(padding_mask) == Tensor
        ):  # workaround for masking bug in FA. This works because Wqkv does not have bias
            # and attention scores will be also automatically zeroed.
            u = u * padding_mask[..., None]
        u = (
            self.inner_mha_cls.forward(
                self.pre_norm.forward(u),
                inference_params=inference_params,
            )
            + u
        )

        if type(padding_mask) == Tensor:  # guard against bias
            u = u * padding_mask[..., None]

        u = self.mlp.forward(self.post_norm.forward(u)) + u
        return u, None


class HyenaCascade(nn.Cell):
    def __init__(self, config, layer_idx, hyena_filter_groups=None, fir_inner_filter_length=None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hyena_filter_groups = hyena_filter_groups
        self.print_activations = config.get("print_activations", False)
        self.ground_truth_activations_path = config.get("ground_truth_activations_path", None)

        self.use_flashfft = config.get("use_flashfft", False)
        self.state_size = config.state_size
        self.hidden_size = config.hidden_size
        self.num_filters = config.num_filters
        self.inference_mode = config.get("inference_mode", True)
        self.counter = 0
        self.column_split_hyena = config.get("column_split_hyena", True)
        self.hyena_flip_x1x2 = config.get("hyena_flip_x1x2", False)

        assert self.hidden_size % self.num_filters == 0 and self.num_filters <= self.hidden_size

        # attention heads are not used except to split post short_filter
        # projections in the same way as the checkpoint
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        self.fir_inner_filter_length = fir_inner_filter_length
        self.short_filter_length = config.short_filter_length
        self.short_filter_weight = Parameter(mnp.randn(3 * config.hidden_size, 1, config.short_filter_length))
        self.short_filter_bias = Parameter(mnp.randn(3 * config.hidden_size)) if config.short_filter_bias else None

        self.engine = HyenaInferenceEngine(
            layer_idx=layer_idx,
            ground_truth_activations_path=self.ground_truth_activations_path,
            print_activations=self.print_activations,
            hyena_flip_x1x2=config.get("hyena_flip_x1x2", False),
        )
        self.use_flash_depthwise = config.get("use_flash_depthwise", False)
        self.data_dtype = None

        if self.use_flash_depthwise:
            try:
                from flashfftconv import FlashDepthwiseConv1d

                self.fir_fn = FlashDepthwiseConv1d(
                    channels=3 * self.hidden_size,
                    kernel_size=self.short_filter_length,
                    padding=self.short_filter_length - 1,
                    weights=self.short_filter_weight,
                    bias=self.short_filter_bias,
                    device=None,
                    dtype=self.config.get("depthwise_dtype", ms.bfloat16),
                )
            except ImportError:
                "flashfftconv not installed"
        else:
            self.fir_fn = F.conv1d

            self.fir_inner_fn = F.conv1d

        self.fftconv_fn = None
        self.long_fir_threshold = config.get("long_fir_threshold", None)
        if self.long_fir_threshold is not None:
            assert self.use_flashfft is False, "long_fir_threshold not compatible with fused flashfft"

        self.num_systems = self.hyena_filter_groups
        self.channels_per_group = self.hidden_size // self.hyena_filter_groups

        if self.fir_inner_filter_length:
            self.h = Parameter(mnp.randn(self.hyena_filter_groups, 1, fir_inner_filter_length))

            if fir_inner_filter_length >= 128:
                self.D = Parameter(mint.zeros(self.hidden_size))

            if fir_inner_filter_length < 128:
                self.D = None

        else:
            self.log_poles = Parameter(mnp.randn(self.num_systems, self.state_size, 1, dtype=ms.float32))
            self.residues = Parameter(mnp.randn(self.num_systems, self.state_size, dtype=ms.float32))
            self.D = Parameter(mint.zeros(self.hidden_size))
            self.h = None
        self.t = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
            return self.sequential_forward(u, inference_params)

        else:
            return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )

        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn,
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            L,
            dims=dims,
            gate=False,
            column_split_hyena=self.column_split_hyena,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask,
            dim_last=True,
        )

        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.config.interleave:
            z_pre = interleave(z_pre)

        if self.h is None:
            h, _, _, _ = self.compute_filter(L, u.device)
        else:
            h = self.h

        D = self.D

        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)

        # if inference_params is not None, we plan to perform generation:
        # prefilling is handled by the engine.
        if self.fir_inner_filter_length is not None:
            y, fir_inner_state = self.engine.parallel_fir(
                self.fir_inner_fn,
                z_pre,
                h,
                D,
                L,
                dims=dims,
                gate=True,
                gated_bias=self.fir_inner_filter_length >= 128,
                dim_last=False,
                column_split_hyena=self.column_split_hyena,
                fir_length=self.fir_inner_filter_length,
                inference_params=inference_params,
                padding_mask=padding_mask,
                groups=self.hyena_filter_groups,
            )
            y = y.permute(0, 2, 1)
            if inference_params:
                inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            y = self.engine.parallel_iir(
                z_pre,
                h,
                D,
                L,
                t=self.t,
                poles=self.log_poles,
                residues=self.residues,
                dims=dims,
                inference_params=inference_params,
                layer_idx=self.layer_idx,
                prefill_style=self.config.get("prefill_style", "fft"),
                use_flashfft=self.use_flashfft,
                fftconv_fn=self.fftconv_fn,
                column_split_hyena=self.column_split_hyena,
                long_fir_threshold=self.long_fir_threshold,
                padding_mask=padding_mask,
            )

        return y, inference_params

    def sequential_forward(self, u, inference_params):
        if self.data_dtype is None:
            self.data_dtype = u.dtype

        if len(u.shape) > 2:
            u = u[:, -1]

        z_pre, fir_state = self.engine.step_fir(
            u,
            inference_params.fir_state_dict[self.layer_idx],
            weight=self.short_filter_weight,
            bias=self.short_filter_bias,
        )
        inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.config.interleave:
            z_pre = interleave(z_pre)

        x2, x1, v = (
            column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
            if self.column_split_hyena
            else z_pre.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)
        )

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        if self.fir_inner_filter_length is not None:
            if self.hyena_filter_groups > 1:
                h = self.h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)
            else:
                h = self.h

            y, fir_inner_state = self.engine.step_fir(
                x1 * v,
                inference_params.fir_inner_state_dict[self.layer_idx],
                weight=h,
                bias=self.D,
                flip_filter=self.fir_inner_filter_length >= 128,
                gated_bias=self.fir_inner_filter_length >= 128,
            )
            y = y * x2
            inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            y, iir_state = self.engine.step_iir(
                x2,
                x1,
                v,
                self.D,
                self.residues,
                self.log_poles,
                inference_params.state_dict[self.layer_idx],
                iir_groups=1,
            )
            inference_params.state_dict[self.layer_idx] = iir_state

        y = y.to(self.data_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device):
        """
        Set [0, 1, ..., L-1] where L is the length of the current batch of inputs.
        If L is greater than the length of the previous batch, then the time vector is
        reinitialized. Otherwise, the time vector is truncated from cache.
        """
        if self.t is None:
            self.t = mint.arange(L)[None, None]
        elif self.t.shape[-1] < L:
            self.t = mint.arange(L)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        filter_dtype = ms.float32
        residues, log_poles = (
            Parameter(self.residues.astype(filter_dtype)),
            Parameter(self.log_poles.astype(filter_dtype)),
        )
        h = (residues[..., None] * (log_poles * self.t).exp()).sum(1)[None]  # B, D, L
        return h, filter_dtype, log_poles, residues


class ParallelGatedConvBlock(nn.Cell):
    def __init__(self, config, layer_idx, hyena_filter_groups=None, fir_inner_filter_length=None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.print_activations = config.get("print_activations", False)
        self.ground_truth_activations_path = config.get("ground_truth_activations_path", None)
        self.low_mem_mode = config.get("low_mem_mode", False)
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = hyena_filter_groups if hyena_filter_groups is not None else config.hidden_size
        dtype = config.get("hyena_block_dtype", ms.bfloat16)
        mlp_dtype = config.get("mlp_dtype", ms.bfloat16)
        self.pre_norm, self.post_norm = (
            RMSNorm(config).to_float(dtype),
            RMSNorm(config).to_float(dtype),
        )
        self.filter = HyenaCascade(
            config,
            layer_idx,
            hyena_filter_groups=self.hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
        ).to_float(dtype)

        # For posterity/debugging: TELinear can be easily replaced by
        # nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.qkv_proj_bias).to(dtype=dtype)
        # which sometimes is very useful when debugging FP8.
        self.projections = nn.Dense(
            config.hidden_size,
            3 * config.hidden_size,
            has_bias=config.qkv_proj_bias,
        ).to_float(dtype)

        self.out_filter_dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.hyena_out_proj_bias
        ).to_float(dtype)
        self.mlp = ParallelGatedMLP(config, layer_idx).to_float(mlp_dtype)

    def pad_to_multiple(self, x, multiple=16):
        """Pad input tensor to multiple of 16 only when FP8 is enabled"""
        if not self.config.get("use_fp8_input_projections", False):
            return x

        batch_size, seq_len, hidden_dim = ops.shape(x)
        print("--- batch_size, seq_len, hidden_dim: ", batch_size, seq_len, hidden_dim)
        pad_len = (multiple - (seq_len % multiple)) % multiple
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))

    def proj_norm(self, x):
        normalized = self.pre_norm.forward(x)
        normalized = self.pad_to_multiple(normalized)
        projected = self.projections(normalized)

        if isinstance(projected, tuple):
            projected = projected[0]

        original_seq_len = ops.shape(x)[1]
        # Slice back to original sequence length if padding was added
        if ops.shape(projected)[1] > original_seq_len:
            projected = projected[:, :original_seq_len, :]

        return projected

    def res_mlp_norm(self, x):
        return self.mlp.forward(self.post_norm.forward(x)) + x

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        z = self.proj_norm(u)

        if type(padding_mask) == Tensor:  # guard against bias
            z = z * padding_mask[..., None]

        z, inference_params = self.filter.forward(z, inference_params=inference_params, padding_mask=padding_mask)

        z_in = self.out_filter_dense(z) + u

        # if self.layer_idx == 0:
        #    z_in = z_savanna.squeeze() + u + self.out_filter_dense.bias

        if type(padding_mask) == Tensor:  # guard against bias
            z_in = z_in * padding_mask[..., None]

        y = self.res_mlp_norm(z_in)

        return y, inference_params


def get_block(config, layer_idx, flash_fft=None):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    elif layer_idx in config.hcl_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx)
        if config.get("use_flashfft", "False"):
            block.filter.fftconv_fn = flash_fft
        return block
    elif layer_idx in config.hcm_layer_idxs:
        block = ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcm_filter_groups,
            fir_inner_filter_length=config.hcm_filter_length,
        )
        return block
    elif layer_idx in config.hcs_layer_idxs:
        block = ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcs_filter_groups,
            fir_inner_filter_length=config.hcs_filter_length,
        )
        return block
    else:
        raise NotImplementedError


class StripedHyena(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.print_activations = config.get("print_activations", False)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.ground_truth_activations_path = config.get("ground_truth_activations_path", None)
        self.logger.info(f"Initializing StripedHyena with config: {config}")

        self.embedding_layer = VocabParallelEmbedding(config)  # under single process, forward nn.Embedding

        if config.get("use_flashfft", "True"):
            try:
                from flashfftconv import FlashFFTConv

                self.flash_fft = FlashFFTConv(config.seqlen, dtype=ms.bfloat16)
            except ImportError:
                "flashfftconv not installed"
        else:
            self.flash_fft = None
        if not self.config.get('evo2_style_activations', False):
            self.logger.warning(
                "⚠️  Not using Evo2 style activations  ⚠️\n"
                "⚠️ Set 'evo2_style_activations: True' in config if you are using Evo 2 checkpoints ⚠️"
            )
        self.logger.info(f"Initializing {config.num_layers} blocks...")
        self.blocks = nn.CellList()
        self.block_idx_to_device = {}

        # Calculate layers per GPU
        # num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # num_gpus = device_count() if is_available() else 1
        num_gpus = 1  # LYY note: temporarily single card
        layers_per_gpu = math.ceil(config.num_layers / num_gpus)  # Model parallelism
        self.logger.info(f"Distributing across {num_gpus} GPUs, approximately {layers_per_gpu} layers per GPU")

        for layer_idx in tqdm(range(config.num_layers)):
            # Determine which GPU should handle this layer
            device_idx = min(layer_idx // layers_per_gpu, num_gpus - 1)
            device = f"npu:{device_idx}" if is_available() else "cpu"

            block = get_block(config, layer_idx, flash_fft=self.flash_fft)

            self.blocks.append(block)
            self.block_idx_to_device[layer_idx] = device
            self.logger.info(f"Assigned {layer_idx=} to {device=}")
            self.logger.info(
                f"block parameters type: {type(self.blocks[-1].get_parameters())}"
            )

        self.norm = RMSNorm(config) if config.get("final_norm", True) else None
        if config.tie_embeddings:
            # Lambda usage is to be able to use forward() on caller side, which in
            # turn is needed for PyTorch hooks to work properly.
            self.unembed = Lambda(self.embedding_layer.unembed)
        else:
            if config.tie_embeddings:
                # Technically we can support this mode, just need to
                # copy tensors across GPUs then. But let's implement it
                # once/if needed.
                self.logger.info("Ignoring tie_embeddings for now.")
            self.unembed = VocabParallelUnembedding(config)

        self.logger.info("Initialized model")

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        x = self.embedding_layer(x)

        if inference_params_dict is not None:
            print("--- stateful_forward")
            x, inference_params_dict_out = self.stateful_forward(
                x,
                inference_params_dict=inference_params_dict,
            )
        else:
            print("--- stateless_forward")
            x, inference_params_dict_out = self.stateless_forward(x, padding_mask=padding_mask)

        x = self.norm(x)

        x = self.unembed(x)
        return x, inference_params_dict_out

    def block_idx_to_name(self, block_idx):
        if block_idx in self.config.attn_layer_idxs:
            return "mha"
        elif block_idx in self.config.hcl_layer_idxs:
            return "hcl"
        elif block_idx in self.config.hcm_layer_idxs:
            return "hcm"
        elif block_idx in self.config.hcs_layer_idxs:
            return "hcs"
        else:
            raise ValueError(f"Block index {block_idx} not found")

    def cross_device_transfer(self, x, block_idx):
        if self.block_idx_to_device[max(block_idx - 1, 0)] != self.block_idx_to_device[block_idx]:
            x = x.to(self.block_idx_to_device[block_idx])
        return x

    def stateful_forward(self, x, inference_params_dict=None):
        for block_idx, block in enumerate(self.blocks):
            inference_params = inference_params_dict[self.block_idx_to_name(block_idx)]

            x = self.cross_device_transfer(x, block_idx)
            x, _ = block.forward(x, inference_params=inference_params)

        return x, inference_params_dict

    def stateless_forward(self, x, padding_mask=None):
        if type(padding_mask) == Tensor:
            x = x * padding_mask[..., None]

        for block_idx, block in enumerate(self.blocks):
            x = self.cross_device_transfer(x, block_idx)
            x, _ = block.forward(x, inference_params=None, padding_mask=padding_mask)

        return x, None

    def initialize_inference_params(self, max_seqlen=None):
        ## Input seqlen takes priority over config!
        ## WARNING: This avoids potential errors but means the model can be used beyond length it was trained at
        config_seqlen = self.config.get("max_seqlen", None)
        if config_seqlen is None:
            print("No max_seqlen found in config!!! using default value of 8192")
            config_seqlen = 8192
        new_max_seqlen = max_seqlen if max_seqlen != None else config_seqlen
        # self.config["max_seqlen"] = new_max_seqlen
        ## Note: changing the stored config max_seqlen will change the max_seqlen used in flash attention, leading to minor logit differences
        print(f"Initializing inference params with max_seqlen={new_max_seqlen}")

        inference_params_dict = {
            "mha": InferenceParams(
                max_seqlen=new_max_seqlen,
                max_batch_size=self.config.get("max_batch_size", 1),
                seqlen_offset=0,
            ),
            "hcl": HyenaCascadeIIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                state_dim=self.config.state_size,
                seqlen_offset=0,
            ),
            "hcm": HyenaCascadeFIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                fir_inner_filter_length=self.config.hcm_filter_length,
                seqlen_offset=0,
            ),
            "hcs": HyenaCascadeFIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                fir_inner_filter_length=self.config.hcs_filter_length,
                seqlen_offset=0,
            ),
        }
        return inference_params_dict

    def precompute_filters(self, L, device):
        for _, block in enumerate(self.blocks):
            if type(block) == ParallelGatedConvBlock:
                if type(block.filter) == HyenaCascade:
                    L = block.filter.long_fir_threshold or L
                    print_rank_0(f"Precomputing filters, L={L}...")

                    filter_dtype = ms.float16 if L >= 2048 else ms.float32

                    block.filter._set_time(L, device)
                    residues, poles = (
                        block.filter.residues.to_float(ms.float16),
                        block.filter.poles.to_float(ms.float16),
                    )

                    block.filter.h = (residues * poles**block.filter.t).real.sum(1)[None]
                    block.filter.h = block.filter.h.to_float(filter_dtype)

    def custom_load_state_dict(self, state_dict, strict=True):
        """
        Post-processes the state_dict to convert savanna checkpoints to vortex checkpoints.
        """
        self.logger.debug(f"Loading state dict: {state_dict}, (ignoring extra keys) with strict: {strict}")
        model_dict = self.state_dict()

        # Find keys that are in model_dict but not in state_dict
        missing_in_state_dict = model_dict.keys() - state_dict.keys()
        # Find keys that are in state_dict but not in model_dict
        extra_in_state_dict = state_dict.keys() - model_dict.keys()

        if missing_in_state_dict:
            print(f"Keys missing in state_dict: {missing_in_state_dict}")
        if extra_in_state_dict:
            print(f"Extra keys in state_dict: {extra_in_state_dict}")

        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        # Iterate over filtered_dict to ensure _extra_state for TE Linear layers have 'recipe'
        for k in list(filtered_dict.keys()): # Iterate over a copy of keys as we might modify the dict
            if k.endswith('._extra_state'):
                module_path = k.rsplit('._extra_state', 1)[0]
                try:
                    current_module = self
                    for attr in module_path.split('.'):
                        current_module = getattr(current_module, attr)

                except AttributeError:
                    self.logger.debug(f"Could not find module for {k}, skipping recipe injection.")
                except Exception as e:
                    self.logger.warning(f"Error processing _extra_state for {k}: {e}")
        
        # Handle _extra_state keys that are entirely missing from the loaded state_dict
        for k in missing_in_state_dict:
            if k.endswith('._extra_state'):
                module_path = k.rsplit('._extra_state', 1)[0]
                try:
                    current_module = self
                    for attr in module_path.split('.'):
                        current_module = getattr(current_module, attr)
                except AttributeError:
                    self.logger.debug(f"Module for missing key {k} not found. Skipping.")
                except Exception as e:
                    self.logger.warning(f"Error creating missing _extra_state for {k}: {e}")

        self.load_state_dict(filtered_dict, strict=strict)

        if self.config.get("column_split", True):
            self.logger.info("Adjusting Wqkv for column split (permuting rows)")
            for layer_idx, block in enumerate(self.blocks):
                if type(block) == AttentionBlock:
                    target_device = block.inner_mha_cls.Wqkv.weight.device

                    Wqkv = state_dict[f"blocks.{layer_idx}.inner_mha_cls.Wqkv.weight"]
                    try:
                        bias = state_dict[f"blocks.{layer_idx}.inner_mha_cls.Wqkv.bias"]
                    except:
                        bias = None

                    size_att_head = block.hidden_size_per_attention_head

                    Wqkv = Wqkv.permute(1, 0)
                    Wqkv = Wqkv.reshape(block.hidden_size, block.num_attention_heads, 3, size_att_head)
                    Wq, Wk, Wv = Wqkv.unbind(dim=-2)
                    Wq = Wq.reshape(block.hidden_size, -1)
                    Wk = Wk.reshape(block.hidden_size, -1)
                    Wv = Wv.reshape(block.hidden_size, -1)
                    Wqkv = mint.cat([Wq, Wk, Wv], dim=-1)
                    Wqkv = Wqkv.permute(1, 0)

                    # Single device transfer at the end
                    block.inner_mha_cls.Wqkv.weight.data = Wqkv.to(target_device)

                    if bias is not None:
                        bias = bias.cpu()  # Process on CPU
                        bias = bias.reshape(block.num_attention_heads, 3, size_att_head)
                        bias_q, bias_k, bias_v = bias.unbind(dim=-2)
                        bias_q = bias_q.reshape(block.hidden_size)
                        bias_k = bias_k.reshape(block.hidden_size)
                        bias_v = bias_v.reshape(block.hidden_size)
                        bias = mint.cat([bias_q, bias_k, bias_v], dim=0)
                        try:
                            block.inner_mha_cls.Wqkv.bias.data = bias.to(target_device)
                        except:
                            pass

    def to_bfloat16_except_pr_lc(self, to_float32=False):
        """Convert all parameters to bfloat16 except for the poles and residues.

        Particularly important for longer prompts.
        """
        excluded_shapes = [(4096, 1, 128)]
        for k, p in self.named_parameters():
            if "projections" not in k:  # avoid TE linears
                if "log_poles" not in k and "residues" not in k and p.shape not in excluded_shapes:
                    p.data = p.data.to_float(ms.bfloat16)
                else:
                    if to_float32:
                        p.data = p.data.to_float(ms.float32)
        for k, b in self.named_buffers():
            if "inv_freq" in k:
                if to_float32:
                    b.data = b.data.to_float(ms.float32)

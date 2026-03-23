# Copyright (c) 2024, Michael Poli.

import io # For BytesIO handling
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Union

from .layers import (
    ParallelGatedMLP,
    RMSNorm,
)
from vortex.logging import activations_logger

try:
    from vortex.model.positional_embeddings import swap_mha_rope
except ImportError:
    "could not import swap_mha_rope from src.positional_embeddings"

class RotaryEmbeddingMock(nn.Module):
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
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
    
    def forward(
        self,
        qkv: Tensor,
        kv: Optional[Tensor] = None,
        seqlen_offset: Union[int, Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ):
        return qkv

class MHAMock(nn.Module):
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



class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.pre_norm, self.post_norm = RMSNorm(config), RMSNorm(config)
        self.layer_idx = layer_idx
        self.print_activations = config.get("print_activations", False)
        self.proj_groups = config.get("proj_groups", 1)
        dtype = config.get("attn_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
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
        ).to(dtype=dtype)

        # check if using interpolated rotary pos emb from config, and swap the rope emb
        if config.get("use_interpolated_rotary_pos_emb", False):
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={"scaling_factor": config.get("rotary_emb_scaling_factor", 1.0)},
            )

        if self.config.get("smeared_gqa", False):
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
        self.inner_mha_cls.rotary_emb.register_buffer("inv_freq", self.inner_mha_cls.rotary_emb.inv_freq)

        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if (
            type(padding_mask) == torch.Tensor
        ):  # workaround for masking bug in FA. This works because Wqkv does not have bias
            # and attention scores will be also automatically zeroed.
            u = u * padding_mask[..., None]

        if self.print_activations:
            activations_logger.info(f"pre mha: {u}")

        u = (
            self.inner_mha_cls(
                self.pre_norm(u),
                inference_params=inference_params,
            )
            + u
        )
        if self.print_activations:
            activations_logger.info(f"post mha: {u}")

        if type(padding_mask) == torch.Tensor:  # guard against bias
            u = u * padding_mask[..., None]

        if self.print_activations:
            activations_logger.info(f"pre mlp: {u} {u.min()} {u.max()} {self.mlp.__class__}")
            activations_logger.info(
                f"post mlp norm: {self.post_norm(u)} {self.post_norm(u).min()} {self.post_norm(u).max()}"
            )
            activations_logger.info(
                f"post mlp: {self.mlp(self.post_norm(u))} {self.mlp(self.post_norm(u)).min()} {self.mlp(self.post_norm(u)).max()}"
            )

        u = self.mlp(self.post_norm(u)) + u
        return u, None
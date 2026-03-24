# Copyright (c) 2023, Tri Dao.

from typing import Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
import mindspore.numpy as mnp
from einops import rearrange, repeat
from vortex.ops.embedding.rotary import apply_rotary


def rotate_half(x: Tensor, interleaved: bool = False) -> Tensor:
    if not interleaved:
        x1, x2 = ops.split(x, split_size_or_sections=x.shape[-1] // 2, axis=-1)
        return ops.concat((-x2, x1), axis=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(stacked, "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torchlike(x: Tensor, cos: Tensor, sin: Tensor, interleaved: bool = False) -> Tensor:
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1], f"ro_dim {ro_dim} > headdim {x.shape[-1]}"
    if not interleaved:
        cos = repeat(cos, "... d -> ... 1 (2 d)")
        sin = repeat(sin, "... d -> ... 1 (2 d)")
    else:
        cos = repeat(cos, "... d -> ... 1 (d 2)")
        sin = repeat(sin, "... d -> ... 1 (d 2)")
    x_rotated = x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin
    return ops.concat([x_rotated, x[..., ro_dim:]], axis=-1)


class ApplyRotaryEmb(nn.Cell):
    """
    MindSpore Cell for applying rotary embeddings.
    Equivalent to PyTorch's ApplyRotaryEmb autograd.Function.
    Uses vortex.ops.embedding.rotary for the actual computation.
    """
    
    def __init__(self, interleaved: bool = False, inplace: bool = False):
        super().__init__()
        self.interleaved = interleaved
        self.inplace = inplace
        
    def construct(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        seqlen_offsets: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tensor:
        """
        Forward pass using vortex.ops.embedding.rotary.
        
        Args:
            x: (batch_size, seqlen, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            seqlen_offsets: offset for each sequence
            cu_seqlens: cumulative sequence lengths for variable length sequences
            max_seqlen: maximum sequence length
        
        Returns:
            Tensor with rotary embeddings applied
        """
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=self.inplace,
        )
        return out if not self.inplace else x

    def bprop(self, x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen, out, dout):
        """
        Backward pass using MindSpore's standard bprop signature.
        The gradient of rotary embedding is the conjugate operation.
        """
        # For conjugate, we apply with conjugate=True
        dx = apply_rotary(
            dout,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=self.interleaved,
            inplace=self.inplace,
            conjugate=True,
        )
        return (dx, None, None, None, None, None)


def apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tensor:
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    apply_fn = ApplyRotaryEmb(interleaved=interleaved, inplace=inplace)
    return apply_fn(x, cos, sin, seqlen_offsets, cu_seqlens, max_seqlen)


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(nn.Cell):
    def __init__(self, interleaved: bool = False, num_heads_q: Optional[int] = None):
        super().__init__()
        self.interleaved = interleaved
        self.num_heads_q = num_heads_q
        
    def construct(
        self,
        qkv: Tensor,
        cos: Tensor,
        sin: Tensor,
        cos_k: Optional[Tensor] = None,
        sin_k: Optional[Tensor] = None,
        seqlen_offsets: Union[int, Tensor] = 0,
    ) -> Tensor:
        """
        Apply rotary embedding to Q and K in QKV tensor.
        
        Args:
            qkv: (batch_size, seqlen, 3, nheads, headdim) or 
                 (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: optional separate embeddings for K
            seqlen_offsets: offset for each sequence
        
        Returns:
            qkv with rotary embeddings applied to Q and K
        """
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            else:
                assert qkv.dim() == 4
                assert self.num_heads_q is not None
                num_heads_k = (qkv.shape[2] - self.num_heads_q) // 2
                assert qkv.shape[2] == self.num_heads_q + 2 * num_heads_k
                qk = qkv[:, :, : self.num_heads_q + num_heads_k]
            apply_rotary(
                qk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=self.interleaved,
                inplace=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if qkv.dim() == 5:
                q, k = qkv[:, :, 0], qkv[:, :, 1]
            else:
                assert qkv.dim() == 4
                assert self.num_heads_q is not None
                num_heads_k = (qkv.shape[2] - self.num_heads_q) // 2
                assert qkv.shape[2] == self.num_heads_q + 2 * num_heads_k
                q, k = (
                    qkv[:, :, :self.num_heads_q],
                    qkv[:, :, self.num_heads_q : num_heads_q + num_heads_k],
                )
            apply_rotary(q, cos, sin, seqlen_offsets, interleaved=self.interleaved, inplace=True)
            apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=self.interleaved, inplace=True)
            
        return qkv

    def bprop(self, qkv, cos, sin, cos_k, sin_k, seqlen_offsets, out, dqkv):
        """Backward pass."""
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            if dqkv.ndim == 5:
                dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
            else:
                assert dqkv.ndim == 4
                assert self.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - self.num_heads_q) // 2
                assert dqkv.shape[2] == self.num_heads_q + 2 * num_heads_k
                dqk = dqkv[:, :, : self.num_heads_q + num_heads_k]
            
            apply_rotary(
                dqk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=self.interleaved,
                inplace=True,
                conjugate=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            
            if dqkv.ndim == 5:
                dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            else:
                assert dqkv.ndim == 4
                assert self.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - self.num_heads_q) // 2
                assert dqkv.shape[2] == self.num_heads_q + 2 * num_heads_k
                dq = dqkv[:, :, : self.num_heads_q]
                dk = dqkv[:, :, self.num_heads_q : self.num_heads_q + num_heads_k]
            
            apply_rotary(
                dq,
                cos,
                sin,
                seqlen_offsets,
                interleaved=self.interleaved,
                inplace=True,
                conjugate=True,
            )
            apply_rotary(
                dk,
                cos_k,
                sin_k,
                seqlen_offsets,
                interleaved=self.interleaved,
                inplace=True,
                conjugate=True,
            )
        
        return (dqkv, None, None, None, None, None)


def apply_rotary_emb_qkv_(
    qkv: Tensor,
    cos: Tensor,
    sin: Tensor,
    cos_k: Optional[Tensor] = None,
    sin_k: Optional[Tensor] = None,
    interleaved: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    num_heads_q: Optional[int] = None,
) -> Tensor:
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    """Functional interface for ApplyRotaryEmbQKV_."""
    apply_fn = ApplyRotaryEmbQKV_(interleaved=interleaved, num_heads_q=num_heads_q)
    return apply_fn(qkv, cos, sin, cos_k, sin_k, seqlen_offsets)


class ApplyRotaryEmbKV_(nn.Cell):
    """
    Apply rotary embeddings to KV tensor (only to K).
    Equivalent to PyTorch's ApplyRotaryEmbKV_ autograd.Function.
    Uses vortex.ops.embedding.rotary for the actual computation.
    """
    
    def __init__(self, interleaved: bool = False):
        super().__init__()
        self.interleaved = interleaved
        
    def construct(
        self,
        kv: Tensor,
        cos: Tensor,
        sin: Tensor,
        seqlen_offsets: Union[int, Tensor] = 0,
    ) -> Tensor:
        """
        Apply rotary embedding to K in KV tensor.
        
        Args:
            kv: (batch_size, seqlen, 2, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            seqlen_offsets: offset for each sequence
        
        Returns:
            kv with rotary embeddings applied to K
        """
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        
        k = kv[:, :, 0]
        
        apply_rotary(
            k,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=self.interleaved,
            inplace=True,
        )
        
        return kv


    def bprop(self, kv, cos, sin, seqlen_offsets, out, dkv):
        """Backward pass."""
        apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=self.interleaved,
            inplace=True,
            conjugate=True,
        )
        return (dkv, None, None, None)




def apply_rotary_emb_kv_(
    kv: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
) -> Tensor:
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    apply_fn = ApplyRotaryEmbKV_(interleaved=interleaved)
    return apply_fn(kv, cos, sin, seqlen_offsets)


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

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        device: Optional[str] = None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.interleaved = interleaved
        self.scale_base = scale_base
        # Generate and save the inverse frequency buffer (non trainable)
        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq()
        self.inv_freq = Parameter(inv_freq, requires_grad=False, name="inv_freq")
        # Compute scale if using XPos
        if scale_base is not None:
            scale = (mnp.arange(0, dim, 2, dtype=ms.float32) + 0.4 * dim) / (1.4 * dim)
            self.scale = Parameter(scale, requires_grad=False, name="scale")
        else:
            self.scale = None
# Cached values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self) -> Tensor:
         """Compute inverse frequency buffer."""
        return 1.0 / (self.base ** (mnp.arange(0, self.dim, 2, dtype=ms.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen: int, dtype: Optional[ms.dtype] = None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training

        # Determine if we need to update
        need_update = (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or (dtype is not None and self._cos_cached.dtype != dtype)
        )
        
        if need_update:
            self._seq_len_cached = seqlen
            
            # Compute position indices
            if self.pos_idx_in_fp32:
                t = mnp.arange(seqlen, dtype=ms.float32)
                # Use fp32 inv_freq for precision
                if self.inv_freq.dtype != ms.float32:
                    inv_freq = self._compute_inv_freq()
                else:
                    inv_freq = self.inv_freq
            else:
                t = mnp.arange(seqlen, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            
            # Compute frequencies: outer product of positions and inverse frequencies
            # freqs: (seqlen, dim // 2)
            inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = ops.outer(t, inv_freq)
            # Compute cosine and sine
            target_dtype = dtype if dtype is not None else ms.float32
            
            if self.scale is None:
                # Standard RoPE
                self._cos_cached = ops.cos(freqs).astype(target_dtype)
                self._sin_cached = ops.sin(freqs).astype(target_dtype)
            else:
                power = (mnp.arange(seqlen, dtype=ms.float32) - seqlen // 2) / self.scale_base
                # scale: (seqlen, dim // 2)
                scale = self.scale ** power.reshape(-1, 1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (mint.cos(freqs) * scale).to(dtype)
                self._sin_cached = (mint.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (mint.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (mint.sin(freqs) / scale).to(dtype)

    def construct(
        self,
        qkv: Tensor,
        kv: Optional[Tensor] = None,
        seqlen_offset: Union[int, Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply rotary embedding to qkv and/or kv.
        
        Args:
            qkv: (batch, seqlen, 3, nheads, headdim) or 
                 (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            kv: optional (batch, seqlen, 2, nheads, headdim). If provided,
                qkv is treated as just Q.
            seqlen_offset: offset for each sequence (for KV cache in inference)
            max_seqlen: maximum sequence length (for updating cache with tensor offset)
            num_heads_q: number of query heads (for GQA/MQA)
        
        Returns:
            If kv is None: qkv with rotary applied
            If kv is provided: (q, kv) with rotary applied
        """
        seqlen = qkv.shape[1]
        
        # Update cache
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, dtype=qkv.dtype)
        else:
            # Tensor offset: need to compute max
            max_offset = int(ops.max(seqlen_offset).asnumpy())
            self._update_cos_sin_cache(seqlen + max_offset, dtype=qkv.dtype)
        
        if kv is None:
            # Apply to QKV
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
        else:
            # Apply to Q and KV separately
            q = qkv
            
            # Apply to Q
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            
            # Apply to K in KV
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            
            return q, kv
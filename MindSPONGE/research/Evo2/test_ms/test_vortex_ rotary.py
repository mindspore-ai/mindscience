# rotary_ms.py
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import mindspore.numpy as mnp
from typing import Optional, Tuple, Union

def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = ops.split(x, split_size=x.shape[-1]//2, axis=-1)
        return ops.concat((-x2, x1), axis=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        # einops 语法与 numpy 一致，mindspore 也支持
        from einops import rearrange
        return rearrange(ops.stack((-x2, x1), axis=-1), "... d two -> ... (d two)", two=2)

def apply_rotary_emb_ms(x, cos, sin, interleaved=False):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    from einops import repeat
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return ops.concat([
        x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
        x[..., ro_dim:],
    ], axis=-1)

class RotaryEmbeddingMS(nn.Cell):
    def __init__(self, dim: int, base=10000.0, interleaved=False, scale_base=None):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        inv_freq = 1.0 / (self.base ** (mnp.arange(0, dim, 2, dtype=ms.float32) / dim))
        self.inv_freq = Parameter(inv_freq, requires_grad=False, name="inv_freq")
        scale = ((mnp.arange(0, dim, 2, dtype=ms.float32) + 0.4 * dim) / (1.4 * dim)) if scale_base else None
        self.scale = Parameter(scale, requires_grad=False, name="scale") if scale is not None else None
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seqlen, dtype=ms.float16):
        if seqlen > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seqlen
            t = mnp.arange(seqlen, dtype=ms.float32)
            freqs = ops.outer(t, self.inv_freq)          # (seqlen, dim//2)
            self._cos_cached = ops.cos(freqs).astype(dtype)
            self._sin_cached = ops.sin(freqs).astype(dtype)

    def construct(self, qkv, seqlen_offset=0):
        seqlen = qkv.shape[1]
        self._update_cos_sin_cache(seqlen + seqlen_offset, qkv.dtype)
        return apply_rotary_emb_ms(qkv, self._cos_cached, self._sin_cached, self.interleaved)
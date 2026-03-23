# test_rotary_ms.py
import pytest
from hypothesis import given, strategies as st
import torch
import mindspore as ms
from mindspore import nn

from rotary_ms import LinearlyScaledRotaryEmbedding, swap_mha_rope


# ----------------------------------------------------------------------
# 策略：生成合法超参
st_dim = st.integers(32, 256)
st_seqlen = st.integers(1, 512)
st_dtype = st.sampled_from([ms.float16, ms.float32])
st_factor = st.floats(0.5, 4.0)


# ----------------------------------------------------------------------
class DummyMHA(nn.Cell):
    """
    一个仅用于承载 rotary_emb 的「空壳」MHA。
    """
    def __init__(self, dim, heads=8, cross=False):
        super().__init__()
        self.cross_attn = cross
        d_head = dim // heads
        if cross:
            self.Wq = nn.Dense(dim, dim)
            self.Wk = nn.Dense(dim, dim)
            self.Wv = nn.Dense(dim, dim)
        else:
            self.Wqkv = nn.Dense(dim, 3 * dim)
        # 先放一个「Torch 风格」的 RoPE 基类
        from vortex.model.rotary import RotaryEmbedding
        self.rotary_emb = RotaryEmbedding(dim=d_head)


# ----------------------------------------------------------------------
@given(
    dim=st_dim,
    seqlen=st_seqlen,
    dtype=st_dtype,
    factor=st_factor,
)
def test_linearly_scaled_shape(dim, seqlen, dtype, factor):
    """
    仅验证：MindSpore 版 LinearlyScaledRotaryEmbedding 产生的 cos/sin
    缓存 shape 与 Torch 版完全一致。
    """
    # --- 1. Torch 参考 ---------------------------------------------------
    torch_rope = TorchLinearlyScaledRoPE(
        dim=dim, scaling_factor=factor, pos_idx_in_fp32=True
    )
    torch_rope._update_cos_sin_cache(seqlen, dtype=_ms_to_torch_dtype(dtype))
    torch_cos_shape = torch_rope._cos_cached.shape
    torch_sin_shape = torch_rope._sin_cached.shape

    # --- 2. MindSpore 待测 ----------------------------------------------
    ms_rope = LinearlyScaledRotaryEmbedding(
        dim=dim, scaling_factor=factor, pos_idx_in_fp32=True
    )
    ms_rope._update_cos_sin_cache(seqlen, dtype=dtype)
    ms_cos_shape = ms_rope._cos_cached.shape
    ms_sin_shape = ms_rope._sin_cached.shape

    # --- 3. 断言 ---------------------------------------------------------
    assert ms_cos_shape == torch_cos_shape, f"cos shape mismatch: {ms_cos_shape} vs {torch_cos_shape}"
    assert ms_sin_shape == torch_sin_shape, f"sin shape mismatch: {ms_sin_shape} vs {torch_sin_shape}"


# ----------------------------------------------------------------------
@given(dim=st_dim, factor=st_factor)
def test_swap_mha_rope(dim, factor):
    """
    验证 swap_mha_rope 之后，rotary_emb 类型正确，且缓存 shape 正常。
    """
    dummy = DummyMHA(dim=dim, cross=False)
    swap_mha_rope(dummy, kwargs_new_rope={"scaling_factor": factor})

    assert isinstance(dummy.rotary_emb, LinearlyScaledRotaryEmbedding)
    dummy.rotary_emb._update_cos_sin_cache(128, ms.float32)
    assert dummy.rotary_emb._cos_cached.shape == (128, dim)


# ----------------------------------------------------------------------
# 辅助：Torch 版「最小实现」用于比对 shape
class TorchLinearlyScaledRoPE(torch.nn.Module):
    def __init__(self, dim, scaling_factor=1.0, base=10_000.0, pos_idx_in_fp32=True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seqlen, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, dtype=torch.float32)
                t = t / self.scaling_factor
                inv_freq = self.inv_freq.float()
            else:
                t = torch.arange(seqlen, dtype=self.inv_freq.dtype)
                t = t / self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)


def _ms_to_torch_dtype(ms_dtype):
    return {ms.float16: torch.float16, ms.float32: torch.float32}[ms_dtype]


# ----------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
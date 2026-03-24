# vortex/test_ms/test_linearly_scaled_rope.py
import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, nn, Tensor
from torch import from_numpy as torch_from_numpy
from unittest.mock import MagicMock

# ------------------------------------------------------------------
#  0. 先 mock 缺失的 RotaryEmbedding
# ------------------------------------------------------------------
import sys
sys.modules['vortex.model.rotary'] = MagicMock()
sys.modules['vortex.model.rotary'].RotaryEmbedding = MagicMock

# ------------------------------------------------------------------
#  1. 引入 MindSpore 待测模块
# ------------------------------------------------------------------
from vortex.model.model import LinearlyScaledRotaryEmbedding, swap_mha_rope

# ------------------------------------------------------------------
#  2. PyTorch 参考实现（行为一致即可，不求源码一致）
# ------------------------------------------------------------------
class TorchLinearlyScaledRoPE:
    def __init__(self, dim, scaling_factor=1.0, base=10000.0,
                 interleaved=False, scale_base=None, pos_idx_in_fp32=True):
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.base = base
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def _update_cos_sin_cache(self, seqlen, dtype):
        if seqlen > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seqlen
            t = np.arange(seqlen, dtype=np.float32) / self.scaling_factor
            freqs = np.outer(t, self.inv_freq)
            cos = np.cos(freqs).astype(dtype)
            sin = np.sin(freqs).astype(dtype)
            self._cos_cached = cos
            self._sin_cached = sin

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        self._update_cos_sin_cache(seq_len, x.dtype)
        return self._cos_cached, self._sin_cached


# ------------------------------------------------------------------
#  3. 测试工具：固定种子 + 相对误差
# ------------------------------------------------------------------
def allclose(a, b, rtol=1e-3, atol=1e-5):
    return np.abs(a - b).max() < atol + rtol * np.abs(b).max()


# ------------------------------------------------------------------
#  4. 测试用例
# ------------------------------------------------------------------
class TestLinearlyScaledRoPE:
    @pytest.mark.parametrize("scaling_factor", [1.0, 2.0, 4.0])
    @pytest.mark.parametrize("seqlen", [128, 256])
    def test_cache_and_values(self, scaling_factor, seqlen):
        """缓存刷新 + 数值 vs PyTorch 对齐"""
        dim = 128
        dtype = ms.bfloat16
        torch_dtype = 'bfloat16'

        # MindSpore
        rope_ms = LinearlyScaledRotaryEmbedding(
            dim=dim, scaling_factor=scaling_factor, base=10000.0,
            interleaved=False, scale_base=None, pos_idx_in_fp32=True
        )

        # PyTorch
        rope_pt = TorchLinearlyScaledRoPE(
            dim=dim, scaling_factor=scaling_factor, base=10000.0,
            interleaved=False, scale_base=None, pos_idx_in_fp32=True
        )

        # 构造相同随机输入（仅用来触发缓存）
        x_np = np.random.randn(1, seqlen, dim).astype(np.float32)
        x_ms = Tensor(x_np, dtype=dtype)
        x_pt = torch_from_numpy(x_np).to(torch_dtype)

        # 前向触发缓存
        cos_ms, sin_ms = rope_ms(x_ms, seq_len=seqlen)
        cos_pt, sin_pt = rope_pt(x_pt, seq_len=seqlen)

        # 比较
        assert cos_ms.shape == cos_pt.shape
        assert sin_ms.shape == sin_pt.shape
        assert allclose(cos_ms.asnumpy(), cos_pt)
        assert allclose(sin_ms.asnumpy(), sin_pt)

    def test_swap_mha_rope(self):
        """验证 swap 后类型与参数正确"""
        # 假 MHA
        class DummyMHA(nn.Cell):
            def __init__(self):
                super().__init__()
                # 先放一个旧 rope
                self.rotary_emb = LinearlyScaledRotaryEmbedding(
                    dim=64, scaling_factor=1.0
                )

        mha = DummyMHA()
        old_dim = mha.rotary_emb.dim
        kwargs = {"scaling_factor": 3.0}

        # 替换
        swap_mha_rope(mha, LinearlyScaledRotaryEmbedding, kwargs)

        # 校验
        assert isinstance(mha.rotary_emb, LinearlyScaledRotaryEmbedding)
        assert mha.rotary_emb.dim == old_dim
        assert mha.rotary_emb._linear_scaling_factor == 3.0


# ------------------------------------------------------------------
#  5. 本地直接跑
# ------------------------------------------------------------------
if __name__ == '__main__':
    pytest.main([__file__, "-v"])
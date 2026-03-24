# vortex/test_ms/test_vortex_ rotary.py
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, mint, ops
from torch import from_numpy as torch_from_numpy
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# 0. 引入待测模块（假设已安装 vortex-ops 前向符号）
# ------------------------------------------------------------------
from vortex.vortex.model.rotary import (
    rotate_half,
    apply_rotary_emb_torch,
    ApplyRotaryEmb,
    apply_rotary_emb,
    ApplyRotaryEmbQKV_,
    apply_rotary_emb_qkv_,
    RotaryEmbedding,
)

# ------------------------------------------------------------------
# 1. PyTorch 参考实现（行为一致即可）
# ------------------------------------------------------------------
def rotate_half_pt(x, interleaved=False):
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2, -1)

def apply_rotary_emb_torch_pt(x, cos, sin, interleaved=False):
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = cos.unsqueeze(-2).repeat_interleave(2, dim=-1)  # (..., 1, 2d)
    sin = sin.unsqueeze(-2).repeat_interleave(2, dim=-1)
    rotated = x[..., :ro_dim] * cos + rotate_half_pt(x[..., :ro_dim], interleaved) * sin
    out = x.clone()
    out[..., :ro_dim] = rotated
    return out


# ------------------------------------------------------------------
# 2. 测试工具
# ------------------------------------------------------------------
def allclose(a, b, rtol=1e-3, atol=1e-5):
    return np.abs(a - b).max() < atol + rtol * np.abs(b).max()


def gen_cos_sin(seqlen, rotary_dim, dtype):
    t = np.arange(seqlen)
    inv_freq = 1.0 / (10000 ** (np.arange(0, rotary_dim, 2) / rotary_dim))
    freqs = np.outer(t, inv_freq)  # (S, rotary_dim//2)
    cos = np.cos(freqs).astype(dtype)
    sin = np.sin(freqs).astype(dtype)
    return cos, sin


# ------------------------------------------------------------------
# 3. 测试用例
# ------------------------------------------------------------------
class TestRotaryFull:
    # ---------- rotate_half ----------
    @pytest.mark.parametrize("interleaved", [False, True])
    def test_rotate_half(self, interleaved):
        x_np = np.random.randn(2, 8, 32).astype(np.float32)
        x_ms = Tensor(x_np, dtype=ms.float32)
        x_pt = torch_from_numpy(x_np)

        out_ms = rotate_half(x_ms, interleaved=interleaved).asnumpy()
        out_pt = rotate_half_pt(x_pt, interleaved=interleaved).numpy()
        assert allclose(out_ms, out_pt)

    # ---------- apply_rotary_emb_torch ----------
    @pytest.mark.parametrize("interleaved", [False, True])
    def test_apply_rotary_emb_torch(self, interleaved):
        B, S, H, D = 2, 64, 8, 64
        rotary_dim = 32
        x_np = np.random.randn(B, S, H, D).astype(np.float32)
        cos, sin = gen_cos_sin(S, rotary_dim, np.float32)

        x_ms = Tensor(x_np, dtype=ms.float32)
        cos_ms = Tensor(cos, dtype=ms.float32)
        sin_ms = Tensor(sin, dtype=ms.float32)

        x_pt = torch_from_numpy(x_np)
        cos_pt = torch_from_numpy(cos)
        sin_pt = torch_from_numpy(sin)

        out_ms = apply_rotary_emb_torch(x_ms, cos_ms, sin_ms, interleaved=interleaved).asnumpy()
        out_pt = apply_rotary_emb_torch_pt(x_pt, cos_pt, sin_pt, interleaved=interleaved).numpy()
        assert allclose(out_ms, out_pt)

    # ---------- ApplyRotaryEmb Cell（含反向） ----------
    @pytest.mark.parametrize("inplace", [False, True])
    def test_apply_rotary_emb_cell(self, inplace):
        B, S, H, D = 2, 128, 8, 64
        rotary_dim = 32
        x_np = np.random.randn(B, S, H, D).astype(np.float32)
        cos, sin = gen_cos_sin(S, rotary_dim, np.float32)

        x_ms = Tensor(x_np, dtype=ms.float32, requires_grad=True)
        cos_ms = Tensor(cos, dtype=ms.float32)
        sin_ms = Tensor(sin, dtype=ms.float32)

        x_pt = torch_from_numpy(x_np).requires_grad_(True)
        cos_pt = torch_from_numpy(cos)
        sin_pt = torch_from_numpy(sin)

        # 前向
        cell = ApplyRotaryEmb(interleaved=False, inplace=inplace)
        out_ms = cell(x_ms, cos_ms, sin_ms)
        out_pt = apply_rotary_emb_torch_pt(x_pt, cos_pt, sin_pt, interleaved=False)

        assert allclose(out_ms.asnumpy(), out_pt.numpy())

        # 反向
        grad_np = np.random.randn(*out_ms.shape).astype(np.float32)
        grad_ms = Tensor(grad_np, dtype=ms.float32)
        grad_pt = torch_from_numpy(grad_np)

        dx_ms = ops.GradOperation()(cell)(x_ms, cos_ms, sin_ms, grad_ms)
        out_pt.backward(grad_pt)
        dx_pt = x_pt.grad
        assert allclose(dx_ms.asnumpy(), dx_pt.numpy())

    # ---------- ApplyRotaryEmbQKV_ ----------
    def test_apply_rotary_emb_qkv_(self):
        B, S, three, H, D = 2, 64, 3, 8, 64
        rotary_dim = 32
        qkv_np = np.random.randn(B, S, three, H, D).astype(np.float32)
        cos, sin = gen_cos_sin(S, rotary_dim, np.float32)

        qkv_ms = Tensor(qkv_np, dtype=ms.float32)
        cos_ms = Tensor(cos, dtype=ms.float32)
        sin_ms = Tensor(sin, dtype=ms.float32)

        # PyTorch 参考：手动对 q、k 旋转
        qkv_pt = torch_from_numpy(qkv_np)
        q_pt = qkv_pt[:, :, 0]
        k_pt = qkv_pt[:, :, 1]
        cos_pt = torch_from_numpy(cos)
        sin_pt = torch_from_numpy(sin)
        q_rot_pt = apply_rotary_emb_torch_pt(q_pt, cos_pt, sin_pt, interleaved=False)
        k_rot_pt = apply_rotary_emb_torch_pt(k_pt, cos_pt, sin_pt, interleaved=False)

        # MindSpore
        out_ms = apply_rotary_emb_qkv_(qkv_ms, cos_ms, sin_ms, interleaved=False, num_heads_q=H)
        assert allclose(out_ms[:, :, 0].asnumpy(), q_rot_pt.numpy())
        assert allclose(out_ms[:, :, 1].asnumpy(), k_rot_pt.numpy())

    # ---------- RotaryEmbedding 端到端 ----------
    @pytest.mark.parametrize("seqlen_offset", [0, 10])
    def test_rotary_embedding_module(self, seqlen_offset):
        B, S, H, D = 2, 128, 8, 64
        rotary_dim = 32
        qkv_np = np.random.randn(B, S, 3, H, D).astype(np.float32)

        qkv_ms = Tensor(qkv_np, dtype=ms.float32, requires_grad=True)
        qkv_pt = torch_from_numpy(qkv_np).requires_grad_(True)

        # MindSpore
        rope_ms = RotaryEmbedding(rotary_dim, interleaved=False, pos_idx_in_fp32=True)
        out_ms = rope_ms(qkv_ms, seqlen_offset=seqlen_offset)

        # PyTorch 参考
        cos, sin = gen_cos_sin(S + seqlen_offset, rotary_dim, np.float32)
        cos_pt = torch_from_numpy(cos[seqlen_offset:])
        sin_pt = torch_from_numpy(sin[seqlen_offset:])
        q_pt = qkv_pt[:, :, 0]
        k_pt = qkv_pt[:, :, 1]
        v_pt = qkv_pt[:, :, 2]
        q_rot_pt = apply_rotary_emb_torch_pt(q_pt, cos_pt, sin_pt, interleaved=False)
        k_rot_pt = apply_rotary_emb_torch_pt(k_pt, cos_pt, sin_pt, interleaved=False)
        out_pt = torch.stack((q_rot_pt, k_rot_pt, v_pt), dim=2)

        assert allclose(out_ms.asnumpy(), out_pt.numpy())

        # 反向
        grad_np = np.random.randn(*out_ms.shape).astype(np.float32)
        grad_ms = Tensor(grad_np, dtype=ms.float32)
        grad_pt = torch_from_numpy(grad_np)

        dx_ms = ops.GradOperation()(rope_ms)(qkv_ms, Tensor(seqlen_offset))
        out_pt.backward(grad_pt)
        assert allclose(dx_ms.asnumpy(), qkv_pt.grad.numpy())


# ------------------------------------------------------------------
# 4. 本地直接跑
# ------------------------------------------------------------------
if __name__ == '__main__':
    pytest.main([__file__, "-v"])
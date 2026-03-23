# 文件路径: vortex/test_ms/test_engine.py

import pytest
import numpy as np
import mindspore as ms
import torch
from unittest.mock import MagicMock, patch


mock_logger = MagicMock()

def mock_column_split(x, num_heads, head_size):
    splits = x.shape[1] // 3
    if isinstance(x, ms.Tensor):
        return ms.ops.split(x, split_size_or_sections=splits, axis=1)
    else:
        return torch.split(x, splits, dim=1)


@patch("vortex.logging.activations_logger", mock_logger)
@patch("vortex.model.utils.column_split", side_effect=mock_column_split)
class TestHyenaEngineShape:
    
    def setup_method(self):
        # 导入 MindSpore 版本
        from vortex.model.engine import HyenaInferenceEngine as EngineMS
        from vortex.model.engine import fftconv_func as fftconv_ms
        
        # 导入 PyTorch 版本
        from vortex.test_ms.torch_scripts.engine import HyenaInferenceEngine as EnginePT
        from vortex.test_ms.torch_scripts.engine import fftconv_func as fftconv_pt
        
        self.EngineMS = EngineMS
        self.fftconv_ms = fftconv_ms
        self.EnginePT = EnginePT
        self.fftconv_pt = fftconv_pt

    def test_fftconv_func_shape(self, mock_split):
        """测试底层 FFT 卷积函数"""
        B, D, L = 2, 8, 16
        u_np = np.random.randn(B, D, L).astype(np.float32)
        k_np = np.random.randn(D, L).astype(np.float32)
        D_np = np.random.randn(D).astype(np.float32)
        
        # MindSpore
        out_ms = self.fftconv_ms(
            ms.Tensor(u_np), ms.Tensor(k_np), ms.Tensor(D_np), 
            dropout_mask=None, gelu=False, bidirectional=False
        )
        
        # PyTorch
        out_pt = self.fftconv_pt(
            torch.from_numpy(u_np), torch.from_numpy(k_np), torch.from_numpy(D_np), 
            dropout_mask=None, gelu=False, bidirectional=False
        )
        
        assert tuple(out_ms.shape) == tuple(out_pt.shape)

    def test_parallel_fir_shape(self, mock_split):
        """测试 Parallel FIR (FFT path)"""
        engine_ms = self.EngineMS(print_activations=False)
        engine_pt = self.EnginePT(print_activations=False)
        
        B, L, D = 2, 130, 12 # L > 128 触发 FFT
        dims = (D, 2, D//2, 0, 0)
        fir_length = 130
        
        u_np = np.random.randn(B, D, L).astype(np.float32)
        weight_np = np.random.randn(D, 1, fir_length).astype(np.float32)
        bias_np = np.random.randn(D).astype(np.float32)
        
        z_ms, _ = engine_ms.parallel_fir(
            fir_fn=ms.nn.Conv1d, 
            u=ms.Tensor(u_np), weight=ms.Tensor(weight_np), bias=ms.Tensor(bias_np), 
            L=L, dims=dims, fir_length=fir_length, gate=False, dim_last=True
        )
        
        z_pt, _ = engine_pt.parallel_fir(
            fir_fn=torch.nn.functional.conv1d,
            u=torch.from_numpy(u_np), weight=torch.from_numpy(weight_np), bias=torch.from_numpy(bias_np), 
            L=L, dims=dims, fir_length=fir_length, gate=False, dim_last=True
        )
        
        assert tuple(z_ms.shape) == tuple(z_pt.shape)

    def test_step_fir_shape(self, mock_split):
        """测试 Step FIR"""
        engine_ms = self.EngineMS()
        engine_pt = self.EnginePT()
        
        B, D = 2, 8
        cache_len = 5
        u_np = np.random.randn(B, D).astype(np.float32)
        state_np = np.random.randn(B, D, cache_len).astype(np.float32)
        w_np = np.random.randn(D, 1, cache_len + 1).astype(np.float32)
        b_np = np.random.randn(D).astype(np.float32)
        
        y_ms, state_ms_new = engine_ms.step_fir(
            ms.Tensor(u_np), ms.Tensor(state_np), ms.Tensor(w_np), ms.Tensor(b_np)
        )
        y_pt, state_pt_new = engine_pt.step_fir(
            torch.from_numpy(u_np), torch.from_numpy(state_np), torch.from_numpy(w_np), torch.from_numpy(b_np)
        )
        
        assert tuple(y_ms.shape) == tuple(y_pt.shape)
        assert tuple(state_ms_new.shape) == tuple(state_pt_new.shape)

    def test_parallel_iir_shape(self, mock_split):
        """测试 Parallel IIR"""
        engine_ms = self.EngineMS()
        engine_pt = self.EnginePT()
        
        B, L, D = 2, 16, 32
        dims = (D, 4, D//4, 16, 1) 
        
        z_pre_np = np.random.randn(B, 3*D, L).astype(np.float32)
        h_np = np.random.randn(D, L).astype(np.float32)
        D_bias_np = np.random.randn(D).astype(np.float32)
        poles_np = np.random.randn(D, 1, 1).astype(np.float32)
        residues_np = np.random.randn(D, 1, 1).astype(np.float32)
        t_np = np.arange(L).reshape(1, 1, L).astype(np.float32)
        
        # MindSpore
        out_ms = engine_ms.parallel_iir(
            z_pre=ms.Tensor(z_pre_np), h=ms.Tensor(h_np), D=ms.Tensor(D_bias_np),
            L=L, poles=ms.Tensor(poles_np), residues=ms.Tensor(residues_np), t=ms.Tensor(t_np),
            dims=dims, layer_idx=0, column_split_hyena=False, long_fir_threshold=None
        )
        
        # PyTorch
        out_pt = engine_pt.parallel_iir(
            z_pre=torch.from_numpy(z_pre_np), h=torch.from_numpy(h_np), D=torch.from_numpy(D_bias_np),
            L=L, poles=torch.from_numpy(poles_np), residues=torch.from_numpy(residues_np), t=torch.from_numpy(t_np),
            dims=dims, layer_idx=0, column_split_hyena=False, long_fir_threshold=None
        )
        
        assert tuple(out_ms.shape) == tuple(out_pt.shape)

if __name__ == "__main__":
    pytest.main(["-s", __file__])
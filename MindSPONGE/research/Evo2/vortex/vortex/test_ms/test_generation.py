# 文件路径: vortex/test_ms/test_generation.py

import pytest
import numpy as np
import mindspore as ms
import torch
from unittest.mock import MagicMock, patch


def mock_sample_fn(logits, top_k=50, top_p=0.7, temperature=1):
    """
    模拟采样函数。
    不进行复杂的概率计算，直接返回随机 Token ID。
    """
    batch_size = logits.shape[0]
    # MindSpore 分支
    if isinstance(logits, ms.Tensor):
        return ms.Tensor(np.random.randint(0, 10, size=(batch_size,)), dtype=ms.int64)
    # PyTorch 分支
    else:
        return torch.randint(0, 10, size=(batch_size,)).to(logits.device)

mock_print_rank_0 = MagicMock()


class MockTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.eos = 1
        self.bos = 0
    
    def tokenize(self, text):
        return [2, 3, 4] 
    
    def detokenize(self, ids):
        return "mock_string"
    
    def detokenize_batch(self, ids):
        return ["mock_string"] * len(ids)

class MockModelMS:
    def __init__(self, hidden_size=16, vocab_size=100):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def __call__(self, x, inference_params_dict=None):
        batch_size, seq_len = x.shape
        logits = ms.Tensor(np.random.randn(batch_size, seq_len, self.vocab_size), dtype=ms.float32)
        return logits, inference_params_dict

    def initialize_inference_params(self, max_seqlen=None):
        return None 

class MockModelPT(torch.nn.Module):
    def __init__(self, hidden_size=16, vocab_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, x, inference_params_dict=None):
        batch_size, seq_len = x.size()
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return logits, inference_params_dict
    
    def initialize_inference_params(self, max_seqlen=None):
        return None

@patch("vortex.model.sample.sample", side_effect=mock_sample_fn)
@patch("vortex.model.utils.print_rank_0", side_effect=mock_print_rank_0)
def test_generation_shape_basic(mock_print, mock_sample):
    """
    测试 Generator 的基础形状对齐
    """
    # 延迟导入，确保 patch 生效后再加载模块
    from vortex.model.generation import Generator as Generator_ms
    from vortex.test_ms.torch_scripts.generation import Generator as Generator_pt

    # 配置
    vocab_size = 50
    batch_size = 2
    prompt_len = 5
    gen_len = 10    
    
    # 初始化
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    model_ms = MockModelMS(vocab_size=vocab_size)
    model_pt = MockModelPT(vocab_size=vocab_size)
    
    gen_ms = Generator_ms(model_ms, tokenizer)
    gen_pt = Generator_pt(model_pt, tokenizer)

    # 构造输入
    input_ids_np = np.random.randint(0, vocab_size, size=(batch_size, prompt_len))
    input_ids_ms = ms.Tensor(input_ids_np, dtype=ms.int64)
    input_ids_pt = torch.from_numpy(input_ids_np).long()
    

    print("Running MindSpore Generation...")
    out_ids_ms, scores_ms, _ = gen_ms.generate(
        device="CPU", # 或者 GPU
        input_ids=input_ids_ms, 
        num_tokens=gen_len, 
        cached_generation=False,
        print_generation=False
    )
    

    print("Running PyTorch Generation...")
    out_ids_pt, scores_pt, _ = gen_pt.generate(
        device="cpu", 
        input_ids=input_ids_pt, 
        num_tokens=gen_len, 
        cached_generation=False,
        print_generation=False
    )


    expected_len = gen_len 
    
    print(f"MS Shape: {out_ids_ms.shape}, PT Shape: {out_ids_pt.shape}")

    assert out_ids_ms.shape == (batch_size, expected_len), \
        f"MS shape mismatch. Expected {(batch_size, expected_len)}, got {out_ids_ms.shape}"
    
    assert tuple(out_ids_ms.shape) == tuple(out_ids_pt.shape), "Shapes differ between frameworks"
    
    assert scores_ms.shape[-1] == vocab_size
    assert tuple(scores_ms.shape) == tuple(scores_pt.shape)

if __name__ == "__main__":
    # 允许直接运行调试
    pytest.main(["-s", __file__])
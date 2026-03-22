# vortex/test_ms/test_vortex_model.py
import numpy as np
import mindspore as ms
from torch import from_numpy, bfloat16

from vortex.model.utils import dotdict

from vortex.model.model import (
    AttentionBlock as AttentionBlock_ms,
)
from .torch_scripts.model import (
    AttentionBlock as AttentionBlock_pt,
)

dict_basic = { "hidden_size": 4096, "eps": 1e-6, "model_parallel_size": 1, "vocab_size": 512  }
dict_basic.update({ "num_attention_heads": 16, "use_flash_attn": False })
config_ms, config_pt = dotdict(dict_basic), dotdict(dict_basic)
layer_idx = 1

x_np = np.random.randn(1, 4, 4096).astype(np.float32)
x_ms = ms.Tensor(x_np, dtype=ms.bfloat16)
x_pt = from_numpy(x_np).to(dtype=bfloat16)

def test_attetion_block():
    attn_ms = AttentionBlock_ms(config_ms, layer_idx).to_float(ms.bfloat16)
    attn_pt = AttentionBlock_pt(config_pt, layer_idx).to(dtype=bfloat16)

    y_ms, _ = attn_ms.forward(x_ms)
    y_pt, _ = attn_pt(x_pt)

    assert tuple(y_ms.shape) == tuple(y_pt.size())
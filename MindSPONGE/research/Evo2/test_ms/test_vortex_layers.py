# vortex/test_ms/test_vortex_layers.py
import numpy as np
import mindspore as ms
from torch import from_numpy

from vortex.model.utils import dotdict
from vortex.model.layers import (
    RMSNorm as RMSNorm_ms,
    ParallelGatedMLP as ParallelGatedMLP_ms,
    VocabParallelEmbedding as VocabParallelEmbedding_ms,
)
from .torch_scripts.layers import (
    RMSNorm as RMSNorm_pt,
    ParallelGatedMLP as ParallelGatedMLP_pt,
    VocabParallelEmbedding as VocabParallelEmbedding_pt,
)

np.random.seed(123)

dict_basic = { "hidden_size": 16, "eps": 1e-6, "model_parallel_size": 1, "vocab_size": 512 }
config_ms, config_pt = dotdict(dict_basic), dotdict(dict_basic)
config_ms["dtype"] = "ms.float32"
config_pt["dtype"] = "torch.float32"

x_np = np.random.randn(2, 16, config_ms.hidden_size).astype(np.float32)
x_ms = ms.Tensor(x_np)
x_pt = from_numpy(x_np)

emb_input_np = np.array([[65, 67, 71, 84]])
emb_input_ms = ms.Tensor(emb_input_np)
emb_input_pt = from_numpy(emb_input_np)


def test_rmsnorm():
    norm_layer_ms = RMSNorm_ms(config_ms)
    norm_layer_pt = RMSNorm_pt(config_pt)

    y_ms = norm_layer_ms.forward(x_ms)
    y_pt = norm_layer_pt(x_pt)

    assert tuple(y_ms.shape) == tuple(y_pt.size())


def test_parallel_gated_mlp():
    mlp_ms = ParallelGatedMLP_ms(config_ms, 1)
    mlp_pt = ParallelGatedMLP_pt(config_pt, 1)

    y_ms = mlp_ms.forward(x_ms)
    y_pt = mlp_pt(x_pt)
    assert tuple(y_ms.shape) == tuple(y_pt.size())


def test_vocab_embedding():
    emb_ms = VocabParallelEmbedding_ms(config_ms)
    emb_pt = VocabParallelEmbedding_pt(config_pt)

    y_ms = emb_ms.forward(emb_input_ms)
    y_pt = emb_pt(emb_input_pt)
    
    assert tuple(y_ms.shape) == tuple(y_pt.size())

    y_emb_ms = emb_ms.unembed(y_ms)
    y_emb_pt = emb_pt.unembed(y_pt)

    assert tuple(y_emb_ms.shape) == tuple(y_emb_pt.size())

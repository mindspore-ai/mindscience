import torch
import torch.nn as nn
from src.layers import RMSNorm
from src.utils import dotdict

import mindspore as ms


def test_aa_fp_error(pytestconfig):
    input_dim = 1000
    output_dim = 1000
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear = nn.Linear(input_dim, output_dim).to(device).to(dtype)

    x1 = torch.randn(1, input_dim)
    x4 = x1.repeat(4, 1).to(dtype).to(device)

    y1 = linear(x1)
    y4 = linear(x4)

    if pytestconfig.getoption("verbose") > 0:
        print(y1[0])
        print(y4[0])

    assert False


def test_batched_norm(pytestconfig):
    config = {
        "eps": 1e-5,
        "hidden_size": 64,
        "params_dtype": ms.float32,
        "use_flash_rmsnorm": False,
    }
    config = dotdict(config)
    rmsnorm = RMSNorm(config).to_float(ms.bfloat16)

    inputs = ms.mint.randn(1, 64, dtype=ms.bfloat16)
    inputs = inputs.repeat(4, 1, 1)
    outputs_1 = rmsnorm(inputs[:1])
    outputs_4 = rmsnorm(inputs)

    if pytestconfig.getoption("verbose") > 0:
        print(outputs_1)
        print(outputs_4)

    assert False

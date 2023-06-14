# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test"""
import numpy as np

from mindspore import Tensor, ops
import torch
from torch.nn import functional as F

from mindchemistry.e3 import Gate, Irreps
from e3nn import nn


def test_gate_cmp():
    scalars = '6x0o+4x0e'
    gates = '5x0o+3x0e'
    gated = '2x2e+2x1o+4x0e'

    gate_ = nn.Gate(scalars, [torch.tanh, F.silu], gates, [torch.tanh, F.silu], gated)
    gate = Gate(scalars, [ops.tanh, ops.silu], gates, [ops.tanh, ops.silu], gated)

    v_np = np.random.rand(gate.irreps_in.dim).astype(np.float32)

    v_ = torch.tensor(v_np, requires_grad=True)
    vout_ = gate_(v_)
    vout_.backward(torch.ones_like(vout_))
    dv_ = v_.grad

    v = Tensor(v_np)
    vout = gate(v)
    grad = ops.grad(gate)
    dv = grad(v)

    assert np.allclose(vout_.detach().numpy(), vout.asnumpy(), rtol=1e-2, atol=1e-3)
    assert np.allclose(dv_.numpy(), dv.asnumpy(), rtol=1e-2, atol=1e-3)


def test_gate():
    g = Gate('2x0e', [ops.tanh], '1x0o+2x0e', [ops.abs], '2x1o+1x2e')
    v = Tensor([.1, .2, .1, .2, .3, .5, .6, .7, .6, .7, .8, .1, .2, .3, .4, .5])
    grad = ops.grad(g)

    assert g.irreps_in == Irreps('1x0o+4x0e+2x1o+1x2e')
    assert g.irreps_out == Irreps('2x0e+2x1o+1x2e')
    assert g(v).shape[-1] == g.irreps_out.dim
    assert grad(v).shape[-1] == g.irreps_in.dim


if __name__ == '__main__':
    test_gate_cmp()
    test_gate()

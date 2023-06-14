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
import numpy as np

from mindspore import Tensor, ops
import torch

from mindchemistry.e3 import Activation, Irreps
from e3nn import nn


def test_activation_cmp():
    scalars = '6x0e+4x0o'

    act_ = nn.Activation(scalars, [torch.tanh, torch.tanh])
    act = Activation(scalars, [ops.tanh, ops.tanh])

    v_np = np.random.rand(act.irreps_in.dim).astype(np.float32) * 10.

    v_ = torch.tensor(v_np, requires_grad=True)
    vout_ = act_(v_)
    vout_.backward(torch.ones_like(vout_))
    dv_ = v_.grad

    v = Tensor(v_np)
    vout = act(v)
    grad = ops.grad(act)
    dv = grad(v)

    assert np.allclose(vout_.detach().numpy(), vout.asnumpy(), rtol=1e-2, atol=1e-4)
    assert np.allclose(dv_.numpy(), dv.asnumpy(), rtol=1e-2, atol=1e-4)


def test_activation():
    a = Activation('3x0o+2x0e+1x0o', [ops.abs, ops.tanh, ops.sin])
    v = Tensor([.1, .2, .3, .4, .5, .6])
    grad = ops.grad(a)

    assert a.irreps_out == Irreps('3x0e+2x0e+1x0o')
    assert np.allclose(a(v).asnumpy(), np.array([[0.1, 0.2, 0.3, 0.6050552, 0.73590523, 0.85918677]]), rtol=1e-2,
                       atol=1e-4)
    assert np.allclose(grad(v).asnumpy(), np.array([1., 1., 1., 1.3625745, 1.2523901, 1.2558697]), rtol=1e-2, atol=1e-4)


if __name__ == '__main__':
    test_activation_cmp()
    test_activation()

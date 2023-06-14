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

from mindchemistry.e3 import TensorProduct, Irreps
from e3nn import o3


def test_tp_cmp():
    in1 = '2x1o+4x0o'
    in2 = '1x1o+3x0e'
    out = '1x2e+1x1o+5x0e'
    tp_ = o3.FullyConnectedTensorProduct(in1, in2, out)
    tp = TensorProduct(in1, in2, out, 'connect', weight_mode='custom')

    v1_np = np.random.rand(tp.irreps_in1.dim).astype(np.float32)
    v2_np = np.random.rand(tp.irreps_in2.dim).astype(np.float32)

    v1_ = torch.tensor(v1_np, requires_grad=True)
    v2_ = torch.tensor(v2_np, requires_grad=True)
    vout_ = tp_(v1_, v2_)
    vout_.backward(torch.ones_like(vout_))
    dv1_ = v1_.grad

    v1 = Tensor(v1_np)
    v2 = Tensor(v2_np)
    w = Tensor(tp_.weight.detach().numpy()).reshape(1, -1)
    vout = tp(v1, v2, w)
    grad = ops.grad(tp)
    dv1 = grad(v1, v2, w)

    assert np.allclose(vout_.detach().numpy(), vout.asnumpy(), rtol=1e-3, atol=1e-5)
    assert np.allclose(dv1_.numpy(), dv1.asnumpy(), rtol=1e-3, atol=1e-5)


def test_auto_complete():
    tp1 = TensorProduct('2x1o', '1x1o+3x0e')
    assert tp1.irreps_out == Irreps('2x0e+6x1o+2x1e+2x2e')
    v1 = Tensor([.1, .2, .3, .3, .2, .4])
    v2 = Tensor([.3, .2, .1, .5, .4, .3])
    assert tp1(v1, v2).shape[-1] == Irreps('2x0e+6x1o+2x1e+2x2e').dim

    tp2 = TensorProduct('2x2e+4x1o', '3x1e+3x0o', instructions='element')
    ins_auto = tp2.instructions
    ins_expect = [(0, 0, 0, 'uuu', False), (0, 0, 1, 'uuu', False), (0, 0, 2, 'uuu', False), (1, 1,
                                                                                              3, 'uuu', False),
                  (1, 1, 4, 'uuu', False), (1, 1, 5, 'uuu', False), (2, 2, 6, 'uuu', False)]
    assert ins_auto == ins_expect


def test_grad():
    tp = TensorProduct('2x1o+1x0o', '2x1o+1x2e', '2x2e+2x1e+3x1o', [(0, 0, 1, 'uvu', True), (1, 0, 1, 'uvuv', False), (
        0, 1, 2, 'uvw', True)], irrep_norm='component', path_norm='element', weight_init='normal')
    v1 = Tensor([.18, .26, .34, .25, .23, .45, .5])
    v2 = Tensor([.33, .24, .12, .55, .48, .37, .25, .76, .54, .14, .87])
    grad_fn = ops.value_and_grad(tp, grad_position=(0, 1), weights=tp.weights)
    v3, ((dv1, dv2), dw) = grad_fn(v1, v2)

    assert dv1.shape == (Irreps('2x1o+1x0o').dim,)
    assert dv2.shape == (Irreps('2x1o+1x2e').dim,)
    assert dw.shape == tp.weights.shape


if __name__ == '__main__':
    import mindspore as ms

    ms.set_context(device_target="CPU", device_id=7, mode=ms.GRAPH_MODE, save_graphs=False)
    test_tp_cmp()
    test_auto_complete()
    test_grad()

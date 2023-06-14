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

from mindspore import Tensor, float32
from mindspore import ops

from mindchemistry.e3 import Norm


def test_norm():
    n = Norm('3x1o')
    v = Tensor(np.linspace(1., 2., n.irreps_in.dim), dtype=float32)
    grad = ops.grad(n, grad_position=(0))

    assert np.allclose(n(v).asnumpy(), np.array([[1.9565594, 2.6040833, 3.252403]]), rtol=1e-4, atol=1e-6)
    assert np.allclose(grad(v).asnumpy(), np.array(
        [0.51110125, 0.57498896, 0.63887656, 0.52801687, 0.57601845, 0.6240199, 0.53806365, 0.57649684, 0.61492985]),
                       rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
    test_norm()

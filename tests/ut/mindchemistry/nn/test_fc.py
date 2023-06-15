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

from mindchemistry.e3 import FullyConnectedNet


def test_FullyConnectedNet():
    fc = FullyConnectedNet([2, 3, 1], ops.tanh, init_method='ones')
    v = Tensor([[.1, .2], [3., 4.]])
    grad = ops.grad(fc)
    assert np.allclose(fc(v).asnumpy(), np.array([[0.57660174], [2.7584996]]), rtol=1e-3, atol=1e-4)
    assert np.allclose(grad(v).asnumpy(), np.array([[1.8655336e+00, 1.8655336e+00], [3.9160994e-04, 3.9160994e-04]]),
                       rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    test_FullyConnectedNet()

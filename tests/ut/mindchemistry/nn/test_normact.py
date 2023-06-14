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

from mindspore import Tensor, float32
from mindspore import ops

from mindchemistry.e3 import NormActivation


def test_normact():
    nact = NormActivation("2x1e", ops.sigmoid, bias=True)
    v = Tensor(np.linspace(1., 1., nact.irreps_in.dim), dtype=float32)
    grad = ops.grad(nact, weights=nact.trainable_params())

    assert np.allclose(nact(v).reshape(1, 2, 3).norm(None, -1).asnumpy(),
                       ops.sigmoid(v.reshape(1, 2, 3).norm(None, -1)).asnumpy(), rtol=1e-4, atol=1e-6)
    assert grad(v)[1][0].shape == (2,)


if __name__ == '__main__':
    test_normact()

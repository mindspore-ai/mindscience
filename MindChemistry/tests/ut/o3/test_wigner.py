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
import pytest
import numpy as np

from mindspore import Tensor
from mindspore import ops

from mindchemistry.e3 import wigner_3j, wigner_D, rand_angles, angles_to_matrix, spherical_harmonics


def test_wigner_D():
    v = Tensor([.1, .2, .4])
    angles = rand_angles(2)
    rot = angles_to_matrix(*angles)
    wigD = wigner_D(1, *angles)
    assert np.allclose(ops.matmul(rot, v).asnumpy(), ops.matmul(
        wigD, v).asnumpy(), rtol=1e-4, atol=1e-6)

    def sh_3(x): return spherical_harmonics(3, x)

    assert np.allclose(sh_3(ops.matmul(rot, v)).asnumpy(), ops.matmul(
        wigner_D(3, *angles), sh_3(v)).asnumpy(), rtol=1e-4, atol=1e-6)


def test_wiger_3j():
    assert np.allclose(wigner_3j(2, 1, 3).asnumpy(), wigner_3j(
        1, 3, 2).transpose((2, 0, 1)).asnumpy())


if __name__ == '__main__':
    test_wigner_D()
    test_wiger_3j()

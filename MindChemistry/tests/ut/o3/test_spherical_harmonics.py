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

from mindchemistry.e3 import SphericalHarmonics, spherical_harmonics


def test_SphericalHarmonics():
    sh_2 = SphericalHarmonics(2, True)
    x = Tensor([[.1, .2, .4]])
    y = sh_2(x)
    y_expect = np.array([[0.2081044316291809, 0.10405221581459045, -
    0.13516780734062195, 0.4162088632583618, 0.3901958465576172]])
    assert np.allclose(y.asnumpy(), y_expect)

    grad_fn = ops.value_and_grad(sh_2)
    dx = grad_fn(x)[1]
    dx_expect = np.array(
        [[1.3643676042556763, 1.9296667575836182, -1.3059251308441162]])
    assert np.allclose(dx.asnumpy(), dx_expect)


def test_spherical_harmonics():
    def sh_3(x): return spherical_harmonics(3, x)

    x = Tensor([[.1, .2, .4]])
    y = sh_3(x)
    y_expect = np.array([[0.28817278146743774, 0.2402982860803604, -0.004749312065541744, -
    0.3334905803203583, -0.018997250124812126, 0.4505593478679657, 0.3188294768333435]])
    assert np.allclose(y.asnumpy(), y_expect)

    grad_fn = ops.value_and_grad(sh_3)
    dx = grad_fn(x)[1]
    dx_expect = np.array(
        [[0.7583111524581909, 3.519239664077759, -1.9491972923278809]])
    assert np.allclose(dx.asnumpy(), dx_expect)


if __name__ == '__main__':
    test_SphericalHarmonics()
    test_spherical_harmonics()

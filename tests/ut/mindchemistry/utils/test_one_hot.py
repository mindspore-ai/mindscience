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
import pytest
import numpy as np

from mindspore import Tensor

from mindchemistry.e3 import soft_one_hot_linspace


@pytest.mark.parametrize('basis', ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'])
@pytest.mark.parametrize('cutoff', [True, False])
def test_soft_one_hot_linspace(basis, cutoff):
    v = Tensor(np.random.rand(2, 3).astype(np.float32))
    out = soft_one_hot_linspace(v, 1., 2., 4, basis, cutoff)
    assert out.shape == v.shape + (4,)


if __name__ == '__main__':
    test_soft_one_hot_linspace('gaussian', False)

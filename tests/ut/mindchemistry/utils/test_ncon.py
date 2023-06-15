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

import mindspore as ms

from mindchemistry.e3 import Ncon


def ncon_cmp(con_list, ein_str, v_list, dtype):
    dtype_ms = {
        np.float16: ms.float16,
        np.float32: ms.float32,
        np.float64: ms.float64
    }[dtype]

    out_np = np.einsum(ein_str, *v_list, optimize='optimal')

    v_list = [ms.Tensor(t, dtype_ms) for t in v_list]
    out = Ncon(con_list)(v_list)

    assert np.allclose(out.asnumpy(), out_np, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ncon(dtype):
    con_list = [[], [-1, -2]]
    ein_str = ',ij'
    v1 = 1.4
    v2 = np.random.rand(5, 6).astype(dtype)
    v_list = [v1, v2]

    ncon_cmp(con_list, ein_str, v_list, dtype)


if __name__ == '__main__':
    test_ncon(np.float32)

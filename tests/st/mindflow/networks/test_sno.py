# Copyright 2024 Huawei Technologies Co., Ltd
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
"""mindflow st testcase"""

import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindflow.cell import SNO1D, SNO2D, SNO3D, get_poly_transform


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_sno_output():
    """
    Feature: Test SNO1D, SNO2D and SNO3D network in platform ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    context.set_context(mode=context.GRAPH_MODE)

    batch_size = 2
    n_modes = 4
    res = 8
    transform_data = get_poly_transform(res, n_modes, 'Chebyshev_t')
    transform = Tensor(transform_data["analysis"], mstype.float32)
    inv_transform = Tensor(transform_data["synthesis"], mstype.float32)

    model1d = SNO1D(in_channels=2, out_channels=2,
                    transforms=[[transform, inv_transform]], compute_dtype=mstype.float32)
    model2d = SNO2D(in_channels=2, out_channels=2,
                    transforms=[[transform, inv_transform]]*2,
                    num_usno_layers=1, compute_dtype=mstype.float32)
    model3d = SNO3D(in_channels=2, out_channels=2,
                    transforms=[[transform, inv_transform]]*3, compute_dtype=mstype.float16)

    data1d = Tensor(np.random.rand(batch_size, 2, res), mstype.float32)
    data2d = Tensor(np.random.rand(batch_size, 2, res, res), mstype.float32)
    data3d = Tensor(np.random.rand(batch_size, 2, res, res, res), mstype.float16)
    output1d = model1d(data1d)
    output2d = model2d(data2d)
    output3d = model3d(data3d)
    assert output1d.shape == (batch_size, 2, res)
    assert output1d.dtype == mstype.float32
    assert output2d.shape == (batch_size, 2, res, res)
    assert output2d.dtype == mstype.float32
    assert output3d.shape == (batch_size, 2, res, res, res)
    assert output3d.dtype == mstype.float16

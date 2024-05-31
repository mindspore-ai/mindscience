# Copyright 2023 Huawei Technologies Co., Ltd
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

from mindspore import Tensor, context, set_seed, load_param_into_net, load_checkpoint
from mindspore import dtype as mstype
from mindflow.cell import FNO1D, FNO2D, FNO3D
from mindflow.cell.neural_operators.dft import SpectralConv1dDft, SpectralConv2dDft, SpectralConv3dDft

RTOL = 0.001
set_seed(123456)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_fno_output():
    """
    Feature: Test FNO1D, FNO2D and FNO3D network in platform gpu and ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    Need to adaptive 910B
    """
    context.set_context(mode=context.GRAPH_MODE)
    model1d = FNO1D(
        in_channels=2, out_channels=2, n_modes=[2], resolutions=[6], fno_compute_dtype=mstype.float32)
    model2d = FNO2D(
        in_channels=2, out_channels=2, n_modes=[2, 2], resolutions=[6, 6], fno_compute_dtype=mstype.float32)
    model3d = FNO3D(
        in_channels=2, out_channels=2, n_modes=[2, 2, 2], resolutions=[6, 6, 6], fno_compute_dtype=mstype.float32)
    data1d = Tensor(np.ones((2, 6, 2)), dtype=mstype.float32)
    data2d = Tensor(np.ones((2, 6, 6, 2)), dtype=mstype.float32)
    data3d = Tensor(np.ones((2, 6, 6, 6, 2)), dtype=mstype.float32)
    output1d = model1d(data1d)
    output2d = model2d(data2d)
    output3d = model3d(data3d)
    assert output1d.shape == (2, 6, 2)
    assert output1d.dtype == mstype.float32
    assert output2d.shape == (2, 6, 6, 2)
    assert output2d.dtype == mstype.float32
    assert output3d.shape == (2, 6, 6, 6, 2)
    assert output3d.dtype == mstype.float32


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_spectralconvdft_output():
    """
    Feature: Test SpectralConv1dDft, SpectralConv2dDft and SpectralConv3dDft network in platform gpu and ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    model1d = SpectralConv1dDft(in_channels=2, out_channels=2, n_modes=[2], resolutions=[6])
    model2d = SpectralConv2dDft(in_channels=2, out_channels=2, n_modes=[2, 2], resolutions=[6, 6])
    model3d = SpectralConv3dDft(in_channels=2, out_channels=2, n_modes=[2, 2, 2], resolutions=[6, 6, 6])
    data1d = Tensor(np.ones((2, 2, 6)), dtype=mstype.float32)
    data2d = Tensor(np.ones((2, 2, 6, 6)), dtype=mstype.float32)
    data3d = Tensor(np.ones((2, 2, 6, 6, 6)), dtype=mstype.float32)
    target1d = 3.64671636
    target2d = 35.93239212
    target3d = 149.64256287
    param1 = load_checkpoint("./spectralconv1d.ckpt")
    param2 = load_checkpoint("./spectralconv2d.ckpt")
    param3 = load_checkpoint("./spectralconv3d.ckpt")
    load_param_into_net(model1d, param1)
    load_param_into_net(model2d, param2)
    load_param_into_net(model3d, param3)
    output1d = model1d(data1d)
    output2d = model2d(data2d)
    output3d = model3d(data3d)
    assert output1d.shape == (2, 2, 6)
    assert output1d.dtype == mstype.float32
    assert output1d.sum() - target1d < RTOL
    assert output2d.shape == (2, 2, 6, 6)
    assert output2d.dtype == mstype.float32
    assert output2d.sum() - target2d < RTOL
    assert output3d.shape == (2, 2, 6, 6, 6)
    assert output3d.dtype == mstype.float32
    assert output3d.sum() - target3d < RTOL

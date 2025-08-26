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
"""ffno testcase"""

import os
import sys
import time

import pytest
import numpy as np

import mindspore as ms
from mindspore import nn, Tensor, set_seed, load_param_into_net, load_checkpoint
from mindspore import dtype as mstype

from mindscience.models import FFNO1D, FFNO2D, FFNO3D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# pylint: disable=wrong-import-position

from tools import compare_output, FP32_RTOL

# pylint: enable=wrong-import-position

set_seed(123456)
folder_path = "/home/workspace/mindspore_dataset/mindscience/ffno"

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno1d_output(mode):
    """
    Feature: Test FFNO1D network in platform ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model1d = FFNO1D(in_channels=2,
                     out_channels=2,
                     n_modes=[2],
                     resolutions=[6],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data1d = Tensor(np.load(os.path.join(folder_path, "ffno_data1d.npy")), dtype=mstype.float32)
    param1d = load_checkpoint(os.path.join(folder_path, "ffno1d.ckpt"))
    load_param_into_net(model1d, param1d)
    output1d = model1d(data1d)
    target1d = np.load(os.path.join(folder_path, "ffno_target1d.npy"))

    assert output1d.shape == (2, 6, 2)
    assert output1d.dtype == mstype.float32
    assert compare_output(output1d.asnumpy(), target1d, rtol=FP32_RTOL, atol=FP32_RTOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno1d_mse_loss_output(mode):
    """
    Feature: Test FFNO1D MSE Loss in platform ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model1d = FFNO1D(in_channels=2,
                     out_channels=2,
                     n_modes=[2],
                     resolutions=[6],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data1d = Tensor(np.ones((2, 6, 2)), dtype=mstype.float32)
    label_1d = Tensor(np.ones((2, 6, 2)), dtype=mstype.float32)
    param1d = load_checkpoint(os.path.join(folder_path, "ffno1d.ckpt"))
    load_param_into_net(model1d, param1d)

    loss_fn = nn.MSELoss()
    optimizer_1d = nn.SGD(model1d.trainable_params(), learning_rate=0.01)
    net_with_loss_1d = nn.WithLossCell(model1d, loss_fn)
    train_step_1d = nn.TrainOneStepCell(net_with_loss_1d, optimizer_1d)

    # calculate two steps of loss
    loss_1d = train_step_1d(data1d, label_1d)
    target_loss_1_1d = 0.63846040
    assert compare_output(loss_1d.asnumpy(), target_loss_1_1d, rtol=FP32_RTOL, atol=FP32_RTOL)

    loss_1d = train_step_1d(data1d, label_1d)
    target_loss_2_1d = 0.04462930
    assert compare_output(loss_1d.asnumpy(), target_loss_2_1d, rtol=FP32_RTOL, atol=FP32_RTOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno2d_output(mode):
    """
    Feature: Test FFNO2D network in platform ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model2d = FFNO2D(in_channels=2,
                     out_channels=2,
                     n_modes=[2, 2],
                     resolutions=[6, 6],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data2d = Tensor(np.load(os.path.join(folder_path, "ffno_data2d.npy")), dtype=mstype.float32)
    param2d = load_checkpoint(os.path.join(folder_path, "ffno2d.ckpt"))
    load_param_into_net(model2d, param2d)
    output2d = model2d(data2d)
    target2d = np.load(os.path.join(folder_path, "ffno_target2d.npy"))

    assert output2d.shape == (2, 6, 6, 2)
    assert output2d.dtype == mstype.float32
    assert compare_output(output2d.asnumpy(), target2d, rtol=FP32_RTOL, atol=FP32_RTOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno2d_mse_loss_output(mode):
    """
    Feature: Test FFNO2D MSE Loss in platform ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model2d = FFNO2D(in_channels=2,
                     out_channels=2,
                     n_modes=[2, 2],
                     resolutions=[6, 6],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data2d = Tensor(np.ones((2, 6, 6, 2)), dtype=mstype.float32)
    label_2d = Tensor(np.ones((2, 6, 6, 2)), dtype=mstype.float32)
    param2d = load_checkpoint(os.path.join(folder_path, "ffno2d.ckpt"))
    load_param_into_net(model2d, param2d)

    loss_fn = nn.MSELoss()
    optimizer_2d = nn.SGD(model2d.trainable_params(), learning_rate=0.01)
    net_with_loss_2d = nn.WithLossCell(model2d, loss_fn)
    train_step_2d = nn.TrainOneStepCell(net_with_loss_2d, optimizer_2d)

    # calculate two steps of loss
    loss_2d = train_step_2d(data2d, label_2d)
    target_loss_1_2d = 1.70347130
    assert compare_output(loss_2d.asnumpy(), target_loss_1_2d, rtol=FP32_RTOL, atol=FP32_RTOL)

    loss_2d = train_step_2d(data2d, label_2d)
    target_loss_2_2d = 0.28143430
    assert compare_output(loss_2d.asnumpy(), target_loss_2_2d, rtol=FP32_RTOL, atol=FP32_RTOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno3d_output(mode):
    """
    Feature: Test FFNO3D network in platform ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model3d = FFNO3D(in_channels=2,
                     out_channels=2,
                     n_modes=[2, 2, 2],
                     resolutions=[6, 6, 6],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data3d = Tensor(np.load(os.path.join(folder_path, "ffno_data3d.npy")), dtype=mstype.float32)
    param3d = load_checkpoint(os.path.join(folder_path, "ffno3d.ckpt"))
    load_param_into_net(model3d, param3d)
    output3d = model3d(data3d)
    target3d = np.load(os.path.join(folder_path, "ffno_target3d.npy"))

    assert output3d.shape == (2, 6, 6, 6, 2)
    assert output3d.dtype == mstype.float32
    assert compare_output(output3d.asnumpy(), target3d, rtol=FP32_RTOL, atol=FP32_RTOL)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno3d_mse_loss_output(mode):
    """
    Feature: Test FFNO3D MSE Loss in platform ascend.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model3d = FFNO3D(in_channels=2,
                     out_channels=2,
                     n_modes=[2, 2, 2],
                     resolutions=[6, 6, 6],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data3d = Tensor(np.ones((2, 6, 6, 6, 2)), dtype=mstype.float32)
    label_3d = Tensor(np.ones((2, 6, 6, 6, 2)), dtype=mstype.float32)
    param3d = load_checkpoint(os.path.join(folder_path, "ffno3d.ckpt"))
    load_param_into_net(model3d, param3d)

    loss_fn = nn.MSELoss()
    optimizer_3d = nn.SGD(model3d.trainable_params(), learning_rate=0.01)
    net_with_loss_3d = nn.WithLossCell(model3d, loss_fn)
    train_step_3d = nn.TrainOneStepCell(net_with_loss_3d, optimizer_3d)

    # calculate two steps of loss
    loss_3d = train_step_3d(data3d, label_3d)
    target_loss_1_3d = 1.94374371
    assert compare_output(loss_3d.asnumpy(), target_loss_1_3d, rtol=FP32_RTOL, atol=FP32_RTOL)

    loss_3d = train_step_3d(data3d, label_3d)
    target_loss_2_3d = 0.24034855
    assert compare_output(loss_3d.asnumpy(), target_loss_2_3d, rtol=FP32_RTOL, atol=FP32_RTOL)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno1d_speed(mode):
    """
    Feature: Test FFNO1D training speed in platform ascend.
    Description: The speed of each training step.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model1d = FFNO1D(in_channels=32,
                     out_channels=32,
                     n_modes=[16],
                     resolutions=[128],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data1d = Tensor(np.ones((32, 128, 32)), dtype=mstype.float32)
    label_1d = Tensor(np.ones((32, 128, 32)), dtype=mstype.float32)

    loss_fn = nn.MSELoss()
    optimizer_1d = nn.SGD(model1d.trainable_params(), learning_rate=0.01)
    net_with_loss_1d = nn.WithLossCell(model1d, loss_fn)
    train_step_1d = nn.TrainOneStepCell(net_with_loss_1d, optimizer_1d)

    steps = 10
    for _ in range(10):
        train_step_1d(data1d, label_1d)

    start_time = time.time()
    for _ in range(10):
        train_step_1d(data1d, label_1d)
    end_time = time.time()

    assert (end_time - start_time) / steps < 0.5


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno2d_speed(mode):
    """
    Feature: Test FFNO2D training speed in platform ascend.
    Description: The speed of each training step.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model2d = FFNO2D(in_channels=32,
                     out_channels=32,
                     n_modes=[16, 16],
                     resolutions=[64, 64],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data2d = Tensor(np.ones((32, 64, 64, 32)), dtype=mstype.float32)
    label_2d = Tensor(np.ones((32, 64, 64, 32)), dtype=mstype.float32)

    loss_fn = nn.MSELoss()
    optimizer_2d = nn.SGD(model2d.trainable_params(), learning_rate=0.01)
    net_with_loss_2d = nn.WithLossCell(model2d, loss_fn)
    train_step_2d = nn.TrainOneStepCell(net_with_loss_2d, optimizer_2d)

    steps = 10
    for _ in range(steps):
        train_step_2d(data2d, label_2d)

    start_time = time.time()
    for _ in range(steps):
        train_step_2d(data2d, label_2d)
    end_time = time.time()

    assert (end_time - start_time) / steps < 1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ffno3d_speed(mode):
    """
    Feature: Test FFNO3D training speed in platform ascend.
    Description: The speed of each training step.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=mode)
    model3d = FFNO3D(in_channels=2,
                     out_channels=2,
                     n_modes=[16, 16, 16],
                     resolutions=[32, 32, 32],
                     hidden_channels=2,
                     n_layers=2,
                     share_weight=True,
                     r_padding=8,
                     ffno_compute_dtype=mstype.float32)

    data3d = Tensor(np.ones((2, 32, 32, 32, 2)), dtype=mstype.float32)
    label_3d = Tensor(np.ones((2, 32, 32, 32, 2)), dtype=mstype.float32)

    loss_fn = nn.MSELoss()
    optimizer_3d = nn.SGD(model3d.trainable_params(), learning_rate=0.01)
    net_with_loss_3d = nn.WithLossCell(model3d, loss_fn)
    train_step_3d = nn.TrainOneStepCell(net_with_loss_3d, optimizer_3d)

    steps = 10
    for _ in range(steps):
        train_step_3d(data3d, label_3d)

    start_time = time.time()
    for _ in range(steps):
        train_step_3d(data3d, label_3d)
    end_time = time.time()

    assert (end_time - start_time) / steps < 3

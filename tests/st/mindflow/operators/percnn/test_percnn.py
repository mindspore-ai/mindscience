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
# ==============================================================================
"""Test mindflow percnn"""

import time

import torch
import numpy as np
import pytest
import mindspore as ms
from mindflow.cell.neural_operators.percnn import PeRCNN

from rcnn_3d import PeRCNN3D
from rcnn_2d import PeRCNN2D
from constant import laplace_3d, lap_2d_op

torch.set_default_dtype(torch.float64)
ms.context.set_context(mode=ms.GRAPH_MODE)


def ms2torch(params_ms, params_torch):
    """convert parameter from torch to mindspore"""
    ms2torch_dict = {
        "coef_u": "ca",
        "coef_v": "cb",
        "u_0.weight": "wh1_u.weight",
        "u_0.bias": "wh1_u.bias",
        "u_1.weight": "wh2_u.weight",
        "u_1.bias": "wh2_u.bias",
        "u_2.weight": "wh3_u.weight",
        "u_2.bias": "wh3_u.bias",
        "u_conv.weight": "wh4_u.weight",
        "u_conv.bias": "wh4_u.bias",
        "v_0.weight": "wh1_v.weight",
        "v_0.bias": "wh1_v.bias",
        "v_1.weight": "wh2_v.weight",
        "v_1.bias": "wh2_v.bias",
        "v_2.weight": "wh3_v.weight",
        "v_2.bias": "wh3_v.bias",
        "v_conv.weight": "wh4_v.weight",
        "v_conv.bias": "wh4_v.bias",
    }

    for item in params_ms:
        value = params_torch[ms2torch_dict[item.name]].data.cpu().numpy()
        item.set_data(ms.Tensor(value, ms.float32))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_percnn_3d():
    """
    Feature: Compare the inference accuracy of PeRCNN with its Torch implementation on a 3D physical field.
    Description: None.
    Expectation: Success or raise AssertionError.
    Need to adaptive 910B
    """
    laplace = np.array(laplace_3d)
    grid_size = 48
    computation_field = 100
    dx_3d = computation_field / grid_size
    laplace_3d_kernel = ms.Tensor(1 / dx_3d**2 * laplace, dtype=ms.float32)

    rcnn_ms = PeRCNN(
        dim=3,
        in_channels=2,
        hidden_channels=2,
        kernel_size=1,
        dt=0.5,
        nu=0.274,
        laplace_kernel=laplace_3d_kernel,
        conv_layers_num=3,
        compute_dtype=ms.float32,
    )
    rcnn_torch = PeRCNN3D(input_channels=2, hidden_channels=2)

    params_ms = rcnn_ms.trainable_params()
    params_torch = rcnn_torch.state_dict()
    ms2torch(params_ms, params_torch)
    inputs = np.random.randn(1, 2, 48, 48, 48)
    input_ms = ms.Tensor(inputs, ms.float32)
    output_ms = rcnn_ms(input_ms)

    input_torch = torch.tensor(inputs)
    output_torch = rcnn_torch(input_torch)
    mse = ms.ops.mse_loss(output_ms, ms.Tensor(output_torch.detach().numpy()))
    assert mse.float() < 1e-6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_percnn_2d():
    """
    Feature: Compare the inference accuracy of PeRCNN with its Torch implementation on a 2D physical field.
    Description: None.
    Expectation: Success or raise AssertionError.
    Need to adaptive 910B
    """
    grid_size = 100
    computation_field = 1
    dx_2d = computation_field / grid_size
    laplace_2d_kernel = ms.Tensor(
        1 / dx_2d**2 * np.array(lap_2d_op), dtype=ms.float32
    )

    rcnn_ms_2d = PeRCNN(
        dim=2,
        in_channels=2,
        hidden_channels=16,
        kernel_size=5,
        dt=0.00025,
        nu=0.01,
        laplace_kernel=laplace_2d_kernel,
        conv_layers_num=3,
        compute_dtype=ms.float32,
    )
    rcnn_torch_2d = PeRCNN2D(
        input_channels=2,
        hidden_channels=4,
        input_kernel_size=5,
        input_stride=1,
        input_padding=2,
    )
    ms2torch(rcnn_ms_2d.trainable_params(), rcnn_torch_2d.state_dict())
    inputs_2d = np.random.randn(1, 2, 48, 48)
    input_ms_2d = ms.Tensor(inputs_2d, ms.float32)
    output_ms_2d = rcnn_ms_2d(input_ms_2d)

    input_torch_2d = torch.tensor(inputs_2d)
    output_torch_2d = rcnn_torch_2d(input_torch_2d)
    mse = ms.ops.mse_loss(output_ms_2d, ms.Tensor(
        output_torch_2d.detach().numpy()))
    assert mse.float() < 1e-6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_percnn_compile_time():
    """
    Feature: Test the time of compiling 3d percnn.
    Description: rollout 100 steps.
    Expectation: Success or raise AssertionError.
    Need to adaptive 910B
    """
    laplace = np.array(laplace_3d)
    dx_3d = 100 / 48
    laplace_3d_kernel = ms.Tensor(1 / dx_3d**2 * laplace, dtype=ms.float32)

    rcnn_ms = PeRCNN(
        dim=3,
        in_channels=2,
        hidden_channels=2,
        kernel_size=1,
        dt=0.5,
        nu=0.274,
        laplace_kernel=laplace_3d_kernel,
        conv_layers_num=3,
        compute_dtype=ms.float32,
    )

    class Wrapper(ms.nn.Cell):
        """trainer wrapper"""
        def __init__(self, rcnn, rollout):
            super().__init__()
            self.model = rcnn
            self.rollout = rollout

        def get_res(self, x):
            for _ in range(self.rollout):
                x = self.model(x)
            return x

        def construct(self, x):
            return self.get_res(x)

    trainer = Wrapper(rcnn_ms, 100)
    x = np.random.randn(1, 2, 48, 48, 48)
    x = ms.Tensor(x, ms.float32)
    begin = time.time()
    x = trainer(x)
    now = time.time()
    assert now-begin < 150

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
import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
import torch
from mindflow.cell import UNet2D as UNet2D_mindspore

from tests.st.mindflow.networks.unet2d.model import UNet2D as UNet2D_torch


RTOL = 0.003

def inference_alignment(base_channels):
    """inference alignment"""
    pt_model = UNet2D_torch(3, 3, base_channels)
    device = torch.device("cpu")
    pt_model.to(device)

    ms_model = UNet2D_mindspore(3, 3, base_channels)
    ms_model.set_train()

    data = np.ones((2, 128, 128, 3))
    data_ms = ms.Tensor(data, dtype=ms.float32)
    data_pt = torch.Tensor(data)
    data_pt = data_pt.to(device)

    ms_key_list = []
    pt_key_list = []
    for k, _ in ms_model.parameters_dict().items():
        if '.0.weight' in k or '.3.weight' in k or 'up.weight' in k or 'outc.weight' in k:
            ms_key_list.append(k)
    for k in pt_model.state_dict().keys():
        if '.0.weight' in k or '.3.weight' in k or 'up.weight' in k or 'outc.weight' in k:
            pt_key_list.append(k)
    ms_to_torch_dict = {c2: c1 for (c1, c2) in zip(pt_key_list, ms_key_list)}
    ms_ckpt_dict = ms_model.parameters_dict()

    for k, v in ms_to_torch_dict.items():
        init = ms.Parameter(pt_model.state_dict()[v].cpu().detach().numpy())
        ms_ckpt_dict[k] = init

    ms.load_param_into_net(ms_model, ms_ckpt_dict)

    output_torch = pt_model(data_pt)
    output_torch = output_torch.cpu().detach().numpy()
    output_mindspore = ms_model(data_ms)
    output_mindspore = output_mindspore.asnumpy()
    return output_torch, output_mindspore


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_unet2d_precision():
    """
    Feature: Test UNet2D network precision on platform cpu.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    for base_channels in [4, 8, 16, 32, 64]:
        output_torch, output_mindspore = inference_alignment(base_channels)
        assert np.average(abs(abs(output_mindspore) - abs(output_torch))) / np.average(abs(output_torch)) < RTOL


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_unet2d_shape():
    """
    Feature: Test UNet2D network output shape on platform cpu.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    input_tensor = Tensor(np.ones((2, 128, 128, 3)), mstype.float32)
    model = UNet2D_mindspore(in_channels=3, out_channels=3, base_channels=64, data_format="NHWC")
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (2, 128, 128, 3)
    assert output_tensor.dtype == mstype.float32

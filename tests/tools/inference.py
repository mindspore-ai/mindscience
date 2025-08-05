# ============================================================================
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
"""inference"""
import os
import numpy as np
from mindspore import load_checkpoint, load_param_into_net

from .utils import compare_output


def validate_output_dtype(model, inputs, dtype):
    r"""
    Validate model output dtype
    Args:
        model (nn.Cell): Network.
        inputs (Tensor): Mindspore input tensor
        dtype (nn.dtype): Data dtype

    Returns:
        bool: Whether the test passes (i.e., output matches the target data within the tolerance)
    """
    if isinstance(inputs, (list, tuple)):
        output = model(*inputs)
    else:
        output = model(inputs)
    assert output.dtype == dtype, f'output dtype is {output.dtype} not equal {dtype}'


def validate_model_infer(model, inputs, ckpt_path, target_data_path, rtol, atol):
    r"""
    Validate model output is as expected
    Args:
        model (nn.Cell): Network.
        inputs (Tensor): Mindspore input tensor
        ckpt_path (str): Checkpoint path
        target_data_path (str): Target data path
        rtol (float): Relative tolerance for allowed error, default is 1e-5
        atol (float): Absolute tolerance for allowed error, default is 1e-5

    Returns:
        bool: Whether the test passes (i.e., output matches the target data within the tolerance)
    """
    assert os.path.exists(ckpt_path)
    params_dict = load_checkpoint(ckpt_path)
    load_param_into_net(model, params_dict)
    if isinstance(inputs, (list, tuple)):
        out = model(*inputs)
    else:
        out = model(inputs)
    out_target = np.load(target_data_path)
    return compare_output(out_target, out.numpy(), rtol, atol)

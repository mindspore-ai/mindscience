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
"""Test Inference"""
import numpy as np

from ais_bench.infer.interface import InferSession

from .utils import compare_output


def validate_om_infer(inputs, omm_path, target_data_path, device_id=0, rtol: float = 1e-5,
                      atol: float = 1e-5,):
    r"""
    Validate if the inference output matches the target data within the specified tolerance
    Args:
        inputs (Tensor): Input data for inferenc
        omm_path (str): Path to the OM model used for inference
        target_data_path (str): Path to the target output data to compare against
        device_id (int): Device ID, default is 0
        rtol (float): Relative tolerance for allowed error, default is 1e-5
        atol (float): Absolute tolerance for allowed error, default is 1e-5

    Returns:
        bool: Whether the test passes (i.e., output matches the target data within the tolerance)
    """
    model = InferSession(device_id, omm_path)
    if isinstance(inputs, (list, tuple)):
        out = model.infer(inputs)
    else:
        out = model.infer([inputs])
    out_target = np.load(target_data_path)
    return compare_output(out_target, out, rtol, atol)

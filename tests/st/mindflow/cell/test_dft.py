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
"""DFT Test Case"""
import os
import random
import sys

import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops, set_seed
from mindspore import dtype as mstype
from mindflow.cell.neural_operators.dft import dft1, idft1

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from common.cell import compare_output
from common.cell import FP32_RTOL, FP32_ATOL

set_seed(0)
np.random.seed(0)
random.seed(0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dft1_forward_accuracy(mode):
    """
    Feature: DFT1 forward accuracy test
    Description: Test the accuracy of the DFT1 operation in both GRAPH_MODE and PYNATIVE_MODE
            with input data loaded from './mindflow/cell/dft/data/dft1_input.npy'.
            The expected output is compared to a reference output stored in
            './mindflow/cell/dft/data/dft1_output.npy'.
    Expectation: The output should match the target data within the defined relative and absolute tolerance,
            ensuring the DFT1 forward computation is accurate.
    """
    ms.set_context(mode=mode)
    input_data = np.load('./dft/data/dft1_input.npy')
    x_re = Tensor(input_data, dtype=mstype.float32)
    x_im = ops.zeros_like(x_re)
    dft1_cell = dft1(shape=(len(x_re),), modes=2, compute_dtype=mstype.float32)
    ret, _ = dft1_cell((x_re, x_im))
    ret = ret.asnumpy()
    output_target = np.load('./dft/data/dft1_output.npy')
    validate_ans = compare_output(ret, output_target, rtol=FP32_RTOL, atol=FP32_ATOL)
    assert validate_ans, "The verification of dft1 forward accuracy is not successful."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_idft1_forward_accuracy(mode):
    """
    Feature: IDFT1 forward accuracy test
    Description: Test the accuracy of the IDFT1 operation in both GRAPH_MODE and PYNATIVE_MODE
                with input data loaded from './mindflow/cell/dft/data/idft1_input.npy'.
                The expected output is compared to a reference output stored in
                './mindflow/cell/dft/data/idft1_output.npy'.
    Expectation: The output should match the target data within the defined relative and absolute tolerance,
                ensuring the IDFT1 forward computation is accurate.
    """
    ms.set_context(mode=mode)
    input_data = np.load('./dft/data/idft1_input.npy')
    x_re = Tensor(input_data, dtype=mstype.float32)
    x_im = x_re
    idft1_cell = idft1(shape=(4,), modes=2, compute_dtype=mstype.float32)
    ret, _ = idft1_cell((x_re, x_im))
    ret = ret.asnumpy()
    output_target = np.load('./dft/data/idft1_output.npy')
    validate_ans = compare_output(ret, output_target, rtol=FP32_RTOL, atol=FP32_ATOL)
    assert validate_ans, "The verification of dft1 forward accuracy is not successful."

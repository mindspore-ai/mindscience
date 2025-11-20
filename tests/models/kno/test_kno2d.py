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
"""KNO2D Test Case"""
import os
import random
import sys

import pytest
import numpy as np

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, ops, set_seed
from mindspore import dtype as mstype
from mindscience import KNO2D, RelativeRMSELoss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# pylint: disable=wrong-import-order,wrong-import-position

from tools import compare_output, FP16_RTOL, FP16_ATOL

# pylint: enable=wrong-import-order,wrong-import-position

set_seed(0)
np.random.seed(0)
random.seed(0)

test_data_path = '/home/workspace/mindspore_dataset/mindscience/kno2d'

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_kno2d_forward_accuracy():
    """
    Feature: KNO2D forward accuracy test
    Description: Test the forward accuracy of the KNO2D model in GRAPH_MODE.
    Expectation: The output should match the target prediction data within the specified relative and absolute
                tolerance values, ensuring the forward pass of the KNO2D model is accurate.
    """
    ms.set_device(device_target='Ascend')
    ms.set_context(mode=ms.GRAPH_MODE)
    ckpt_path = os.path.join(test_data_path, 'kno2d.ckpt')

    model = KNO2D()
    params = load_checkpoint(ckpt_path)
    load_param_into_net(model, params)
    input_data = np.load(os.path.join(test_data_path, 'kno2d_input.npy'))
    test_inputs = Tensor(input_data, mstype.float32)
    output, output_rec = model(test_inputs)
    output = output.asnumpy()
    output_rec = output_rec.asnumpy()
    load_data = np.load(os.path.join(test_data_path, 'kno2d_output.npz'))
    output_tgt, output_rec_tgt = load_data['output'], load_data['output_rec']
    validate_ans = compare_output(output, output_tgt, rtol=FP16_RTOL, atol=FP16_ATOL) and \
        compare_output(output_rec, output_rec_tgt, rtol=FP16_RTOL, atol=FP16_ATOL)
    assert validate_ans, "The verification of KNO2D forward accuracy is not successful."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_kno2d_grad_accuracy():
    """
    Feature: KNO2D gradient accuracy test
    Description: Test the accuracy of the computed gradients for the KNO2D model. 
    Expectation: The computed gradients should match the reference gradients within the specified relative and
                absolute tolerance values, ensuring the gradient calculation is accurate.
    """
    ms.set_device(device_target='Ascend')
    ms.set_context(mode=ms.GRAPH_MODE)
    ckpt_path = os.path.join(test_data_path, 'kno2d.ckpt')

    model = KNO2D()
    params = load_checkpoint(ckpt_path)
    load_param_into_net(model, params)
    input_data = np.load(os.path.join(test_data_path, 'kno2d_input.npy'))
    input_label = np.load(os.path.join(test_data_path, 'kno2d_label.npy'))
    test_inputs = Tensor(input_data, mstype.float32)
    test_label = Tensor(input_label, mstype.float32)

    loss_func = RelativeRMSELoss()
    def forward_fn(data, label):
        out, out_rec = model(data)
        loss = loss_func(out, label) + loss_func(data, out_rec)
        return loss
    grad_fn = ops.value_and_grad(
        forward_fn, None, model.trainable_params(), has_aux=False)

    _, grads = grad_fn(test_inputs, test_label)
    convert_grads = tuple(grad.asnumpy() for grad in grads)
    with np.load(os.path.join(test_data_path, 'kno2d_grads.npz')) as data:
        output_target = tuple(data[key] for key in data.files)
    validate_ans = compare_output(convert_grads, output_target, rtol=FP16_RTOL, atol=FP16_ATOL)
    assert validate_ans, "The verification of KNO2D grad accuracy is not successful."

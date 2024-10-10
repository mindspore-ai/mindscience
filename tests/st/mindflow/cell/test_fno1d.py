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
"""FNO1D Test Case"""
import os
import random
import sys

import pytest
import numpy as np

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor, ops, set_seed
from mindspore import dtype as mstype
from mindflow import FNO1D, RelativeRMSELoss, load_yaml_config
from mindflow.pde import SteadyFlowWithLoss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from common.cell import validate_checkpoint, compare_output
from common.cell import FP16_RTOL, FP16_ATOL

set_seed(0)
np.random.seed(0)
random.seed(0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_fno1d_checkpoint():
    """
    Feature: FNO1D checkpoint loading and verification
    Description: Test the consistency of the FNO1D model when loading from a saved checkpoint.
                Two FNO1D models are initialized with the same parameters, and one of them
                loads weights from the specified checkpoint located at './mindflow/cell/fno1d/ckpt/fno1d.ckpt'.
                The test input is a randomly generated tensor, and the validation checks if
                both models (one with loaded parameters) produce the same outputs.
    Expectation: The model loaded from the checkpoint should behave identically to a newly initialized
                model with the same parameters, verifying that the checkpoint restores the model's state correctly.
    """
    config = load_yaml_config('./fno1d/configs/fno1d.yaml')
    model_params = config["model"]
    ckpt_path = './fno1d/ckpt/fno1d.ckpt'

    model1 = FNO1D(in_channels=model_params["in_channels"],
                   out_channels=model_params["out_channels"],
                   n_modes=model_params["modes"],
                   resolutions=model_params["resolutions"],
                   hidden_channels=model_params["hidden_channels"],
                   n_layers=model_params["depths"],
                   projection_channels=4*model_params["hidden_channels"],
                  )

    model2 = FNO1D(in_channels=model_params["in_channels"],
                   out_channels=model_params["out_channels"],
                   n_modes=model_params["modes"],
                   resolutions=model_params["resolutions"],
                   hidden_channels=model_params["hidden_channels"],
                   n_layers=model_params["depths"],
                   projection_channels=4*model_params["hidden_channels"],
                  )

    params = load_checkpoint(ckpt_path)
    load_param_into_net(model1, params)
    test_inputs = Tensor(np.random.randn(1, 1024, 1), mstype.float32)

    validate_ans = validate_checkpoint(model1, model2, (test_inputs,))
    assert validate_ans, "The verification of FNO1D checkpoint is not successful."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fno1d_forward_accuracy(mode):
    """
    Feature: FNO1D forward accuracy test
    Description: Test the forward accuracy of the FNO1D model in both GRAPH_MODE and PYNATIVE_MODE.
                The model is initialized with parameters from './mindflow/cell/fno1d/configs/fno1d.yaml',
                and weights are loaded from the checkpoint located at './mindflow/cell/fno1d/ckpt/fno1d.ckpt'.
                The input data is loaded from './mindflow/cell/fno1d/data/fno1d_input.npy', and the output
                is compared against the expected prediction stored in './mindflow/cell/fno1d/data/fno1d_pred.npy'.
    Expectation: The output should match the target prediction data within the specified relative and absolute
                tolerance values, ensuring the forward pass of the FNO1D model is accurate.
    """
    ms.set_context(mode=mode)
    config = load_yaml_config('./fno1d/configs/fno1d.yaml')
    model_params = config["model"]
    ckpt_path = './fno1d/ckpt/fno1d.ckpt'

    model = FNO1D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
                  n_modes=model_params["modes"],
                  resolutions=model_params["resolutions"],
                  hidden_channels=model_params["hidden_channels"],
                  n_layers=model_params["depths"],
                  projection_channels=4*model_params["hidden_channels"],
                  )

    params = load_checkpoint(ckpt_path)
    load_param_into_net(model, params)
    input_data = np.load('./fno1d/data/fno1d_input.npy')
    test_inputs = Tensor(input_data, mstype.float32)
    output = model(test_inputs)
    output = output.asnumpy()
    output_target = np.load('./fno1d/data/fno1d_pred.npy')
    validate_ans = compare_output(output, output_target, rtol=FP16_RTOL, atol=FP16_ATOL)
    assert validate_ans, "The verification of FNO1D forward accuracy is not successful."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_fno1d_amp():
    """
    Feature: FNO1D AMP (Automatic Mixed Precision) accuracy test
    Description: Test the accuracy of FNO1D model with and without AMP (Automatic Mixed Precision).
                Two FNO1D models are initialized with identical parameters. The first model uses the
                default precision(float16), while the second model is set to use float32 precision for computation.
                Both models load the same checkpoint from './mindflow/cell/fno1d/ckpt/fno1d.ckpt'.
                The input data is loaded from './mindflow/cell/fno1d/data/fno1d_input.npy', and outputs
                of the two models are compared to check if they match within the specified tolerance.
    Expectation: The outputs of both models (with and without AMP) should match within the defined
                relative and absolute tolerance values, verifying that AMP does not affect the accuracy.
    """
    config = load_yaml_config('./fno1d/configs/fno1d.yaml')
    model_params = config["model"]
    ckpt_path = './fno1d/ckpt/fno1d.ckpt'

    model1 = FNO1D(in_channels=model_params["in_channels"],
                   out_channels=model_params["out_channels"],
                   n_modes=model_params["modes"],
                   resolutions=model_params["resolutions"],
                   hidden_channels=model_params["hidden_channels"],
                   n_layers=model_params["depths"],
                   projection_channels=4*model_params["hidden_channels"],
                  )

    model2 = FNO1D(in_channels=model_params["in_channels"],
                   out_channels=model_params["out_channels"],
                   n_modes=model_params["modes"],
                   resolutions=model_params["resolutions"],
                   hidden_channels=model_params["hidden_channels"],
                   n_layers=model_params["depths"],
                   projection_channels=4*model_params["hidden_channels"],
                   fno_compute_dtype=mstype.float32,
                  )

    params = load_checkpoint(ckpt_path)
    load_param_into_net(model1, params)
    load_param_into_net(model2, params)
    input_data = np.load('./fno1d/data/fno1d_input.npy')
    test_inputs = Tensor(input_data, mstype.float32)
    output1 = model1(test_inputs)
    output1 = output1.asnumpy()
    output2 = model2(test_inputs)
    output2 = output2.asnumpy()
    validate_ans = compare_output(output1, output2, rtol=FP16_RTOL, atol=FP16_ATOL)
    assert validate_ans, "The verification of FNO1D AMP accuracy is not successful."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_fno1d_grad_accuracy():
    """
    Feature: FNO1D gradient accuracy test
    Description: Test the accuracy of the computed gradients for the FNO1D model. The model is initialized
                with parameters from './mindflow/cell/fno1d/configs/fno1d.yaml' and weights are loaded
                from the checkpoint located at './mindflow/cell/fno1d/ckpt/fno1d.ckpt'. The loss function used
                is RelativeRMSELoss. The input data is loaded from './mindflow/cell/fno1d/data/fno1d_input.npy'
                and the label is from './mindflow/cell/fno1d/data/fno1d_input_label.npy'. Gradients are computed
                using MindSpore's value_and_grad and compared against the reference gradients stored in
                './mindflow/cell/fno1d/data/fno1d_grads.npz'.
    Expectation: The computed gradients should match the reference gradients within the specified relative and
                absolute tolerance values, ensuring the gradient calculation is accurate.
    """
    config = load_yaml_config('./fno1d/configs/fno1d.yaml')
    model_params = config["model"]
    ckpt_path = './fno1d/ckpt/fno1d.ckpt'

    model = FNO1D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
                  n_modes=model_params["modes"],
                  resolutions=model_params["resolutions"],
                  hidden_channels=model_params["hidden_channels"],
                  n_layers=model_params["depths"],
                  projection_channels=4*model_params["hidden_channels"],
                  )

    params = load_checkpoint(ckpt_path)
    load_param_into_net(model, params)
    input_data = np.load('./fno1d/data/fno1d_input.npy')
    input_label = np.load('./fno1d/data/fno1d_input_label.npy')
    test_inputs = Tensor(input_data, mstype.float32)
    test_label = Tensor(input_label, mstype.float32)

    problem = SteadyFlowWithLoss(
        model, loss_fn=RelativeRMSELoss())

    def forward_fn(data, label):
        loss = problem.get_loss(data, label)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, model.trainable_params(), has_aux=False)

    _, grads = grad_fn(test_inputs, test_label)
    convert_grads = tuple(grad.asnumpy() for grad in grads)
    with np.load('./fno1d/data/fno1d_grads.npz') as data:
        output_target = tuple(data[key] for key in data.files)
    validate_ans = compare_output(convert_grads, output_target, rtol=FP16_RTOL, atol=FP16_ATOL)
    assert validate_ans, "The verification of FNO1D grad accuracy is not successful."

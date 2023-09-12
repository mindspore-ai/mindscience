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
"""test derivatives"""
import pytest

import mindspore as ms
from mindspore import ops, nn, context

from sciai.common import TrainStepCell
from sciai.operators.derivatives import grad
from tests.st.sciai.test_utils.basic_nets import Net2In3Out, Net2In2Out, Net2In3OutTensor
from tests.st.sciai.test_utils.func_utils import tuple_tensor_equal


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_grad_should_success_when_tuple_in_int_out(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Net2In2Out()  # output: (a,b), input:(x, y)  a = 2x + y, b = x^2 + 4xy + 3y
    x = ops.ones((2, 3), ms.float32)
    y = ops.ones((2, 3), ms.float32)
    first_grad_net = grad(net, 1, (0, 1))  # output: (db/dx, db/dy)
    second_grad_net = grad(first_grad_net, 0, (0, 1))  # output: d2b/dx2, d2b/dxdy
    first_grad_res = first_grad_net(x, y)
    second_grad_res = second_grad_net(x, y)
    fist_grad_gt = (ms.Tensor([[6.00000000e+00, 6.00000000e+00, 6.00000000e+00],
                               [6.00000000e+00, 6.00000000e+00, 6.00000000e+00]]),
                    ms.Tensor([[7.00000000e+00, 7.00000000e+00, 7.00000000e+00],
                               [7.00000000e+00, 7.00000000e+00, 7.00000000e+00]]))
    assert tuple_tensor_equal(fist_grad_gt, first_grad_res)
    second_grad_gt = (ms.Tensor([[2.00000000e+00, 2.00000000e+00, 2.00000000e+00],
                                 [2.00000000e+00, 2.00000000e+00, 2.00000000e+00]]),
                      ms.Tensor([[4.00000000e+00, 4.00000000e+00, 4.00000000e+00],
                                 [4.00000000e+00, 4.00000000e+00, 4.00000000e+00]]))
    assert tuple_tensor_equal(second_grad_gt, second_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_grad_should_success_when_int_in_int_out(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Net2In3Out()
    x = ops.ones((2, 3), ms.float32)
    y = ops.ones((2, 3), ms.float32)
    first_grad_net = grad(net, 2, 1)
    second_grad_net = grad(first_grad_net, 0, 1)
    first_grad_res = first_grad_net(x, y)
    second_grad_res = second_grad_net(x, y)
    fist_grad_gt = ms.Tensor([[11.00000000e+00, 11.00000000e+00, 11.00000000e+00],
                              [11.00000000e+00, 11.00000000e+00, 11.00000000e+00]])
    assert tuple_tensor_equal(fist_grad_gt, first_grad_res)
    second_grad_gt = ms.Tensor([[8.00000000e+00, 8.00000000e+00, 8.00000000e+00],
                                [8.00000000e+00, 8.00000000e+00, 8.00000000e+00]])
    assert tuple_tensor_equal(second_grad_gt, second_grad_res)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_train_step_cell_should_success_when_first_loss(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Net2In3OutTensor()
    x = ops.ones((3, 2), ms.float32)
    y = ops.ones((3, 2), ms.float32)
    optim = nn.Adam(net.trainable_params(), 1e-3)
    train_cell = TrainStepCell(net, optim, True)
    _ = train_cell(x, y)

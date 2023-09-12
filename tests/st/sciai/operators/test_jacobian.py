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
"""test jacobian"""
import mindspore as ms
from mindspore import ops, context
import pytest

from sciai.operators.jacobian_weights import JacobianWeights
from tests.st.sciai.test_utils.basic_nets import Net1In1OutTensor


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_success_when_single_weight(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Net1In1OutTensor()
    x = ops.ones((100, 2), ms.float32)
    params = net.trainable_params()
    out = net(x)
    jw = JacobianWeights(net, out.shape)
    jacobian_weights = jw(x, params[0])
    assert jacobian_weights.shape == (100, 1, 1, 2)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_success_when_single_weight2(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Net1In1OutTensor()
    x = ops.ones((100, 2), ms.float32)
    params = net.trainable_params()
    out = net(x)
    jw = JacobianWeights(net, out.shape)
    jacobian_weights = jw(x, params[1])
    assert jacobian_weights.shape == (100, 1, 1)

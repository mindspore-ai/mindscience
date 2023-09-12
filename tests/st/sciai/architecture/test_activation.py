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
"""test sciai architecture activation"""
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops, context
import numpy as np
import pytest

from sciai.architecture import Swish, SReLU, get_activation, AdaptActivation


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_swish_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    swish = Swish()
    x = ops.ones((2, 3), ms.float32)
    y = swish(x)
    expected = ms.Tensor([[0.73105854, 0.73105854, 0.73105854],
                          [0.73105854, 0.73105854, 0.73105854]])
    assert mnp.isclose(y, expected, equal_nan=True).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_srelu_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    srelu = SReLU()
    x = ms.Tensor(np.array([[1.2, 0.1], [0.2, 3.2]], dtype=np.float32))
    y = srelu(x)
    expected = ms.Tensor([[0., 0.05290067],
                          [0.15216905, 0.]])
    assert mnp.isclose(y, expected, equal_nan=True).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_get_activation_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    act = get_activation("swish")
    assert act is not None
    act = get_activation("sin")
    assert act is not None
    act = get_activation("tanh")
    assert act is not None
    with pytest.raises(KeyError):
        get_activation("abc")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_adaptive_activation_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    a = ms.Tensor(0.1, ms.float32)
    net = AdaptActivation("tanh", a=a, scale=10)
    x = ops.ones((2, 3), ms.float32)
    y = net(x)
    expected = ms.Tensor([[0.7615942, 0.7615942, 0.7615942],
                          [0.7615942, 0.7615942, 0.7615942]])
    assert mnp.isclose(y, expected, equal_nan=True).all()

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
"""test sciai architecture basic block"""
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, context
from mindspore.common.initializer import Normal
import pytest

from sciai.architecture import MLP, MLPAAF, MSE, FirstOutputCell, NoArgNet, SSE, Normalize
from sciai.common import LeCunNormal, LeCunUniform
from tests.st.sciai.test_utils.basic_nets import Net2In3Out, Net2In2Out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_mlp_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    layers = [2, 10, 20, 1]
    mlp = MLP(layers, weight_init=LeCunNormal(), bias_init=Normal(sigma=1), activation=nn.Tanh())
    opt = nn.Adam(mlp.trainable_params(), 1e-3)
    cell = nn.TrainOneStepCell(mlp, opt)
    x = ops.ones((2, 2), ms.float32)
    res = cell(x)
    weights, biases = mlp.weights(), mlp.biases()
    assert res.shape == (2, 1)
    assert not res.isnan().any()
    assert len(weights) == 3
    assert len(biases) == 3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_mlpaaf_should_success_when_layerwise(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    layers = [2, 10, 20, 1]
    mlp = MLPAAF(layers, weight_init=LeCunUniform(), bias_init=Normal(sigma=1), activation=nn.Tanh(),
                 share_type='layer_wise')
    opt = nn.Adam(mlp.trainable_params(), 1e-3)
    cell = nn.TrainOneStepCell(mlp, opt)
    x = ops.ones((2, 2), ms.float32)
    res = cell(x)
    a_values = mlp.a_value()
    assert res.shape == (2, 1)
    assert not res.isnan().any()
    assert len(a_values) == 2
    assert len(set(mlp.a_list)) == 2


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_mlpaaf_should_success_when_global(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    layers = [2, 10, 20, 1]
    mlp = MLPAAF(layers, weight_init=LeCunUniform(), bias_init=Normal(sigma=1), activation=nn.Tanh(),
                 share_type='global')
    opt = nn.Adam(mlp.trainable_params(), 1e-3)
    cell = nn.TrainOneStepCell(mlp, opt)
    x = ops.ones((2, 2), ms.float32)
    res = cell(x)
    a_values = mlp.a_value()
    assert res.shape == (2, 1)
    assert not res.isnan().any()
    assert isinstance(a_values, ms.Parameter)
    assert len(set(mlp.a_list)) == 1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_mlpshortcut_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    layers = [2, 10, 20, 1]
    mlp = MLPAAF(layers, weight_init=LeCunUniform(), bias_init=Normal(sigma=1), activation=nn.Tanh(),
                 share_type='global')
    opt = nn.Adam(mlp.trainable_params(), 1e-3)
    cell = nn.TrainOneStepCell(mlp, opt)
    x = ops.ones((2, 2), ms.float32)
    res = cell(x)
    assert res.shape == (2, 1)
    assert not res.isnan().any()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_mse_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    mse = MSE()
    x = ops.ones((2, 3), ms.float32)
    res = mse(x)
    assert res.shape == ()
    assert float(res) == pytest.approx(1, 0.000001)
    assert not res.isnan().any()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_sse_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    sse = SSE()
    x = ops.ones((2, 3), ms.float32)
    res = sse(x)
    assert res.shape == ()
    assert float(res) == pytest.approx(6, 0.000001)
    assert not res.isnan().any()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_first_out_net_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Net2In3Out()
    first_output_cell = FirstOutputCell(net)
    x, y = ops.ones((2, 3), ms.float32), ops.ones((2, 3), ms.float32)
    res = first_output_cell(x, y)
    assert res.shape == ()
    assert float(res) == pytest.approx(12.0, 0.000001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_no_args_net_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    x, y = ops.ones((2, 3), ms.float32), ops.ones((2, 3), ms.float32)
    net = Net2In2Out()
    no_args_cell = NoArgNet(net, x, y)
    res1, _ = no_args_cell()
    assert res1.shape == ()
    assert float(res1) == pytest.approx(18.0, 0.000001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_normalize_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    x = ops.ones((3, 2), ms.float32)
    lb, ub = ops.Tensor([0, -0.5], ms.float32), ops.Tensor([2, 3.5], ms.float32)
    normalize = Normalize(lb, ub)
    res = normalize(x)
    assert res.shape == (3, 2)
    expected = ms.Tensor([[0., -0.25], [0., -0.25], [0., -0.25]])
    assert mnp.isclose(res, expected, equal_nan=True).all()

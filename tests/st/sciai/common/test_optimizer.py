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
"""test sciai common optimizer"""

import mindspore as ms
from mindspore import context
from mindspore import ops
import mindspore.numpy as mnp
import pytest

from sciai.architecture.basic_block import NoArgNet
from sciai.common import LbfgsOptimizer, lbfgs_train
from tests.st.sciai.test_utils.basic_nets import Net1In1OutAbs


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_comb_lbfgs_optimizer_should_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    ms.set_seed(1234)
    context.set_context(mode=mode)
    net = Net1In1OutAbs()
    x = ops.ones((3, 2), ms.float32)
    cell = NoArgNet(net, x)
    optim_lbfgs = LbfgsOptimizer(cell, list(cell.trainable_params()))
    _ = optim_lbfgs.construct(options=dict(maxiter=None, gtol=1e-6))
    loss = net(x)
    assert mnp.isclose(loss, ms.Tensor(0.01), atol=0.003, equal_nan=True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_comb_lbfgs_train_should_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    ms.set_seed(1234)
    net = Net1In1OutAbs()
    x = ops.ones((3, 2), ms.float32)
    cell = NoArgNet(net, x)
    optim_lbfgs = LbfgsOptimizer(cell, list(cell.trainable_params()))
    _ = optim_lbfgs.construct(options=dict(maxiter=None, gtol=1e-6))
    loss = net(x)
    assert mnp.isclose(loss, ms.Tensor(0.01), atol=0.03, equal_nan=True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_lbfgs_train_should_right_when_right_2(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    ms.set_seed(1234)
    net = Net1In1OutAbs()
    x = ops.ones((3, 2), ms.float32)
    lbfgs_train(net, (x,), 1000)
    loss = net(x)
    assert mnp.isclose(loss, ms.Tensor(1e-5), atol=5e-6, equal_nan=True)

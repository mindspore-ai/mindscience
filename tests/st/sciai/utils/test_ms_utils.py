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
"""test utils ms_utils"""
import sys

import mindspore as ms
from mindspore import context, ops
from mindspore.nn import Dense
import numpy as np
import pytest

from sciai.context import init_project
from sciai.utils import print_log, to_tensor, set_seed, parse_arg
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_init_project_should_success_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    args = parse_arg({})
    init_project(mode=mode, args=args)
    print_log("test")
    outputs = sys.stdout.getvalue().strip()
    assert outputs.endswith("test")
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_to_tensor_should_right_when_number_in(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    a = 1.0
    b = 2.0
    a_tensor = to_tensor(a)
    assert a_tensor == ms.Tensor(1.0)
    a_b_tensor = to_tensor((a, b))
    assert a_b_tensor == (ms.Tensor(1.0), ms.Tensor(2.0))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_to_tensor_should_right_when_tensor_in(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    a = ms.Tensor(1.0)
    b = ms.Tensor(2.0)
    a_tensor = to_tensor(a)
    assert a_tensor == ms.Tensor(1.0)
    a_b_tensor = to_tensor((a, b))
    assert a_b_tensor == (ms.Tensor(1.0), ms.Tensor(2.0))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_set_seed_should_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    seed = 1234
    set_seed(seed)
    assert getattr(ms.common.seed, "_GLOBAL_SEED") == seed
    assert np.random.get_state()[1][0] == seed


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_to_float_should_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Dense(3, 2)
    a = ops.ones((1, 3), ms.float32)
    # out_32 = net(a)
    # assert out_32.dtype == ms.float32
    net.to_float(ms.float16)
    out_16 = net(a)
    assert out_16.dtype == ms.float16

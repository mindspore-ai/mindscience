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
"""test sciai common initializer"""
import math

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import context
from mindspore.common.initializer import initializer
import numpy as np
import pytest

from sciai.common.initializer import LeCunNormal, LeCunUniform, StandardUniform, XavierTruncNormal


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_lecun_normal_should_initialize_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    np.random.seed(1234)
    tensor = initializer(LeCunNormal(), [20, 1000], ms.float32)
    assert mnp.isclose(tensor.mean(), ms.Tensor(0, dtype=ms.float32), atol=0.05, equal_nan=True)
    assert mnp.isclose(tensor.std(), ms.Tensor(1 / math.sqrt(20)), rtol=0.05, equal_nan=True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_lecun_uniform_should_initialize_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    np.random.seed(1234)
    tensor = initializer(LeCunUniform(), [20, 1000], ms.float32)
    assert mnp.isclose(tensor.mean(), ms.Tensor(0, dtype=ms.float32), atol=0.05, equal_nan=True)
    assert mnp.isclose(tensor.std(), ms.Tensor(1 / math.sqrt(20)), rtol=0.05, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_standard_uniform_should_initialize_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    np.random.seed(1234)
    tensor = initializer(StandardUniform(), [20, 1000], ms.float32)
    assert mnp.isclose(tensor.mean(), ms.Tensor(0, dtype=ms.float32), atol=0.05, equal_nan=True)
    assert mnp.isclose(tensor.std(), ms.Tensor(1 / math.sqrt(60)), rtol=0.05, equal_nan=True)  # std = |b-a|/2√3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_xavier_trunc_normal_should_initialize_right_when_right(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    np.random.seed(1234)
    tensor = initializer(XavierTruncNormal(), [20, 30], ms.float32)
    std_expected = ms.Tensor(1 / 5)
    assert mnp.isclose(tensor.mean(), ms.Tensor(0, dtype=ms.float32), atol=0.05, equal_nan=True)
    assert mnp.isclose(tensor.std(), std_expected, rtol=0.1, equal_nan=True)
    assert mnp.less_equal(tensor, 2 * std_expected).all()  # mindspore bug, when fixed, uncomment it
    assert mnp.greater_equal(tensor, -2 * std_expected).all()

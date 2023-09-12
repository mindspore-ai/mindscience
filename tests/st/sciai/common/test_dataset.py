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
"""test sciai common dataset"""

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import context
import numpy as np
import pytest

from sciai.common import DatasetGenerator, Sampler


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_dataset_should_print_double_loss_with_right_names(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    data = np.array(range(128)).reshape(-1, 2)
    dg = DatasetGenerator(data)
    assert len(dg) == 64
    assert mnp.isclose(dg[30], ms.Tensor([60., 61.]), equal_nan=True).all()
    assert mnp.isclose(dg[50], ms.Tensor([100., 101.]), equal_nan=True).all()


def u(x_):
    t = x_[:, 0:1]
    x = x_[:, 1:2]
    return np.exp(-t) * np.sin(500 * np.pi * x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_generate_dataset(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    np.random.seed(1234)
    ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
    ics_sampler = Sampler(2, ics_coords, lambda x: u(x), name='Initial Condition 1')  # pylint: disable=W0108
    x_batch, y_batch = ics_sampler.sample(10)
    # Chebyshev inequality:P((1/N Σ i=1:N Xi) - μ) >= ε) <= σ^2/N/ε^2
    # Mean(x) = (a+b)/2, Std(x) = |b-a|/sqrt(12), for uniform distribution
    mean, std = ics_sampler.normalization_constants(1000)
    x, _ = ics_sampler.fetch_minibatch(1000, mean, std)
    assert x_batch.shape == (10, 2)
    assert y_batch.shape == (10, 1)
    assert mean[0] == 0
    assert float(mean[1]) == pytest.approx(0.5, 0.01)
    assert std[0] == 0
    assert float(std[1]) == pytest.approx(1 / np.sqrt(12), 0.05)
    assert x.mean()
    assert x.std()

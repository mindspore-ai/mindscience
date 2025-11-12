# Copyright 2025 Huawei Technologies Co., Ltd
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
"""cbs testcase"""
from time import time as toc
from math import isclose
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor

from mindscience import CBS


def gen_input_2d():
    ''' prepare c_star & f_star '''
    resolution = 256

    velo = np.ones([resolution] * 2) * 1500 / 20
    mask = np.zeros_like(velo)
    omgs = np.arange(30, 40) * np.pi

    velo[64:72, 64:72] *= 1.1  # add discontinuity on velocity field
    mask[64, 64] = 1 # add one source point

    c_star = Tensor(velo / omgs.reshape(-1, 1, 1), dtype=ms.float32)
    f_star = Tensor(np.broadcast_to(mask, c_star.shape), dtype=ms.float32, const_arg=True)

    return c_star, f_star


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_solve_2d(mode):
    """
    Feature: Test CBS forward solving.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_device('Ascend')
    ms.set_context(mode=mode)
    c_star, f_star = gen_input_2d()
    warmup_steps = 1

    cbs = CBS(c_star.shape[-2:], remove_pml=False)

    # warmup runs to eliminate the initiation time
    for _ in range(warmup_steps):
        cbs(c_star, f_star)

    # run and time a complete solution process
    tic = toc()
    ur, ui, errs = cbs.solve(c_star, f_star)
    time_spent = toc() - tic
    n_steps = len(errs)
    step_time = time_spent / n_steps

    assert n_steps <= 260
    assert isclose((ur - ui).std(), 0.029885478, rel_tol=1e-3, abs_tol=1e-3), (ur - ui).std()
    assert time_spent <= 60
    assert step_time <= 0.5


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_grad_2d(mode):
    """
    Feature: Test CBS solver backward propagation.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_device('Ascend')
    ms.set_context(mode=mode)
    c_star, f_star = gen_input_2d()
    warmup_steps = 1

    cbs = CBS(c_star.shape[-2:], n_iter=5)

    def loss_func(c, f):
        ur, ui, _ = cbs(c, f)
        return ops.norm(ur) + ops.norm(ui)

    grd_func = ms.value_and_grad(loss_func, 0, None)

    # warmup runs to eliminate the initiation time
    for _ in range(warmup_steps):
        grd_func(c_star, f_star)

    # run and time a complete forward & backward process
    tic = toc()
    loss, grad = grd_func(c_star, f_star)
    time_spent = toc() - tic

    assert isclose(loss, 6.23106, rel_tol=1e-3, abs_tol=1e-3), loss
    assert isclose(grad.std(), 0.0087822, rel_tol=1e-3, abs_tol=1e-3), grad.std()
    assert time_spent <= 30


def gen_input_3d():
    ''' prepare c_star & f_star '''
    resolution = 128

    velo = np.ones([resolution] * 3) * 1500 / 20
    mask = np.zeros_like(velo)
    omgs = np.arange(30, 40) * np.pi

    velo[32:40, 32:40, 32:40] *= 1.1  # add discontinuity on velocity field
    mask[32, 32, 32] = 1 # add one source point

    c_star = Tensor(velo / omgs.reshape(-1, 1, 1, 1), dtype=ms.float32)
    f_star = Tensor(np.broadcast_to(mask, c_star.shape), dtype=ms.float32, const_arg=True)

    return c_star, f_star


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_solve_3d(mode):
    """
    Feature: Test CBS forward solving.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_device('Ascend')
    ms.set_context(mode=mode)
    c_star, f_star = gen_input_3d()
    warmup_steps = 1

    cbs = CBS(c_star.shape[-3:], remove_pml=False)

    # warmup runs to eliminate the initiation time
    for _ in range(warmup_steps):
        cbs(c_star, f_star)

    # run and time a complete solution process
    tic = toc()
    ur, ui, errs = cbs.solve(c_star, f_star)
    time_spent = toc() - tic
    n_steps = len(errs)
    step_time = time_spent / n_steps

    assert n_steps <= 180
    assert isclose((ur - ui).std(), 0.00265927, rel_tol=1e-3, abs_tol=1e-3), (ur - ui).std()
    assert time_spent <= 60
    assert step_time <= 0.5


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_grad_3d(mode):
    """
    Feature: Test CBS solver backward propagation.
    Description: None.
    Expectation: Success or throw AssertionError.
    """
    ms.set_device('Ascend')
    ms.set_context(mode=mode)
    c_star, f_star = gen_input_3d()
    warmup_steps = 1

    cbs = CBS(c_star.shape[-3:], n_iter=5)

    def loss_func(c, f):
        ur, ui, _ = cbs(c, f)
        return ops.norm(ur) + ops.norm(ui)

    grd_func = ms.value_and_grad(loss_func, 0, None)

    # warmup runs to eliminate the initiation time
    for _ in range(warmup_steps):
        grd_func(c_star, f_star)

    # run and time a complete forward & backward process
    tic = toc()
    loss, grad = grd_func(c_star, f_star)
    time_spent = toc() - tic

    assert isclose(loss, 4.43072, rel_tol=1e-3, abs_tol=1e-3), loss
    assert isclose(grad.std(), 0.00207235, rel_tol=1e-3, abs_tol=1e-3), grad.std()
    assert time_spent <= 30

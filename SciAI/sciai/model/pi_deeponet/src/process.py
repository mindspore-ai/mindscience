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
"""process"""
import os

import yaml
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops, jit

from sciai.utils import parse_arg
from .network import DataGenerator


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def rbf(x1, x2, params):
    output_scale, lengthscales = params
    diffs = mnp.expand_dims(x1 / lengthscales, 1) - mnp.expand_dims(x2 / lengthscales, 0)
    r2 = mnp.sum(diffs ** 2, axis=2)
    return output_scale * mnp.exp(-0.5 * r2)


def k_fn(x):
    return 0.001 * mnp.ones_like(x)


def v_fn(x):
    return mnp.zeros_like(x)


def g_fn(u):
    return 0.001 * u ** 2


def dg_fn(u):
    return 0.002 * u


def f_fn(x):
    return mnp.zeros_like(x)


def solve_adr(t_, nx, nt, length_scale):
    """
    Solve 1D
    u_ := (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, t_

    # Generate a GP sample
    n = 512
    gp_params = (10.0, length_scale)
    jitter = 5e-4
    x_ = mnp.linspace(xmin, xmax, n)[:, None]
    k_ = rbf(x_, x_, gp_params)
    l_ = ops.cholesky(k_ + jitter * mnp.eye(n))
    gp_sample = mnp.dot(l_, ops.normal(shape=(n,), mean=0, stddev=1))

    # Create a callable interpolation function
    def u0(x):
        return mnp.interp(x, x_.flatten(), gp_sample) * x * (1 - x)

    # Create grid
    x = mnp.linspace(xmin, xmax, nx)
    t = mnp.linspace(tmin, tmax, nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k_fn(x)
    v = v_fn(x)
    f = f_fn(x)

    # Compute finite difference operators
    d1 = mnp.eye(nx, k=1) - mnp.eye(nx, k=-1)
    d2 = -2 * mnp.eye(nx) + mnp.eye(nx, k=-1) + mnp.eye(nx, k=1)
    d3 = mnp.eye(nx - 2)
    m_ = -mnp.diag(d1 @ k) @ d1 - 4 * mnp.diag(k) @ d2
    m_bond = 8 * h2 / dt * d3 + m_[1:-1, 1:-1]
    v_bond = 2 * h * mnp.diag(v[1:-1]) @ d1[1:-1, 1:-1] + 2 * h * mnp.diag(v[2:] - v[: nx - 2])
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * d3 - m_[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = mnp.zeros((nx, nt))
    u[:, 0] = u0(x)

    # Time-stepping update
    def body_fn(i, u):
        gi = g_fn(u[1:-1, i])
        dgi = dg_fn(u[1:-1, i])
        h2dgi = mnp.diag(4 * h2 * dgi)
        a_ = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = ops.matrix_solve(a_, ops.expand_dims((b1 + b2), axis=-1)).squeeze()
        return u

    for i in range(nt - 1):
        u = body_fn(i, u)
    uu = u

    return x, t, uu, u0(x)


@jit
def generate_one_training_data(*inputs):
    """Generate one dataset"""
    p_, q_, d_t, nx, nt, length_scale = inputs
    _, _, _, u = solve_adr(d_t, nx, nt, length_scale)

    u_ic_train = mnp.tile(u, (p_, 1))

    x_ic = mnp.linspace(0, 1, p_)[:, None]
    t_ic = mnp.zeros((p_, 1))
    y_ic_train = mnp.hstack([x_ic, t_ic])

    s_ic_train = u.flatten()

    u_bc_train = mnp.tile(u, (p_, 1))

    x_bc1 = mnp.zeros((p_ // 2, 1))
    x_bc2 = mnp.ones((p_ // 2, 1))
    x_bcs = mnp.vstack((x_bc1, x_bc2))
    t_bcs = ops.uniform(shape=(p_, 1), minval=ms.Tensor(0.0), maxval=ms.Tensor(d_t))

    y_bc_train = mnp.hstack([x_bcs, t_bcs])
    s_bc_train = mnp.zeros((p_, 1))

    u_r_train = mnp.tile(u, (q_, 1))
    x_r = ops.uniform(shape=(q_, 1), minval=ms.Tensor(0.0), maxval=ms.Tensor(1.0))
    t_r = ops.uniform(shape=(q_, 1), minval=ms.Tensor(0.0), maxval=ms.Tensor(d_t))
    y_r_train = mnp.hstack([x_r, t_r])
    s_r_train = mnp.zeros((q_, 1))

    return u_ic_train, y_ic_train, s_ic_train, u_bc_train, y_bc_train, s_bc_train, u_r_train, y_r_train, s_r_train


def generate_one_test_data(t_, p_, nx_test, nt_test, length_scale):
    """Generate one test dataset"""
    x, t, uu, u = solve_adr(t_, nx_test, nt_test, length_scale)

    xx, tt = mnp.meshgrid(x, t)

    u_test = mnp.tile(u, (p_ ** 2, 1))
    y_test = mnp.hstack([xx.flatten()[:, None], tt.flatten()[:, None]])
    s_test = uu.flatten()

    return u_test, y_test, s_test


def stack_by_batch(ind_all, batch=1):
    """split and stack"""
    if batch <= 0:
        raise ValueError("batch of stack_by_batch should be greater than 0")
    input_len = len(ind_all)
    if batch > input_len:
        raise ValueError(f"batch of stack_by_batch should be less than input size {input_len}")
    if batch == 1:
        return ops.stack(ind_all, axis=0)

    step_size = input_len // batch
    idx = list(range(0, input_len, step_size))
    idx[-1] = input_len
    sub_stack_list = []
    for i in range(batch):
        sub_stack_list.append(ops.stack(ind_all[idx[i]:idx[i + 1]], axis=0))
    return ops.vstack(sub_stack_list)


def fake_vmap(func, times):
    """fake vmap"""

    def run_times(*args):
        res = [func(*args) for _ in range(times)]
        concat_res = []
        for ind in range(len(res[0])):
            ind_all = [_[ind] for _ in res]
            ind_res = stack_by_batch(ind_all, batch=6)
            concat_res.append(ind_res)
        return concat_res

    return run_times


def generate_training_data(d_t, batch_size, n_train, dtype):
    """Generate all training data"""
    length_scale_train = 0.2
    # Resolution of the solution
    nx_train, nt_train = 100, 100
    p_train, q_train = 100, 100

    u_ic_train, y_ic_train, s_ic_train, u_bc_train, y_bc_train, s_bc_train, u_r_train, y_r_train, s_r_train = \
        fake_vmap(generate_one_training_data, n_train)(p_train, q_train, d_t, nx_train, nt_train, length_scale_train)

    u_ic_train = u_ic_train.reshape(n_train * p_train, -1)
    y_ic_train = y_ic_train.reshape(n_train * p_train, -1)
    s_ic_train = s_ic_train.reshape(n_train * p_train, -1)

    u_bc_train = u_bc_train.reshape(n_train * p_train, -1)
    y_bc_train = y_bc_train.reshape(n_train * p_train, -1)
    s_bc_train = s_bc_train.reshape(n_train * p_train, -1)

    u_r_train = u_r_train.reshape(n_train * q_train, -1)
    y_r_train = y_r_train.reshape(n_train * q_train, -1)
    s_r_train = s_r_train.reshape(n_train * q_train, -1)
    ics_dataset = DataGenerator(u_ic_train, y_ic_train, s_ic_train, dtype, batch_size)
    bcs_dataset = DataGenerator(u_bc_train, y_bc_train, s_bc_train, dtype, batch_size)
    res_dataset = DataGenerator(u_r_train, y_r_train, s_r_train, dtype, batch_size)

    return ics_dataset, bcs_dataset, res_dataset


def save_loss_logs(save_data_path, model):
    np.save(f'{save_data_path}/DR_loss.npy', model.loss_log)
    np.save(f'{save_data_path}/DR_loss_res.npy', model.loss_res_log)
    np.save(f'{save_data_path}/DR_loss_ics.npy', model.loss_ics_log)
    np.save(f'{save_data_path}/DR_loss_bcs.npy', model.loss_bcs_log)


def load_loss_logs(data_path):
    loss_ics = np.load(f'{data_path}/DR_loss_ics.npy')
    loss_bcs = np.load(f'{data_path}/DR_loss_bcs.npy')
    loss_res = np.load(f'{data_path}/DR_loss_res.npy')
    return loss_ics, loss_bcs, loss_res

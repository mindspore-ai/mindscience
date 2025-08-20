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
# pylint: disable-all
import os
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.pyplot import plot

from src.metrics import l2_relative_error


def power_net_dae_plot(t, x):
    eps = 0.0001
    # parameters
    m_1, m_2, d, d_d, b = .052, .0531, .05, .005, 10.
    v_1, v_2, p_g, p_l, q_l = 1.02, 1.05, -2.0, 3.0, .1

    w1, w2, d2, d3, v3 = x

    f_1 = b * v_1 * v_2 * np.sin(d2) + b * v_2 * v3 * np.sin(d2 - d3) + p_g
    f_2 = b * v_1 * v3 * np.sin(d3) + b * v_2 * v3 * np.sin(d3 - d2) + p_l
    g = 2 * b * (v3 ** 2) - b * v3 * v_1 * np.cos(d3) - \
        b * v3 * v_2 * np.cos(d3 - d2) + q_l

    F0 = (1 / m_1) * (- d * w1 + f_1 + f_2)
    F1 = (1 / m_2) * (- d * w2 - f_1)
    F2 = (w2 - w1)
    F3 = (- w1 - (1 / d_d) * f_2)
    F4 = (- (1 / (eps * v3)) * g)

    return F0, F1, F2, F3, F4


def scipy_integrate(func, x0, h, IRK_times, method='BDF', N=0):
    """
    integrates stiff power network dynamics using scipy for N time steps of size h
    """
    v0 = 0.7   # we fix the voltage initial condition
    t_span = [0.0, h * N]
    t_sim = np.array([t_span[0]])
    for k in range(1, N + 1):
        temp = (k - 1) * h + IRK_times * h
        t_sim = np.vstack((t_sim, temp))
        t_next = np.array([k * h])
        t_sim = np.vstack((t_sim, t_next))
        del temp, t_next
    sol = solve_ivp(func, t_span, [
                    x0[0], x0[1], x0[2], x0[3], v0], method=method, t_eval=t_sim.reshape(-1,))
    y_test = sol.y
    return t_sim[1:, :], y_test[:, 1:]


def visualize(trainer, summary_dir, ode_params):
    # Test one trajectory
    x0 = [0., 0., .1, .1]
    x0_npy = np.array(x0).astype(np.float32)
    h, N, method = ode_params['h'], ode_params['N'], ode_params['method']
    y_pred = trainer.integrate(
        x0_npy, N=N, dyn_state_dim=4, model_restore_path=None)

    IRK_times = trainer.IRK_times
    t, y_eval = scipy_integrate(
        power_net_dae_plot, x0, h, IRK_times, method, N)
    print("plotting trajectory...\n")
    plot_three_bus(t, y_eval, y_pred, fname=os.path.join(
        summary_dir, 'trajectories.png'), size=25, figsize=(16, 24))

    # compute metrics for long-time integration
    l2_error = []
    for i in range(y_eval.shape[0]):
        l2_error.append(l2_relative_error(y_pred[i, ...], y_eval[i, ...]))
        print("L2relative error:", l2_error[i])

    # compute the L_2 relative error as a function of the number of time steps
    error_data = np.empty((N, 5))
    for k in range(1, N+1):
        y_pred_k = trainer.integrate(
            x0_npy, N=k, dyn_state_dim=4, model_restore_path=None)
        _, y_eval_k = scipy_integrate(
            power_net_dae_plot, x0, h, IRK_times, N=k)
        for i in range(5):
            error_data[k-1, i] = l2_relative_error(y_pred_k[i, ...], y_eval_k[i, ...])

    # plot L2 relative error for dynamic and algebraic variables
    N_vec = np.arange(1, N + 0.1)
    for k in range(5):
        fname_k = f'L2relative_error_{k}.png'
        fname = os.path.join(summary_dir, fname_k)
        plot_L2relative_error(
            N_vec, error_data[:, k], fname=fname, size=20, figsize=(8, 6))

    # save data for future use
    np.savez(os.path.join(summary_dir, "L2Relative_error"),
             N=N_vec, error=error_data)

    # regression plot for voltage
    x_line = [-.5, .5]
    y_line = [-.5, .5]
    plot_regression(y_pred[-2, ...], y_eval[-2, ...], fname=os.path.join(summary_dir,
                    'regression-voltage.png'), size=20, figsize=(8, 6), x_line=x_line, y_line=y_line)

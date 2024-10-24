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
# pylint: disable=C0103
"""ICNet dataset"""
import numpy as np

import scipy.io
from pyDOE import lhs

def read_training_data(args):
    """prepare data"""

    data_name = args.data_name
    init_steps = args.init_steps
    stop_steps = args.stop_steps
    time_steps = args.time_steps

    data = scipy.io.loadmat(data_name)

    t = np.real(data['t'][init_steps:stop_steps, :].flatten()[:, None])
    x = np.real(data['x'].flatten()[:, None])
    y = np.real(data['y'].flatten()[:, None])
    exact_u = data['U'][:, :, init_steps:stop_steps]
    exact_v = data['V'][:, :, init_steps:stop_steps]

    X, Y, T = np.meshgrid(x, y, t)

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    # Sample the datapoints at spatial domain
    N_uv_s = 3000
    idx = np.random.choice(X.shape[0]*X.shape[1], N_uv_s, replace=False)
    idx_remainder = idx%(X.shape[0])
    idx_s_y = np.floor(idx/(X.shape[0]))
    idx_s_y = idx_s_y.astype(np.int32)
    idx_idx_remainder = np.where(idx_remainder == 0)[0]
    idx_remainder[idx_idx_remainder] = X.shape[0]
    idx_s_x = idx_remainder-1

    # Sample the datapoints at temporal domain
    N_t_s = time_steps
    idx_t = np.concatenate([np.random.choice(X.shape[2], N_t_s, replace=False)])
    idx_t = idx_t.astype(np.int32)

    X1 = X[idx_s_x, idx_s_y, :]
    X2 = X1[:, idx_t]
    Y1 = Y[idx_s_x, idx_s_y, :]
    Y2 = Y1[:, idx_t]
    T1 = T[idx_s_x, idx_s_y, :]
    T2 = T1[:, idx_t]
    exact_u1 = exact_u[idx_s_x, idx_s_y, :]
    exact_u2 = exact_u1[:, idx_t]
    exact_v1 = exact_v[idx_s_x, idx_s_y, :]
    exact_v2 = exact_v1[:, idx_t]

    X_star_meas = np.hstack((X2.flatten()[:, None], Y2.flatten()[:, None], T2.flatten()[:, None]))
    u_star_meas = exact_u2.flatten()[:, None]
    v_star_meas = exact_v2.flatten()[:, None]

    # Prepare the training datasets
    N_u_train = int(N_uv_s*N_t_s)
    idx_train = np.random.choice(X_star_meas.shape[0], N_u_train, replace=False)
    X_star_train = X_star_meas[idx_train, :]
    u_star_train = u_star_meas[idx_train, :]
    v_star_train = v_star_meas[idx_train, :]

    # Collocation points
    N_f = 20000
    X_f = lb + (ub-lb)*lhs(3, N_f)
    X_f = np.vstack((X_f, X_star_train))

    # Add Noise
    noise = 0.0
    u_star_train = u_star_train + noise*np.std(u_star_train)*np.random.randn(u_star_train.shape[0],
                                                                             u_star_train.shape[1])
    v_star_train = v_star_train + noise*np.std(v_star_train)*np.random.randn(v_star_train.shape[0],
                                                                             v_star_train.shape[1])
    return X_star_train, u_star_train, v_star_train, X_f

def print_pde(lambda_uxx, lambda_uyy, w, rhs_description, ut='u_t'):
    pde = ut + ' = ' + "(%05f)" % (lambda_uxx.real) + rhs_description[0] + "\n"  + "(%05f)" % (lambda_uyy.real) + rhs_description[1] + "\n"
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f)" % (w[i].real) + rhs_description[i+2] + "\n   "
            first = False
    print(pde)

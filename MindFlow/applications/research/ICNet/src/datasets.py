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

    t = np.real(data['t_2'][:, init_steps:stop_steps].flatten()[:, None])
    x = np.real(data['x_4'].flatten()[:, None])
    exact = np.real(data['u_4_2'][:, init_steps:stop_steps]).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    # Sample the datapoints at spatial domain
    N_u_s = 256
    idx_s = np.random.choice(x.shape[0], N_u_s, replace=False)

    # Sample the datapoints at temporal domain
    N_u_t = time_steps
    idx_t = np.random.choice(t.shape[0], N_u_t, replace=False)
    idx_t = idx_t.astype(np.int32)

    X1 = X[:, idx_s]
    X2 = X1[idx_t, :]
    T1 = T[:, idx_s]
    T2 = T1[idx_t, :]
    exact1 = exact[:, idx_s]
    exact2 = exact1[idx_t, :]

    X_u_meas = np.hstack((X2.flatten()[:, None], T2.flatten()[:, None]))
    u_meas = exact2.flatten()[:, None]

    # Prepare the training datasets
    N_u_train = int(N_u_s*N_u_t)
    idx_train = np.random.choice(X_u_meas.shape[0], N_u_train, replace=False)
    X_u_train = X_u_meas[idx_train, :]
    u_train = u_meas[idx_train, :]

    # Collocation points
    N_f = 20000
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    # X_f_train = X_u_train


    # Add Noise
    noise = 0.0
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    return X_u_train, u_train, X_f_train

def print_pde(lambda_uux, w, rhs_description, ut='u_t'):
    pde = ut + '=' + "(%05f)" % (lambda_uux.real) + rhs_description[0] + "\n   "
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f)" % (w[i].real) + rhs_description[i+1] + "\n   "
            first = False
    print(pde)

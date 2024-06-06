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
"""
dataset
"""
import numpy as np
import scipy.io as sio
from scipy import sparse

def second_order_derivative_matix2d(nx, ny, dx, dy):
    """SecondOrderDerivativeMatix2d"""
    d = -2. * np.ones([nx,])
    up = np.ones([nx-1,])
    d_xx = np.diag(d) + np.diag(up, 1) + np.diag(up, -1)
    d_xx[0, -1] = 1.0
    d_xx[-1, 0] = 1.0
    d_xx = d_xx / dx**2

    d = -2. * np.ones([ny,])
    up = np.ones([ny-1,])
    d_yy = np.diag(d) + np.diag(up, 1) + np.diag(up, -1)
    d_yy[0, -1] = 1.0
    d_yy[-1, 0] = 1.0
    d_yy = d_yy / dy**2

    result = sparse.kron(d_xx, sparse.eye(ny)) + sparse.kron(sparse.eye(nx), d_yy)
    return result.tocoo()

def div_u_star_forward(p, dx, dy):
    """divUstar_forward"""
    p_shape = p.shape
    if len(p_shape) == 3:
        p = np.concatenate((p[:, -1:], p, p[:, 0:1]), axis=1)
        p = np.concatenate((p[:, :, -1:], p, p[:, :, 0:1]), axis=2)
        div_u_star = (p[:, 0:-2, 1:-1] - 2 * p[:, 1:-1, 1:-1] + p[:, 2:, 1:-1]) / dx**2 + \
                   (p[:, 1:-1, 0:-2] - 2 * p[:, 1:-1, 1:-1] + p[:, 1:-1, 2:]) / dy**2
    if len(p_shape) == 2:
        p = np.concatenate((p[-1:], p, p[0:1]), axis=0)
        p = np.concatenate((p[:, -1:], p, p[:, 0:1]), axis=1)
        div_u_star = (p[0:-2, 1:-1] - 2 * p[1:-1, 1:-1] + p[2:, 1:-1]) / dx**2 + \
                   (p[1:-1, 0:-2] - 2 * p[1:-1, 1:-1] + p[1:-1, 2:]) / dy**2
    return div_u_star

def read_training_data(t_train, dx, dy):
    """read_training_data"""
    div_u_star, p_out = [], []
    for t in t_train:
        path = f'./dataset/KolmogorovFlow/Re1000k8n512CNAB_TGVinit/KolmogorovFlowRe1000k8n512_t{t:.3f}.mat'
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        div_u = div_u_star_forward(p, dx, dy)
        div_u_star.append(div_u.astype('float32'))
        p = p / 0.01
        p_out.append(p.astype('float32'))
    div_u_star = np.stack(div_u_star, axis=0)
    p_out = np.stack(p_out, axis=0)
    return div_u_star, p_out

def read_test_data(t_test, dx, dy):
    """read_test_data"""
    div_u_star, p_out = [], []
    for t in t_test:
        path = f'./dataset/KolmogorovFlow/Re1000k8n512CNAB_TGVinit/KolmogorovFlowRe1000k8n512_t{t:.3f}.mat'
        p = sio.loadmat(path)['p']
        p = p - p.mean()
        div_u = div_u_star_forward(p, dx, dy)
        div_u_star.append(div_u.astype('float32'))
        p = p / 0.01
        p_out.append(p.astype('float32'))
    div_u_star = np.stack(div_u_star, axis=0)
    p_out = np.stack(p_out, axis=0)
    return div_u_star, p_out

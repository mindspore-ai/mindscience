# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Create dataset for training or evaluation"""
import mindspore.dataset as ds
import numpy as np
import scipy.io as scio
from pyDOE import lhs


class PINNsTrainingSet:
    """
    Training set for PINNs (Schrodinger)

    Args:
        n0 (int): number of sampled training data points for the initial condition
        nb (int): number of sampled training data points for the boundary condition
        nf (int): number of sampled training data points for the collocation points
        lb (np.array): lower bound (x, t) of domain
        ub (np.array): upper bound (x, t) of domain
        path (str): path of dataset
    """
    def __init__(self, n0, nb, nf, lb, ub, path='./data/NLS.mat'):
        data = scio.loadmat(path)
        self.n0 = n0
        self.nb = nb
        self.nf = nf
        self.lb = lb
        self.ub = ub

        # load data
        t = data['tt'].flatten()[:, None]
        x = data['x'].flatten()[:, None]
        exact = data['uu']
        exact_u = np.real(exact)
        exact_v = np.imag(exact)

        idx_x = np.random.choice(x.shape[0], self.n0, replace=False)
        self.x0 = x[idx_x, :]
        self.u0 = exact_u[idx_x, 0:1]
        self.v0 = exact_v[idx_x, 0:1]

        idx_t = np.random.choice(t.shape[0], self.nb, replace=False)
        self.tb = t[idx_t, :]

        self.x_f = self.lb + (self.ub-self.lb)*lhs(2, self.nf)

    def __getitem__(self, index):
        box_0 = np.ones((self.n0, 1), np.float32)
        box_b = np.ones((self.nb, 1), np.float32)
        box_f = np.ones((self.nf, 1), np.float32)

        x = np.vstack((self.x0.astype(np.float32),
                       self.lb[0].astype(np.float32) * box_b,
                       self.ub[0].astype(np.float32) * box_b,
                       self.x_f[:, 0:1].astype(np.float32)))
        t = np.vstack((np.array([0], np.float32) * box_0,
                       self.tb.astype(np.float32),
                       self.tb.astype(np.float32),
                       self.x_f[:, 1:2].astype(np.float32)))
        u_target = np.vstack((self.u0.astype(np.float32),
                              self.ub[0].astype(np.float32) * box_b,
                              self.lb[0].astype(np.float32) * box_b,
                              np.array([0], np.float32) * box_f))
        v_target = np.vstack((self.v0.astype(np.float32),
                              self.tb.astype(np.float32),
                              self.tb.astype(np.float32),
                              np.array([0], np.float32) * box_f))

        return np.hstack((x, t)), np.hstack((u_target, v_target))

    def __len__(self):
        return 1


def generate_pinns_training_set(n0, nb, nf, lb, ub, path='./data/NLS.mat'):
    """
    Generate training set for PINNs

    Args: see class PINNs_train_set
    """
    s = PINNsTrainingSet(n0, nb, nf, lb, ub, path)
    dataset = ds.GeneratorDataset(source=s, column_names=['data', 'label'], shuffle=False,
                                  python_multiprocessing=True)
    return dataset


def get_eval_data(path):
    """
    Get the evaluation data for Schrodinger equation.
    """
    data = scio.loadmat(path)
    t = data['tt'].astype(np.float32).flatten()[:, None]
    x = data['x'].astype(np.float32).flatten()[:, None]
    exact = data['uu']
    exact_u = np.real(exact).astype(np.float32)
    exact_v = np.imag(exact).astype(np.float32)
    exact_h = np.sqrt(exact_u**2 + exact_v**2)

    x_, t_ = np.meshgrid(x, t)

    x_star = np.hstack((x_.flatten()[:, None], t_.flatten()[:, None]))
    u_star = exact_u.T.flatten()[:, None]
    v_star = exact_v.T.flatten()[:, None]
    h_star = exact_h.T.flatten()[:, None]

    return x_star, u_star, v_star, h_star

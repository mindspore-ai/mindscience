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


class DataSetNavierStokes:
    """
    Training set for PINNs(Navier-Stokes)

    Args:
        n_train (int): amount of training data
        path (str): path of dataset
        noise (float): noise intensity, 0 for noiseless training data
        train (bool): True for training set, False for evaluation set
    """
    def __init__(self, n_train, path, noise, batch_size, train=True):
        data = scio.loadmat(path)
        self.n_train = n_train
        self.noise = noise
        self.batch_size = batch_size

        # load data
        x_star = data['X_star'].astype(np.float32)
        t_star = data['t'].astype(np.float32)
        u_star = data['U_star'].astype(np.float32)

        n = x_star.shape[0]  # number of data points per time step
        t_star_shape = t_star.shape[0]  # number of time steps

        xx = np.tile(x_star[:, 0:1], (1, t_star_shape))
        yy = np.tile(x_star[:, 1:2], (1, t_star_shape))
        tt = np.tile(t_star, (1, n)).T
        uu = u_star[:, 0, :]
        vv = u_star[:, 1, :]

        x = xx.flatten()[:, None]
        y = yy.flatten()[:, None]
        t = tt.flatten()[:, None]
        u = uu.flatten()[:, None]
        v = vv.flatten()[:, None]

        self.lb = np.array([np.min(x), np.min(y), np.min(t)], np.float32)
        self.ub = np.array([np.max(x), np.max(y), np.max(t)], np.float32)

        if train:
            idx = np.random.choice(n*t_star_shape, n_train, replace=False)  # sampled data points
            self.noise = noise
            self.x = x[idx, :]
            self.y = y[idx, :]
            self.t = t[idx, :]
            u_train = u[idx, :]
            self.u = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
            self.u = self.u.astype(np.float32)
            v_train = v[idx, :]
            self.v = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
            self.v = self.v.astype(np.float32)
        else:
            self.x = x
            self.y = y
            self.t = t
            self.u = u
            self.v = v

            p_star = data['p_star'].astype(np.float32)
            pp = p_star
            self.p = pp.flatten()[:, None]

    def __getitem__(self, index):
        ans_x = self.x[index * self.batch_size : (index + 1) * self.batch_size]
        ans_y = self.y[index * self.batch_size : (index + 1) * self.batch_size]
        ans_t = self.t[index * self.batch_size : (index + 1) * self.batch_size]
        ans_u = self.u[index * self.batch_size : (index + 1) * self.batch_size]
        ans_v = self.v[index * self.batch_size : (index + 1) * self.batch_size]
        input_data = np.hstack((ans_x, ans_y, ans_t))
        label = np.hstack((ans_u, ans_v, np.zeros([self.batch_size, 1], dtype=np.float32)))
        return input_data, label

    def __len__(self):
        return self.n_train // self.batch_size


def generate_training_set_navier_stokes(batch_size, n_train, path, noise):
    """
    Generate training set for PINNs (Navier-Stokes)

    Args:
        batch_size (int): amount of training data per batch
        n_train (int): amount of training data
        path (str): path of dataset
        noise (float): noise intensity, 0 for noiseless training data
    """
    s = DataSetNavierStokes(n_train, path, noise, batch_size, True)
    lb = s.lb
    ub = s.ub
    dataset = ds.GeneratorDataset(source=s, column_names=['data', 'label'], shuffle=True,
                                  num_parallel_workers=1)
    return dataset, lb, ub

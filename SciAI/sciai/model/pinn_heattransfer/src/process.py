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
"""Data processing"""
import os
import time
from datetime import datetime

import yaml
import numpy as np
import scipy.io
from pyDOE import lhs

from sciai.utils import print_log, parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def prepare_data(data_path, n_u=None, n_f=None):
    """Prepare data"""
    path = os.path.join(data_path, "1d_transient_100.mat")
    # Reading external data [t is 100x1, u_sol is 256x100 (solution), x is 256x1]
    data = scipy.io.loadmat(path)

    # Flatten makes [[]] into [], [:,None] makes it a column vector
    t = data['tau'].flatten()[:, None]  # T x 1
    x = data['eta'].flatten()[:, None]  # N x 1

    # Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
    u = np.real(data['theta']).T  # T x N

    # Meshing x and t in 2D (256,100)
    x_mesh, t_mesh = np.meshgrid(x, t)

    # Preparing the inputs x and t (meshed as x, t) for predictions in one single array, as x_t_star
    x_star = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))

    # Preparing the testing u_star
    u_star = u.flatten()[:, None]

    # Domain bounds (lower-bounds upperbounds) [x, t], which are here ([-1.0, 0.0] and [1.0, 1.0])
    lb = x_star.min(axis=0)
    ub = x_star.max(axis=0)
    # Getting the initial conditions (t=0)
    xx1 = np.hstack((x_mesh[0:1, :].T, t_mesh[0:1, :].T))
    uu1 = u[0:1, :].T
    # Getting the lowest boundary conditions (x=-1)
    xx2 = np.hstack((x_mesh[:, 0:1], t_mesh[:, 0:1]))
    uu2 = u[:, 0:1]
    # Getting the highest boundary conditions (x=1)
    xx3 = np.hstack((x_mesh[:, -1:], t_mesh[:, -1:]))
    uu3 = u[:, -1:]
    # Stacking them in multidimensional tensors for training (X_u_train is for now the continuous boundaries)
    x_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(x_train.shape[0], n_u, replace=False)
    x_train = x_train[idx, :]
    u_train = u_train[idx, :]

    # Generating the x and t collocation points for f, with each having a n_f size
    # We point wise add and multiply to spread the LHS over the 2D domain
    x_f_train = lb + (ub - lb) * lhs(2, n_f)

    return x, t, x_mesh, t_mesh, u, x_star, u_star, x_train, u_train, x_f_train, ub, lb


class Logger:
    """Class for logging"""

    def __init__(self, error_fn=None):
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.error_fn = error_fn
        self.model = None

    def get_epoch_duration(self):
        now = time.time()
        edur = datetime.fromtimestamp(now - self.prev_time).strftime("%S.%f")[:-5]
        self.prev_time = now
        return edur

    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

    def log_train_start(self, model):
        print_log("\nTraining started")
        print_log("================")
        self.model = model

    def log_train_opt(self, name):
        print_log(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, custom=""):
        print_log("==================")
        print_log(f"Training finished (epoch {epoch}): " +
                  f"duration = {self.get_elapsed()}  " +
                  f"error = {self.error_fn():.4e}  " + custom)

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

import yaml
import numpy as np
from mindspore import ops

from sciai.utils import to_tensor, print_log, parse_arg
from sciai.utils.ms_utils import amp2datatype
from .plot import plot_prediction_with_noise, plot_two_sigma_region, plot_probability_density_kernel_estimation


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def ode_rhs(x):
    return np.sin(np.pi * x)


class ValDataset:
    """Class of validation dataset"""

    def __init__(self, args):
        self.dtype = amp2datatype(args.amp_level)

        self.x_val, self.y_val = self._generate_val_data()
        self.x_mean = self.x_val.mean(axis=0)
        self.x_std = self.x_val.std(axis=0)

        self.x_normal = (self.x_val - self.x_mean) / self.x_std

    def _generate_val_data(self):
        x_val = np.linspace(-1, 1, 200)[:, None]
        y_val = ode_rhs(x_val)
        x_val_tensor, y_val_tensor = to_tensor((x_val, y_val), dtype=self.dtype)
        return x_val_tensor, y_val_tensor


class TrainDataset:
    """Class of training dataset"""

    def __init__(self, args):
        self.dtype = amp2datatype(args.amp_level)

        x_col, x_bound, y_bound = self._generate_train_data(args)

        x_mean = np.mean(x_col, axis=0)
        x_std = np.std(x_col, axis=0)
        y_mean = np.mean(y_bound, axis=0)
        y_std = np.std(y_bound, axis=0)
        jacobian = 1 / x_std

        x_col = (x_col - x_mean) / x_std
        x_bound = (x_bound - x_mean) / x_std

        # normalize
        self.x_col, self.x_bound, self.x_mean, self.x_std = to_tensor((x_col, x_bound, x_mean, x_std), dtype=self.dtype)
        self.y_bound, self.y_mean, self.y_std = to_tensor((y_bound, y_mean, y_std), dtype=self.dtype)
        self.jacobian = to_tensor(jacobian, dtype=self.dtype)

    @staticmethod
    def _generate_train_data(args):
        """Generate training data"""
        y_dim = args.layers_p[-1]
        n_col = args.n_col
        n_train = args.n_bound

        # position of collocation points
        x_col = np.linspace(-1, 1, n_col)[:, None]

        # position of boundary of the problem
        x_ut = np.linspace(-1, 1, 2)[:, None]
        x_bound = np.tile(x_ut, (n_train, 1))

        y_ut = ode_rhs(x_ut)
        y_bound = y_ut + 0.05 * np.random.randn(2, y_dim)
        for _ in range(n_train - 1):
            y_ut = 0.05 * np.random.randn(2, y_dim)
            y_bound = np.vstack((y_bound, y_ut))

        return x_col, x_bound, y_bound


def post_process(args, decoder, train_dataset, val_dataset):
    """Post-processing"""
    n_samples = 2000
    y_predict = ops.zeros((val_dataset.x_normal.shape[0], n_samples), dtype=train_dataset.dtype)
    z_dim = args.layers_q[-1]

    for i in range(0, n_samples):
        z_val_prior = ops.randn(y_predict.shape[0], z_dim, dtype=train_dataset.dtype)
        y_predict[:, i:i + 1] = decoder(val_dataset.x_normal, z_val_prior)
    y_predict_np = y_predict.asnumpy()

    if args.save_fig:
        x_bound = train_dataset.x_bound * train_dataset.x_std + train_dataset.x_mean
        y_bound = train_dataset.y_bound
        x_bound_np = x_bound.asnumpy()
        y_bound_np = y_bound.asnumpy()
        x_val_np = val_dataset.x_val.asnumpy()
        y_val_np = val_dataset.y_val.asnumpy()
        plot_prediction_with_noise(x_val_np, y_val_np, y_predict_np, args.figures_path)
        plot_two_sigma_region(x_bound_np, y_bound_np, x_val_np, y_val_np, y_predict_np, args.figures_path)
        plot_probability_density_kernel_estimation(args, x_val_np, y_predict_np, pos=50)
        plot_probability_density_kernel_estimation(args, x_val_np, y_predict_np, pos=150)

    mu_pred = np.mean(y_predict_np, axis=1)
    mu_pred = mu_pred[:, None]

    # np.linalg.norm does not support float16
    rel_error = np.linalg.norm(y_predict_np.astype(float) - mu_pred.astype(float), 2)
    error_u = rel_error / np.linalg.norm(y_predict_np.astype(float), 2)
    print_log('Error u: %e' % error_u)

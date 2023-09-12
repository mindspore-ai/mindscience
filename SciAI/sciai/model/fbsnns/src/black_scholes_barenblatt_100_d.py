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

"""for black scholes barenblatt case"""
import matplotlib.pyplot as plt
import numpy as np
from sciai.utils import lazy_property, print_log

from .network import BlackScholesBarenblatt
from .problem import Problem


class BlackScholesBarenblatt100D(Problem):
    """problem definition for black scholes barenblatt case"""
    def __init__(self, config):  # pylint: disable=W0235
        super().__init__(config)

    @lazy_property
    def net_class(self):
        return BlackScholesBarenblatt

    @lazy_property
    def xi(self):
        return np.array([1.0, 0.5] * int(self.dim / 2))[None, :]

    def u_exact(self, t, x):  # (N+1) x 1, (N+1) x D
        r, sigma_max = 0.05, 0.4
        return np.exp((r + sigma_max ** 2) * (self.args.terminal_time - t)) * np.sum(x ** 2, 1, keepdims=True)  # (N+1) x 1

    def test(self):
        t_test, w_test = self.fetch_minibatch()
        x_pred, y_pred = self.predict(t_test, w_test)

        y_test = np.reshape(self.u_exact(np.reshape(t_test[0:self.args.batch_size, :, :], [-1, 1]),
                                         np.reshape(x_pred[0:self.args.batch_size, :, :], [-1, self.dim])),
                            [self.args.batch_size, -1, 1])
        mse = np.mean(np.square(y_test - y_pred))
        print_log(f"MSE: {mse}")

        if self.args.save_fig:
            self.plot_y_t(t_test, y_pred, y_test)
            self.plot_relative_error(t_test, y_pred, y_test)

    def plot_y_t(self, t_test, y_pred, y_test):
        """to plot y_t"""
        samples = 5
        plt.figure()
        plt.plot(t_test[:1, :, 0].T, y_pred[:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
        plt.plot(t_test[:1, :, 0].T, y_test[:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
        plt.plot(t_test[:1, -1, 0], y_test[:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')
        plt.plot(t_test[1:samples, :, 0].T, y_pred[1:samples, :, 0].T, 'b')
        plt.plot(t_test[1:samples, :, 0].T, y_test[1:samples, :, 0].T, 'r--')
        plt.plot(t_test[1:samples, -1, 0], y_test[1:samples, -1, 0], 'ko')
        plt.plot([0], y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')
        self._plt_file(xlabel='$t$', y_label='$Y_t = u(t,X_t)$', title='100-dimensional Black-Scholes-Barenblatt',
                       file_name="BSB_Apr18_50")

    def plot_relative_error(self, t_test, y_pred, y_test):
        errors = np.sqrt((y_test - y_pred) ** 2 / y_test ** 2)
        mean_errors = np.mean(errors, 0)
        std_errors = np.std(errors, 0)
        plt.figure()
        plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
        plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
        self._plt_file(xlabel='$t$', y_label='relative error', title='100-dimensional Black-Scholes-Barenblatt',
                       file_name="BSB_Apr18_50_errors.png")

    def _plt_file(self, xlabel, y_label, title, file_name):
        plt.xlabel(xlabel)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.savefig(f'{self.args.figures_path}/{file_name}', crop=False)

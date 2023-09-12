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

"""for allen cahn case"""
import matplotlib.pyplot as plt
import numpy as np
from sciai.utils import lazy_property, print_log

from .network import AllenCahn
from .problem import Problem


class AllenCahn20D(Problem):
    """problem definition for allen cahn case"""
    def __init__(self, config):  # pylint: disable=W0235
        super().__init__(config)

    @lazy_property
    def net_class(self):
        return AllenCahn

    @lazy_property
    def xi(self):
        return np.zeros((1, self.dim))

    def test(self):
        t_test, w_test = self.fetch_minibatch()
        x_pred, y_pred = self.predict(t_test, w_test)
        samples = 5
        y_test = 1.0 / (2.0 + 0.4 * np.sum(x_pred[:, -1, :] ** 2, 1, keepdims=True))
        mse = np.mean(np.square(y_test - y_pred[:, -1, :]))
        print_log(f"MSE: {mse}")
        if self.args.save_fig:
            plt.figure()
            plt.plot(t_test[0, :, 0].T, y_pred[0, :, 0].T, 'b', label='Learned $u(t,X_t)$')
            plt.plot(t_test[1:samples, :, 0].T, y_pred[1:samples, :, 0].T, 'b')
            plt.plot(t_test[0:samples, -1, 0], y_test[0:samples, 0], 'ks', label='$Y_T = u(T,X_T)$')
            plt.plot([0], [0.30879], 'ko', label='$Y_0 = u(0,X_0)$')
            plt.xlabel('$t$')
            plt.ylabel('$Y_t = u(t,X_t)$')
            plt.title('20-dimensional Allen-Cahn')
            plt.legend()
            plt.savefig(f'{self.args.figures_path}/AC_Apr18_15.png')

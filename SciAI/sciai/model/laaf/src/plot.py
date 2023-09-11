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

"""laaf plot"""
import matplotlib.pyplot as plt
import numpy as np

from sciai.utils.plot_utils import newfig, savefig


def plot_train(figures_path, sol, x, y):
    """plot training results"""
    solution = np.concatenate(sol, axis=1)
    newfig(1.0, 1.1)
    x, y = x[0:-1].asnumpy(), y[0:-1].asnumpy()
    plt.plot(x, y, 'k-', label='Exact')
    plt.plot(x, solution[0:-1, -1], 'yx-', label='Predicted at Iter = 15000')
    plt.plot(x, solution[0:-1, 1], 'b-.', label='Predicted at Iter = 8000')
    plt.plot(x, solution[0:-1, 0], 'r--', label='Predicted at Iter = 2000')
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.legend(loc='upper left')
    savefig(f'{figures_path}/laaf')


def plot_eval(figures_path, sol, x, y):
    """plot evaluation results"""
    newfig(1.0, 1.1)
    x, y = x[0:-1].asnumpy(), y[0:-1].asnumpy()
    plt.plot(x, y, 'k-', label='Exact')
    plt.plot(x, sol.asnumpy()[0:-1], 'yx-', label='Predicted at Iter = 15001')
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.legend(loc='upper left')
    savefig(f'{figures_path}/laaf_val')

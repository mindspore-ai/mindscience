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
visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
plt.style.use('classic')

def losses_curve(loss_all, save_path=None):
    """losses_curve"""
    ylabel_name = ['weight norm', 'loss_p', 'loss_eq']
    plt.figure(figsize=[8, 8])
    plt.subplots_adjust(hspace=0.3)
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(range(loss_all.shape[0]), loss_all[:, i])
        plt.yscale('log')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(ylabel_name[i])
        plt.grid()
    if save_path is not None:
        plt.savefig(save_path+'losses_curve.png')
    plt.close()

def contourf_comparison(p, p_hat, save_path=None):
    """contourf_comparison"""
    nx = p.shape[-2]
    ny = p.shape[-1]

    dx = 2 * np.pi / nx
    dy = 2 * np.pi / ny

    xce = (np.arange(1, nx+1) - 0.5) * dx
    yce = (np.arange(1, ny+1) - 0.5) * dy
    xx, yy = np.meshgrid(xce, yce, indexing='ij')

    ae = np.abs(p_hat-p)

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    cf1 = plt.contourf(xx, yy, p, levels=50, cmap=seaborn.cm.icefire)
    plt.colorbar(shrink=0.9)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$p_{DNS}$')

    plt.subplot(1, 3, 2)
    plt.contourf(xx, yy, p_hat, levels=cf1.levels, cmap=seaborn.cm.icefire)
    plt.colorbar(shrink=0.9)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'$\hat{p}$')

    plt.subplot(1, 3, 3)
    plt.contourf(xx, yy, ae, levels=50)
    plt.colorbar(shrink=0.9)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(r'$|p_{DNS}-\hat{p}|$')

    if save_path is not None:
        plt.savefig(save_path+'contourf_comparison.png')
    plt.close()

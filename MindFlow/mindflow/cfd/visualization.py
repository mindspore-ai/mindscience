# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""visualization tools"""
import matplotlib.pyplot as plt
import numpy as np


def vis_1d(pri_var, file_name='vis.jpg'):
    """
    Visualize the 1d flow field.

    Args:
        pri_var (Tensor): The primitive variables.
        file_name (str): The name of the picture. Default: ``'vis.jpg'``.

    Supported Platforms:
        ``GPU``
    """
    data = pri_var.asnumpy()
    nx = list(data.shape)[1]
    dx = 1 / nx
    cell_centers = np.linspace(dx / 2, 1 - dx / 2, nx)
    _, ax = plt.subplots(ncols=3, figsize=(20, 7.5))
    ax[0].set_title = 'rho'
    ax[0].plot(cell_centers, data[0, :, 0, 0])

    ax[1].set_title = 'u'
    ax[1].plot(cell_centers, data[1, :, 0, 0])

    ax[2].set_title = 'p'
    ax[2].plot(cell_centers, data[4, :, 0, 0])
    plt.savefig(file_name)


def vis_2d(pri_var, file_name='vis.jpg'):
    """
    Visualize the 2d flow field.

    Args:
        pri_var (Tensor): The primitive variables.
        file_name (str): The name of the picture. Default: ``'vis.jpg'``.

    Supported Platforms:
        ``GPU``
    """
    data = pri_var.asnumpy()
    data = np.transpose(data, (0, 2, 1, 3))
    nx = list(data.shape)[1]
    ny = list(data.shape)[2]
    dx = 1 / nx
    dy = 1 / ny

    x = np.linspace(dx / 2, 1 - dx / 2, nx)
    y = np.linspace(dy / 2, 1 - dy / 2, ny)
    x_grid, y_grid = np.meshgrid(x, y)
    x = x_grid.reshape((-1, 1))
    y = y_grid.reshape((-1, 1))

    plt.figure(figsize=(16, 16))

    rho = data[0, :, :, 0].reshape(-1, 1)
    plt.subplot(2, 2, 1)
    plt.title("rho")
    plt.scatter(x, y, c=rho, cmap=plt.cm.gray, vmin=min(rho[:]), vmax=max(rho[:]))
    plt.colorbar()

    u = data[1, :, :, 0].reshape(-1, 1)
    plt.subplot(2, 2, 2)
    plt.title("velocity-x")
    plt.scatter(x, y, c=u, cmap=plt.cm.gray, vmin=min(u[:]), vmax=max(u[:]))
    plt.colorbar()

    v = data[2, :, :, 0].reshape(-1, 1)
    plt.subplot(2, 2, 3)
    plt.title("velocity-y")
    plt.scatter(x, y, c=v, cmap=plt.cm.gray, vmin=min(v[:]), vmax=max(v[:]))
    plt.colorbar()

    p = data[4, :, :, 0].reshape(-1, 1)
    plt.subplot(2, 2, 4)
    plt.title("pressure")
    plt.scatter(x, y, c=p, cmap=plt.cm.gray, vmin=min(p[:]), vmax=max(p[:]))
    plt.colorbar()

    plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
    plt.savefig(file_name)

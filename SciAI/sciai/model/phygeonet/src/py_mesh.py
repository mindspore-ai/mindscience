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
"""py mesh"""
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from sciai.utils import print_log

ERR_MESSAGE_JOINT = 'check bc nodes failed! The geometry is not closed!'
ERR_MESSAGE_PARALLEL = 'check bc nodes failed! The parallel sides do not have the same number of node!'


def to4_d_tensor(my_list):
    """to 4d tensor"""
    four_d_tensors = []
    for item in my_list:
        if len(item.shape) == 3:
            item = item.reshape([item.shape[0], 1, item.shape[1],
                                 item.shape[2]])
        four_d_tensors.append(item.float())
    return four_d_tensors


def check_geo(*args):
    """check geo"""
    left_x, left_y, right_x, right_y, low_x, low_y, up_x, up_y, tol_joint = args
    print_log('Check bc nodes!')
    if not len(left_x.shape) == len(left_y.shape) == len(right_x.shape) == \
           len(right_y.shape) == len(low_x.shape) == len(low_y.shape) == \
           len(up_x.shape) == len(up_y.shape) == 1:
        raise ValueError("check bc nodes failed! all left(right)X(Y) must be 1d vector!")
    if not np.abs(left_x[0] - low_x[0]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not np.abs(left_x[-1] - up_x[0]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not np.abs(right_x[0] - low_x[-1]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not np.abs(right_x[-1] - up_x[-1]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not np.abs(left_y[0] - low_y[0]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not np.abs(left_y[-1] - up_y[0]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not np.abs(right_y[0] - low_y[-1]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not np.abs(right_y[-1] - up_y[-1]) < tol_joint:
        raise ValueError(ERR_MESSAGE_JOINT)
    if not left_x.shape == left_y.shape == right_x.shape == right_y.shape:
        raise ValueError(ERR_MESSAGE_PARALLEL)
    if not up_x.shape == up_y.shape == low_x.shape == low_y.shape:
        raise ValueError(ERR_MESSAGE_PARALLEL)
    print_log('BC nodes pass!')


def plot_bc(ax, x, y):
    """plot bc"""
    ax.plot(x[:, 0], y[:, 0], '-o', color='orange')  # left BC
    ax.plot(x[:, -1], y[:, -1], '-o', color='red')  # right BC
    ax.plot(x[0, :], y[0, :], '-o', color='green')  # low BC
    ax.plot(x[-1, :], y[-1, :], '-o', color='blue')  # up BC
    return ax


def plot_mesh(ax, x, y, width=0.05):
    """plot mesh"""
    [ny, nx] = x.shape
    for j in range(0, nx):
        ax.plot(x[:, j], y[:, j], color='black', linewidth=width)
    for i in range(0, ny):
        ax.plot(x[i, :], y[i, :], color='black', linewidth=width)
    return ax


def set_axis_label(ax, label_type):
    """set axis label"""
    if label_type == 'p':
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
    elif label_type == 'r':
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')
    else:
        raise ValueError('The axis type only can be reference or physical')


def elliptic_map(x, y, h, tol):
    """elliptic map"""
    eps = 2.2e-16
    if not x.shape == y.shape:
        raise ValueError(f"check bc nodes failed!, The x y shapes do not have match each other!")
    ny, nx = x.shape
    a = np.ones([ny - 2, nx - 2])
    b, c = a, a
    err_list = []
    err = 0
    for _ in range(50000):
        x_ = (a * (x[2:, 1:-1] + x[0:-2, 1:-1]) + c * (x[1:-1, 2:] + x[1:-1, 0:-2]) -
              b / 2 * (x[2:, 2:] + x[0:-2, 0:-2] - x[2:, 0:-2] - x[0:-2, 2:])) / 2 / (a + c)
        y_ = (a * (y[2:, 1:-1] + y[0:-2, 1:-1]) + c * (y[1:-1, 2:] + y[1:-1, 0:-2]) -
              b / 2 * (y[2:, 2:] + y[0:-2, 0:-2] - y[2:, 0:-2] - y[0:-2, 2:])) / 2 / (a + c)
        err = np.max(np.max(np.abs(x[1:-1, 1:-1] - x_))) + np.max(np.max(np.abs(y[1:-1, 1:-1] - y_)))
        err_list.append(err)
        x[1:-1, 1:-1] = x_
        y[1:-1, 1:-1] = y_
        a = ((x[1:-1, 2:] - x[1:-1, 0:-2]) / 2 / h) ** 2 + \
            ((y[1:-1, 2:] - y[1:-1, 0:-2]) / 2 / h) ** 2 + eps
        b = (x[2:, 1:-1] - x[0:-2, 1:-1]) / 2 / h * \
            (x[1:-1, 2:] - x[1:-1, 0:-2]) / 2 / h + \
            (y[2:, 1:-1] - y[0:-2, 1:-1]) / 2 / h * \
            (y[1:-1, 2:] - y[1:-1, 0:-2]) / 2 / h + eps
        c = ((x[2:, 1:-1] - x[0:-2, 1:-1]) / 2 / h) ** 2 + \
            ((y[2:, 1:-1] - y[0:-2, 1:-1]) / 2 / h) ** 2 + eps
        if err < tol:
            print_log('The mesh generation reaches convergence!')
            break
    else:
        print_log('The mesh generation not reaches convergence within 50000 iterations! The current residual is ')
        print_log(err)
    return x, y


def gen_e2vcg(x):
    """generate e2vcg"""
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    nnx = x.shape[1]
    e2vcg = np.zeros([4, nelem])
    for j in range(nelemy):
        for i in range(nelemx):
            e2vcg[:, j * nelemx + i] = np.array(
                [j * nnx + i, j * nnx + i + 1, (j + 1) * nnx + i, (j + 1) * nnx + i + 1])
    return e2vcg.astype('int')


def visualize2d(ax, xyu, colorbar_position='vertical', color_limit=None):
    """visualize 2d"""
    x, y, u = xyu
    xdg0 = np.vstack([x.flatten(order='C'), y.flatten(order='C')])
    udg0 = u.flatten(order='C')
    idx = np.asarray([0, 1, 3, 2])
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    e2vcg0 = gen_e2vcg(x)
    udg_ref = udg0[e2vcg0]
    cmap = cm.coolwarm
    polygon_list = []
    for i in range(nelem):
        polygon_ = Polygon(xdg0[:, e2vcg0[idx, i]].T)
        polygon_list.append(polygon_)
    polygon_ensemble = PatchCollection(polygon_list, cmap=cmap, alpha=1)
    polygon_ensemble.set_edgecolor('face')
    polygon_ensemble.set_array(np.mean(udg_ref, axis=0))
    if color_limit is not None:
        polygon_ensemble.set_clim(color_limit)
    ax.add_collection(polygon_ensemble)
    ax.set_xlim(np.min(xdg0[0, :]), np.max(xdg0[0, :]))
    ax.set_ylim(np.min(xdg0[1, :]), np.max(xdg0[1, :]))
    cbar = plt.colorbar(polygon_ensemble, orientation=colorbar_position)
    return ax, cbar


class HcubeMesh:
    """HcubeMesh"""

    def __init__(self, left_x, left_y, right_x, right_y, low_x, low_y, up_x, up_y,
                 h, plot_flag=False, save_flag=False, save_dir='./figures/mesh.pdf', tol_mesh=1e-8, tol_joint=1e-6):
        self.h = h
        check_geo(left_x, left_y, right_x, right_y, low_x, low_y, up_x, up_y, tol_joint)
        # Extract discretization info
        self.ny = left_x.shape[0]
        self.nx = up_x.shape[0]
        # Preallocate the physical domain. Left->Right->Low->Up
        self.x, self.y = np.zeros([self.ny, self.nx]), np.zeros([self.ny, self.nx])
        self.x[:, 0], self.y[:, 0] = left_x, left_y
        self.x[:, -1], self.y[:, -1] = right_x, right_y
        self.x[0, :], self.y[0, :] = low_x, low_y
        self.x[-1, :], self.y[-1, :] = up_x, up_y
        self.x, self.y = elliptic_map(self.x, self.y, self.h, tol_mesh)
        # Define the ref domain
        eta, xi = np.meshgrid(np.linspace(0, self.ny - 1, self.ny), np.linspace(0, self.nx - 1, self.nx),
                              sparse=False, indexing='ij')
        self.eta, self.xi = eta * h, xi * h
        self.plot_fig(plot_flag, save_dir, save_flag)
        self.dxdxi = (self.x[1:-1, 2:] - self.x[1:-1, 0:-2]) / 2 / self.h
        self.dydxi = (self.y[1:-1, 2:] - self.y[1:-1, 0:-2]) / 2 / self.h
        self.dxdeta = (self.x[2:, 1:-1] - self.x[0:-2, 1:-1]) / 2 / self.h
        self.dydeta = (self.y[2:, 1:-1] - self.y[0:-2, 1:-1]) / 2 / self.h
        self.j = self.dxdxi * self.dydeta - self.dxdeta * self.dydxi
        self.jinv = 1 / self.j
        dxdxi_ho_in, dydxi_ho_in = self.d_dxi_ho_in(self.x, self.h), self.d_dxi_ho_in(self.y, self.h)
        dxdxi_ho_left, dydxi_ho_left = self.d_dxi_ho_left(self.x, self.h), self.d_dxi_ho_left(self.y, self.h)
        dxdxi_ho_right, dydxi_ho_right = self.d_dxi_ho_right(self.x, self.h), self.d_dxi_ho_right(self.y, self.h)

        dxdeta_ho_in, dydeta_ho_in = self.d_deta_ho_in(self.x, self.h), self.d_deta_ho_in(self.y, self.h)
        dxdeta_ho_low, dydeta_ho_low = self.d_deta_ho_low(self.x, self.h), self.d_deta_ho_low(self.y, self.h)
        dxdeta_ho_up, dydeta_ho_up = self.d_deta_ho_up(self.x, self.h), self.d_deta_ho_up(self.y, self.h)

        self.dxdxi_ho, self.dydxi_ho, self.dxdeta_ho, self.dydeta_ho \
            = np.zeros(self.x.shape), np.zeros(self.y.shape), np.zeros(self.x.shape), np.zeros(self.y.shape)
        self.dxdxi_ho[:, :2], self.dydxi_ho[:, :2], self.dxdeta_ho[:2, :], self.dydeta_ho[:2, :] \
            = dxdxi_ho_left[:, :2], dydxi_ho_left[:, :2], dxdeta_ho_low[:2, :], dydeta_ho_low[:2, :]
        self.dxdxi_ho[:, 2:-2], self.dydxi_ho[:, 2:-2], self.dxdeta_ho[2:-2, :], self.dydeta_ho[2:-2, :] \
            = dxdxi_ho_in, dydxi_ho_in, dxdeta_ho_in, dydeta_ho_in
        self.dxdxi_ho[:, -2:], self.dydxi_ho[:, -2:], self.dxdeta_ho[-2:, :], self.dydeta_ho[-2:, :] \
            = dxdxi_ho_right[:, -2:], dydxi_ho_right[:, -2:], dxdeta_ho_up[-2:, :], dydeta_ho_up[-2:, :]

        self.j_ho = self.dxdxi_ho * self.dydeta_ho - self.dxdeta_ho * self.dydxi_ho
        self.jinv_ho = 1 / self.j_ho

    def plot_fig(self, plot_flag, save_dir, save_flag):
        """plot fig"""
        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        plot_bc(ax, self.x, self.y)
        plot_mesh(ax, self.x, self.y)
        set_axis_label(ax, 'p')
        ax.set_aspect('equal')
        ax.set_title('Physics Domain Mesh')
        ax = plt.subplot(1, 2, 2)
        plot_bc(ax, self.xi, self.eta)
        plot_mesh(ax, self.xi, self.eta)
        set_axis_label(ax, 'r')
        ax.set_aspect('equal')
        ax.set_title('Reference Domain Mesh')
        fig.tight_layout(pad=1)
        if save_flag:
            plt.savefig(save_dir, bbox_inches='tight')
        if plot_flag:
            plt.show()
        plt.close(fig)

    def d_dxi_ho_in(self, e, h):
        return (-e[:, 4:] + 8 * e[:, 3:-1] - 8 * e[:, 1:-3] + e[:, :-4]) / 12 / h

    def d_dxi_ho_left(self, e, h):
        return (-11 * e[:, :-3] + 18 * e[:, 1:-2] - 9 * e[:, 2:-1] + 2 * e[:, 3:]) / 6 / h

    def d_dxi_ho_right(self, e, h):
        return (11 * e[:, 3:] - 18 * e[:, 2:-1] + 9 * e[:, 1:-2] - 2 * e[:, :-3]) / 6 / h

    def d_deta_ho_in(self, e, h):
        return (-e[4:, :] + 8 * e[3:-1, :] - 8 * e[1:-3, :] + e[:-4, :]) / 12 / h

    def d_deta_ho_low(self, e, h):
        return (-11 * e[:-3, :] + 18 * e[1:-2, :] - 9 * e[2:-1, :] + 2 * e[3:, :]) / 6 / h

    def d_deta_ho_up(self, e, h):
        return (11 * e[3:, :] - 18 * e[2:-1, :] + 9 * e[1:-2, :] - 2 * e[:-3, :]) / 6 / h

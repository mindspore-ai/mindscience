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
"""plotting results"""
import math

import numpy as np
import scipy.io
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)


def get_matrix_c(data):
    """reshape to 2D"""
    x = data['x_star']
    xx = np.unique(x)
    col_num = xx.shape[0]
    total = x.shape[0]
    row_num = total / col_num
    return col_num, row_num


def plot_prediction(row_num, col_num, data_real, data_imag, figures_path):
    """Plot predictions"""
    row_num, col_num = int(row_num), int(col_num)
    u_real_2d = np.zeros([row_num, col_num])
    start = 0
    plt.subplot(2, 1, 1)
    for i in range(col_num):
        end = int(row_num + start)
        u_real_2d[:, i] = data_real[start:end, 0]
        start = end
    plt.imshow(u_real_2d)
    plt.title("Prediction")

    plt.subplot(2, 1, 2)
    u_imag_2d = np.zeros([row_num, col_num])
    start = 0
    for i in range(col_num):
        end = int(row_num + start)
        u_imag_2d[:, i] = data_imag[start:end, 0]
        start = end
    plt.imshow(u_imag_2d)
    plt.savefig(f'{figures_path}/prediction.png')


def plot_numerical(row_num, col_num, data_real, data_imag, figures_path):
    """Plot numerical"""
    row_num, col_num = int(row_num), int(col_num)
    u_real_2d = np.zeros([row_num, col_num])
    start = 0
    plt.subplot(2, 1, 1)
    for i in range(col_num):
        end = int(row_num + start)
        u_real_2d[:, i] = data_real[start:end, 0]
        start = end
    plt.imshow(u_real_2d)
    plt.title("Numerical Solution")

    plt.subplot(2, 1, 2)
    u_imag_2d = np.zeros([row_num, col_num])
    start = 0
    for i in range(col_num):
        end = int(row_num + start)
        u_imag_2d[:, i] = data_imag[start:end, 0]
        start = end
    plt.imshow(u_imag_2d)
    plt.savefig(f'{figures_path}/numerical_solution.png')


def plot_losses(args):
    """Plot losses"""
    data_loss_adam = scipy.io.loadmat(f'{args.results_path}/loss_adam_100000_{args.amp_level}.mat')
    iter_adam = data_loss_adam['misfit']
    losses = np.transpose(iter_adam)

    _, ax = plt.subplots()

    iters = np.linspace(1, losses.size, losses.size)
    ax.plot(iters, losses)
    ax.set(ylim=(math.pow(10, -1), math.pow(10, 2)))
    ax.set(xlim=(0, losses.size))
    plt.title("losses")
    plt.xlabel("iterations")
    plt.ylabel("losses")
    plt.yscale("log")
    plt.savefig(f'{args.figures_path}/losses_adam.png')


def plot_result(args, file_real="u_real_pred_adam.mat", file_imag="u_imag_pred_adam.mat"):
    """Plot results"""
    data_real_pred = scipy.io.loadmat(f'{args.results_path}/{file_real}')
    data_imag_pred = scipy.io.loadmat(f'{args.results_path}/{file_imag}')
    data_coord = scipy.io.loadmat(f'{args.load_data_path}/Marmousi_3Hz_singlesource_ps.mat')

    u_real_star = data_coord['U_real']
    u_imag_star = data_coord['U_imag']

    u_real_pred = data_real_pred['u_real_pred']
    u_imag_pred = data_imag_pred['u_imag_pred']

    col_num, row_num = get_matrix_c(data_coord)

    plot_prediction(row_num, col_num, u_real_pred, u_imag_pred, args.figures_path)
    plot_numerical(row_num, col_num, u_real_star, u_imag_star, args.figures_path)

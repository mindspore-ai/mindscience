# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
utils for training and testing
"""
import os
import logging
import numpy as np
import mindspore as ms
from scipy.spatial import distance
from matplotlib import pyplot as plt


def calculate_potential(pos, charges, gaussian_width, grid):
    """calculate potential"""
    if pos.ndim == 2:
        np.unsqueeze(pos, 0)
    potentials = np.zeros((len(pos), len(grid)))

    for j in range(pos.shape[1]):
        atom_dist = distance.cdist(pos[:, j, :], grid)
        potentials += np.multiply(np.exp(-np.power(atom_dist, 2) / (2 * (gaussian_width ** 2))), charges[j])
    return potentials


def calculate_l1_error(workspace_dir, test_dir, energy_type, run_id, data_dir, metric):
    """Calculated coefficients error and maes"""
    maes = []
    error_pred = np.load(os.path.join(workspace_dir, test_dir, 'errors_pred_' + energy_type +
                                      '_' + str(run_id) + '.npy'), allow_pickle=True)
    mae = np.mean(error_pred)
    maes.append(mae)
    coefficients = np.load(os.path.join(data_dir, test_dir, 'dft_densities.npy'), mmap_mode="r")

    coefficients_pred = np.load(
        os.path.join(workspace_dir, test_dir, 'coefficients_pred_' + str(run_id) + '.npy'))

    if metric[0] != 'cosine':
        coefficients_error = np.mean(np.linalg.norm(coefficients[:len(coefficients_pred)] - coefficients_pred,
                                                    axis=1, ord=metric[1]))
    else:
        a_norm = np.linalg.norm(coefficients, axis=1, keepdims=True)
        b_norm = np.linalg.norm(coefficients_pred, axis=1, keepdims=True)
        similarity = np.dot(coefficients, coefficients_pred.T) / (a_norm * b_norm)
        dist = 1. - np.diagonal(similarity)
        coefficients_error = np.mean(dist)
    return coefficients_error, maes


def save_mae_chart(energy_types, maes_lists, n_trainings, workspace_dir, opt_type, max_iter):
    """Draw the mae diagram and save """
    plt.figure(dpi=400)
    for i, energy_type in enumerate(energy_types):
        if energy_type == 'cc':
            label = r'$E^{CC}_{ML}[n^{DFT}_{ML}]$'
        elif energy_type == 'dft':
            label = r'$E^{DFT}_{ML}[n^{DFT}_{ML}]$'
        elif energy_type == 'diff':
            label = r'$E^{CC}_{\Delta-DFT}[n^{DFT}_{ML}]$'
        maes_list = maes_lists[i]
        y = np.mean(maes_list, axis=1)
        y1 = np.min(maes_list, axis=1)
        y2 = np.max(maes_list, axis=1)
        plt.plot(n_trainings, y, 'o-', linewidth=2, label=label)
        plt.fill_between(n_trainings, y1, y2, alpha=0.1)
        for x_val, y_val in zip(n_trainings, y):
            plt.text(x_val, y_val, f'{y_val:.4f}', ha='right', va='bottom')
    plt.xticks([200, 400, 600, 800])
    plt.xlabel('Number of training samples')
    plt.ylabel(r'MAE (kal$\cdot$mol$^{-1}$)')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(workspace_dir, opt_type + 'krr_MAE_{}.png'.format(max_iter)))
    plt.close()


def save_error_chart(energy_types, coefficients_errors_lists, n_trainings, workspace_dir, opt_type, max_iter, metric):
    """Draw the error diagram and save """
    plt.figure(dpi=400)
    for i, energy_type in enumerate(energy_types):
        if energy_type == 'cc':
            label = r'$E^{CC}_{ML}[n^{DFT}_{ML}]$'
        elif energy_type == 'dft':
            label = r'$E^{DFT}_{ML}[n^{DFT}_{ML}]$'
        elif energy_type == 'diff':
            label = r'$E^{CC}_{\Delta-DFT}[n^{DFT}_{ML}]$'
        coefs_errors_list = coefficients_errors_lists[i]
        y = np.mean(coefs_errors_list, axis=1)
        y1 = np.min(coefs_errors_list, axis=1)
        y2 = np.max(coefs_errors_list, axis=1)
        plt.plot(n_trainings, y, 'o-', linewidth=2, label=label)
        plt.fill_between(n_trainings, y1, y2, alpha=0.1)
        for x_val, y_val in zip(n_trainings, y):
            plt.text(x_val, y_val, f'{y_val:.4f}', ha='right', va='bottom')
    plt.xticks([200, 400, 600, 800])
    plt.xlabel('Number of training samples')
    plt.ylabel('Mean coef error of {} metric'.format(metric[0]))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(workspace_dir, opt_type + 'krr_coefs_{}_errors_{}.png'.format(metric[0], max_iter)),
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()


def write_error(energy: ms.Tensor, energy_pred: ms.Tensor):
    """
    logging.info correlation coefficients
    """
    logging.info('能量:')
    logging.info('相关系数(CC): %s', np.corrcoef(energy.asnumpy().T, energy_pred.asnumpy().T)[0][1])
    logging.info('均方根误差(RootMSE): %s ', (energy - energy_pred).pow(2).mean(axis=0).mean().sqrt().asnumpy())
    logging.info('平均绝对误差(MAE): %s', (energy.astype(ms.float32) - energy_pred).abs().mean(axis=0).mean().asnumpy())
    logging.info('最大绝对误差(Max MAE): %s', ((energy.astype(ms.float32) - energy_pred).abs()).max().asnumpy())


def calculate_positions(cat_positions):
    """Calculates the maximum and minimum values of positions."""
    max_pos = cat_positions.max(axis=0)
    max_pos = max_pos.max(axis=0)
    min_pos = cat_positions.min(axis=0)
    min_pos = min_pos.min(axis=0)
    max_pos = np.ceil(max_pos) + 1
    min_pos = np.floor(min_pos) - 1
    max_range = max_pos - min_pos
    return max_pos, min_pos, max_range


def calculate_grid(grid_range):
    """calculate grid"""
    yy, xx, zz = np.meshgrid(grid_range[0], grid_range[1], grid_range[2])
    xx = xx.flatten()[:, np.newaxis]
    yy = yy.flatten()[:, np.newaxis]
    zz = zz.flatten()[:, np.newaxis]
    grid = np.concatenate((xx, yy, zz), axis=1)
    return grid

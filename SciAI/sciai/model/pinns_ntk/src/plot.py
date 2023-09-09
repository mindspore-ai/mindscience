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
"""Plotting functions"""
import matplotlib.pyplot as plt
import numpy as np


def plot_figures(x_star, model, u_pred, u_star, figures_path):
    """Figures plotting"""
    plot_point_wise_error(figures_path, u_pred, u_star, x_star)
    # Create empty lists for storing the eigenvalues of NTK
    lambda_k_log = []
    lambda_k_uu_log = []
    lambda_k_rr_log = []
    # Restore the NTK
    k_uu_list = model.k_uu_log
    k_ur_list = model.k_ur_log
    k_rr_list = model.k_rr_log
    k_list = []
    k_uu_list_len = len(k_uu_list)
    for i in range(k_uu_list_len):
        k_uu = k_uu_list[i]
        k_ur = k_ur_list[i]
        k_rr = k_rr_list[i]

        k = np.concatenate([np.concatenate([k_uu, k_ur], axis=1),
                            np.concatenate([k_ur.T, k_rr], axis=1)], axis=0)
        k_list.append(k)

        # Compute eigenvalues
        lambda_k, _ = np.linalg.eig(k)
        lambda_k_uu, _ = np.linalg.eig(k_uu)
        lambda_k_rr, _ = np.linalg.eig(k_rr)

        # Sort in descresing order
        lambda_k = np.sort(np.real(lambda_k))[::-1]
        lambda_k_uu = np.sort(np.real(lambda_k_uu))[::-1]
        lambda_k_rr = np.sort(np.real(lambda_k_rr))[::-1]

        # Store eigenvalues
        lambda_k_log.append(lambda_k)
        lambda_k_uu_log.append(lambda_k_uu)
        lambda_k_rr_log.append(lambda_k_rr)
    plot_eigenvalues(lambda_k_log, lambda_k_rr_log, lambda_k_uu_log, figures_path)
    # Change of the NTK
    plot_ntk_change(k_list, figures_path)

    # Restore the list weights and biases
    plot_weight_change(model, figures_path)


def plot_ntk_change(k_list, figures_path):
    """NTK change plotting"""
    ntk_change_list = []
    k0 = k_list[0]
    for k in k_list:
        diff = np.linalg.norm(k - k0) / np.linalg.norm(k0)
        ntk_change_list.append(diff)
    plt.figure(figsize=(6, 5))
    plt.plot(ntk_change_list)
    plt.xlabel('iterations')
    plt.ylabel('NTK change')
    plt.title('NTK change')
    plt.savefig(f"{figures_path}/NTK_change.png")


def plot_weight_change(model, figures_path):
    """Weight change plotting"""
    weights_log = model.weights_log
    biases_log = model.biases_log
    weights_0 = weights_log[0]
    biases_0 = biases_log[0]
    # Norm of the weights at initialization
    weights_init_norm = compute_weights_norm(weights_0, biases_0)
    weights_change_list = []
    n = len(weights_log)
    for k in range(n):
        weights_diff = compute_weights_diff(weights_log[k], weights_log[0])
        biases_diff = compute_weights_diff(biases_log[k], biases_log[0])

        weights_diff_norm = compute_weights_norm(weights_diff, biases_diff)
        weights_change = weights_diff_norm / weights_init_norm
        weights_change_list.append(weights_change)
    plt.figure(figsize=(6, 5))
    plt.plot(weights_change_list)
    plt.xlabel('iterations')
    plt.ylabel('weights change')
    plt.title('weights change')
    plt.savefig(f"{figures_path}/weights_change.png")


def plot_point_wise_error(figures_path, u_pred, u_star, x_star):
    """Point-wise error plotting"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_star, u_star, label='Exact')
    plt.plot(x_star, u_pred, '--', label='Predicted')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.plot(x_star, np.abs(u_star - u_pred), label='Error')
    plt.yscale('log')
    plt.xlabel('$x$')
    plt.ylabel('Point-wise error')
    plt.tight_layout()
    plt.savefig(f"{figures_path}/y_point_wise_error.png")


def plot_eigenvalues(lambda_k_log, lambda_k_rr_log, lambda_k_uu_log, figures_path):
    """Eigenvalue plotting"""
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    for i in range(1, len(lambda_k_log), 10):
        plt.plot(lambda_k_log[i], '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Eigenvalues of ${k}$')
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    for i in range(1, len(lambda_k_uu_log), 10):
        plt.plot(lambda_k_uu_log[i], '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Eigenvalues of ${k}_{uu}$')
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    for i in range(1, len(lambda_k_log), 10):
        plt.plot(lambda_k_rr_log[i], '--')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Eigenvalues of ${k}_{rr}$')
    plt.tight_layout()
    plt.savefig(f"{figures_path}/eigenvalues.png")


def compute_weights_diff(weights_1, weights_2):
    """Weights difference computing"""
    weights = []
    n = len(weights_1)
    for k in range(n):
        weight = weights_1[k] - weights_2[k]
        weights.append(weight)
    return weights


def compute_weights_norm(weights, biases):
    """Weights norm plotting"""
    norm = 0
    for w in weights:
        norm = norm + np.sum(np.square(w))
    for b in biases:
        norm = norm + np.sum(np.square(b))
    norm = np.sqrt(norm)
    return norm


def plot_loss(args, model):
    """Loss plotting"""
    plt.figure(figsize=(6, 5))
    plt.plot(model.loss_bcs_log, label=r'$\mathcal{L}_{r}$')
    plt.plot(model.loss_res_log, label=r'$\mathcal{L}_{b}$')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.figures_path}/loss.png")

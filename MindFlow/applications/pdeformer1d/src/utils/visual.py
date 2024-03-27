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
r"""This module provides visualization functions."""
import os
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def plot_1d(u_label: NDArray[float],
            u_predict: NDArray[float],
            file_name: str,
            title: str = "",
            save_dir: Optional[str] = None) -> None:
    r"""
    Plot the 1D image containing the label and the prediction.

    Args:
        u_label (numpy.ndarray): The label of the 1D image.
        u_predict (numpy.ndarray): The prediction of the 1D image.
        file_name (str): The name of the saved file.
        title (str): The title of the plot. Default: "".
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """

    plt.rcParams['figure.figsize'] = [6.4, 4.8]
    fig = plt.figure()
    ax_ = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_.plot(u_label, ls='-', c="blue", label="Reference")
    ax_.plot(u_predict, ls=':', c="red", label="Predict")
    ax_.legend()

    fig.suptitle(title)
    fig.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_2d(u_label: NDArray[float],
            u_predict: NDArray[float],
            file_name: str,
            title: str = "",
            save_dir: Optional[str] = None) -> None:
    r"""
    Plot the 2D image containing the label and the prediction.

    Args:
        u_label (numpy.ndarray): The label of the 2D image.
        u_predict (numpy.ndarray): The prediction of the 2D image.
        file_name (str): The name of the saved file.
        title (str): The title of the plot. Default: "".
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    u_error = np.abs(u_label - u_predict)

    vmin_u = u_label.min()
    vmax_u = u_label.max()
    vmin_error = u_error.min()
    vmax_error = u_error.max()
    vmin = [vmin_u, vmin_u, vmin_error]
    vmax = [vmax_u, vmax_u, vmax_error]

    sub_titles = ["Reference", "Predict", "Error"]

    plt.rcParams['figure.figsize'] = [9.6, 3.2]
    fig = plt.figure()
    gs_ = gridspec.GridSpec(2, 6)
    slice_ = [gs_[0:2, 0:2], gs_[0:2, 2:4], gs_[0:2, 4:6]]
    for i, data in enumerate([u_label, u_predict, u_error]):
        ax_ = fig.add_subplot(slice_[i])

        img = ax_.imshow(
            data.T, vmin=vmin[i],
            vmax=vmax[i],
            cmap=plt.get_cmap("jet"),
            origin='lower')

        ax_.set_title(sub_titles[i], fontsize=10)
        plt.xticks(())
        plt.yticks(())

        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(ax_)
        width = axes_size.AxesY(ax_, aspect=1 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cb_ = plt.colorbar(img, cax=cax)
        cb_.ax.tick_params(labelsize=6)

    gs_.tight_layout(fig, pad=1.0, w_pad=3.0, h_pad=1.0)

    fig.suptitle(title, y=1.1)
    fig.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_2dxn(u_list: list,
              file_name: str,
              title: str = "",
              save_dir: Optional[str] = None) -> None:
    r"""
    Plot the images for partially-observed inverse problems.

    Args:
        u_list (list): A list of numpy.ndarrays containing the label, noisy (optional),
            observed (optional), prediction1, and prediction2.
        file_name (str): The name of the saved file.
        title (str): The title of the plot. Default: "".
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    n_plots = len(u_list)
    u_label = u_list[0]
    vmin_u = u_label.min()
    vmax_u = u_label.max()

    if n_plots == 3:
        sub_titles = ["Reference", "Predict1", "Predict2"]
    elif n_plots == 4:
        sub_titles = ["Reference", "Observed", "Predict1", "Predict2"]
    elif n_plots == 5:
        sub_titles = ["Reference", "Noisy", "Observed", "Predict1", "Predict2"]
    else:
        raise NotImplementedError

    plt.rcParams['figure.figsize'] = [9.6, 3.2]
    fig = plt.figure()
    gs_ = gridspec.GridSpec(2, 2 * n_plots)
    slice_ = [gs_[0:2, 2*i:2*i+2] for i in range(n_plots)]
    for i, data in enumerate(u_list):
        ax_ = fig.add_subplot(slice_[i])

        img = ax_.imshow(
            data.T, vmin=vmin_u,
            vmax=vmax_u,
            cmap=plt.get_cmap("jet"),
            origin='lower')

        ax_.set_title(sub_titles[i], fontsize=10)
        plt.xticks(())
        plt.yticks(())

        aspect = 20
        pad_fraction = 0.5
        divider = make_axes_locatable(ax_)
        width = axes_size.AxesY(ax_, aspect=1 / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cb_ = plt.colorbar(img, cax=cax)
        cb_.ax.tick_params(labelsize=6)

    gs_.tight_layout(fig, pad=1.0, w_pad=3.0, h_pad=1.0)

    fig.suptitle(title, y=1.1)
    fig.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_l2_error_and_num_nodes(data: NDArray[float],
                                file_name: str,
                                save_dir: Optional[str] = None) -> None:
    r"""
    Plot the line chart, where the X axis represents the number of
    nodes in the graphormer and the Y axis represents L2 error.

    Args:
        data (numpy.ndarray): A numpy array of shape (size, 2), where size is the number of
            experiments. The first column represents the number of nodes in the
            graphormer, and the second column represents the L2 error.
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    # data preprocessing
    num_nodes = data[:, 0]  # [size]
    l2_error = data[:, 1]  # [size]

    dic = {}
    for idx, num_node in enumerate(num_nodes):
        if num_node in dic:
            dic[num_node].append(l2_error[idx])
        else:
            dic[num_node] = [l2_error[idx]]

    for key, val in dic.items():
        dic[key] = np.array(val).mean()

    sorted_keys = sorted(dic.keys())
    points = []
    for key in sorted_keys:
        points.append([key, dic[key]])
    points = np.array(points)
    x_pts = [int(i) for i in points[:, 0]]
    y_pts = points[:, 1]

    # draw a line plot
    plt.figure(figsize=(6.8, 4.8))
    plt.plot(x_pts, y_pts, marker='o', markerfacecolor='red',
             markeredgecolor='red', markersize=5, ls=':')
    plt.xlabel("number of nodes")
    plt.ylabel("L2 error")
    plt.xticks(x_pts)
    plt.savefig(os.path.join(save_dir, f"line_{file_name}"))
    plt.close()

    # draw a scatter plot
    plt.figure(figsize=(6.8, 4.8))
    plt.scatter(num_nodes, l2_error, marker='o', s=1)
    plt.xlabel("number of nodes")
    plt.ylabel("L2 error")
    plt.xticks(x_pts)
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, f"scatter_{file_name}"))
    plt.close()


def plot_l2_error_and_epochs(data: list,
                             file_name: str,
                             save_dir: Optional[str] = None) -> None:
    r"""
    Plot the line chart, where the X axis represents epoch and the Y axis represents L2 error.

    Args:
        data (list): A list of numpy.ndarrays containing epoch, train_l2_error, and test_l2_error.
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    plt.figure(figsize=(6.8, 4.8))

    plt.plot(data[0], data[1], '--', color='b', label='train')
    plt.plot(data[0], data[2], '-', color='r', label='test')

    xticks_step = max(1, len(data[0]) // 5)
    xticks = [data[0][i] for i in range(0, len(data[0]), xticks_step)]
    plt.xticks(xticks)

    plt.xlabel('epochs')
    plt.ylabel('L2 error')
    plt.yscale('log')

    if min(data[1]) < 0.01 or min(data[2]) < 0.01:
        plt.ylim(0.001, 1.5)
    else:
        plt.ylim(0.01, 1.5)

    plt.legend()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()


def plot_l2_error_histogram(data: NDArray[float],
                            file_name: str,
                            save_dir: Optional[str] = None) -> None:
    r"""
    Plot a histogram of the L2 error distribution, where X represents
    the L2 error and the y-axis represents the frequency.

    Args:
        data (numpy.ndarray): A numpy array of shape (size, 1).
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    plt.figure(figsize=(6.8, 4.8))

    log_data = np.log10(data)

    plt.hist(log_data, bins=50, alpha=0.5, color='b')

    plt.title('L2 error Distribution')
    plt.xlabel('L2 error')
    plt.ylabel('Frequency')
    plt.xlim(log_data.min(), log_data.max())

    xticks = np.linspace(log_data.min(), log_data.max(), 5)
    xticks_label = [f"{10**x:.4f}" for x in xticks]
    plt.xticks(xticks, labels=xticks_label)

    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()


def plot_inverse_coef(label: NDArray[float],
                      pred: NDArray[float],
                      file_name: str,
                      save_dir: Optional[str] = None) -> None:
    r"""
    Plot the results of the inverted equation coefficients, with the X axis
    representing ground truth and the Y axis representing the inverted
    coefficients.

    Args:
        label (numpy.ndarray): A numpy array of shape (size).
        pred (numpy.ndarray): A numpy array of shape (size).
        file_name (str): The name of the saved file.
        save_dir (str): The directory to save the plot. Default: None.

    Returns:
        None.
    """
    plt.figure(figsize=(6, 6))

    range_ = [min(label.min(), np.percentile(pred, 5)), max(label.max(), np.percentile(pred, 95))]
    plt.plot(range_, range_, linewidth='1')

    mae = np.abs(label - pred)

    plt.scatter(label, pred, s=8, c=mae, cmap='jet')
    plt.xlabel(r"Ground Truth", fontsize=16)
    plt.ylabel(r"Recovered", fontsize=16)
    plt.grid(alpha=0.3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.close()


def plot_noise_ic(ic_gt: NDArray[float],
                  ic_noisy: NDArray[float],
                  file_name: str,
                  save_dir: Optional[str] = None) -> None:
    r"""Plot the (ground truth / noise) initial condition for the inverse problem."""
    plt.figure(figsize=(6.8, 4.8))

    plt.plot(ic_gt, label='IC_gt')
    plt.plot(ic_noisy, label='IC_noisy')
    plt.xlabel('x')

    plt.legend()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()


def plot_infer_result(u_pred: NDArray[float],
                      x_coord: NDArray[float],
                      t_coord: NDArray[float],
                      title: str = "Solution Field"):
    r"""Plot PDEformer inference results."""
    fig, ax_ = plt.subplots(figsize=(10, 6))

    cax = ax_.imshow(u_pred, cmap=plt.get_cmap("jet"), origin='lower')

    # Setting up the x and t ticks
    x_ticks = [x_coord[0], x_coord[len(x_coord) // 2], x_coord[-1]]
    t_ticks = [t_coord[0], t_coord[len(t_coord) // 2], t_coord[-1]]
    x_indices = [0, len(x_coord)//2, len(x_coord)-1]
    t_indices = [0, len(t_coord)//2, len(t_coord)-1]

    ax_.set_xticks(x_indices)
    ax_.set_xticklabels([f"{x:.1f}" for x in x_ticks], fontsize=15)
    ax_.set_yticks(t_indices)
    ax_.set_yticklabels([f"{t:.1f}" for t in t_ticks], fontsize=15)

    ax_.set_xlabel('x', fontsize=15)
    ax_.set_ylabel('t', fontsize=15)
    ax_.set_title(title, fontsize=15)

    # Create colorbar, attach it to `ax_` and make its size proportional to the plot
    fig.colorbar(cax, ax=ax_, fraction=0.022, pad=0.04)

    plt.show()

# ============================================================================
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
"""tools"""
import os
import time
import imageio
import matplotlib.pyplot as plt
import numpy as np
from mindspore import ops

from mindflow.utils import print_log


def plot_error_image(output, truth, low_res, xmin, xmax, ymin, ymax, num, fig_save_path):
    '''num: Number of time step'''
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_low_res, v_low_res = low_res[num, 0, ...], low_res[num, 1, ...]
    u_low_res, v_low_res = np.kron(u_low_res, np.ones((2, 2))), \
        np.kron(v_low_res, np.ones((2, 2)))

    u_low_res, v_low_res = u_low_res[1:, 1:], v_low_res[1:, 1:]
    u_star, v_star = truth[num, 0, ...], truth[num, 1, ...]
    u_pred, v_pred = output[num, 0, :, :], output[num, 1, :, :]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=1.0,
                          edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0,
                          edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=1.0,
                          edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('u (Meas.)')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none',
                          cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none',
                          cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=1.0,
                          edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('v (Meas.)')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)

    fig_path = fig_save_path + '/uv_comparison_'+str(num).zfill(3)+'.png'
    plt.savefig(fig_path)
    plt.close('all')

    pred = np.stack((u_pred, v_pred), axis=0).reshape(-1, 1)
    label = np.stack((u_star, v_star), axis=0).reshape(-1, 1)
    diff_norms = np.square(pred - label).sum()
    label_norms = np.square(label).sum()
    return diff_norms / label_norms, fig_path


def post_process(trainer, pattern):
    '''post_process'''
    print_log("================================Start Evaluation================================")
    time_beg = time.time()
    output = trainer.get_output(1800)
    output = ops.concat((output, output[:, :, :, 0:1]), axis=3)
    output = ops.concat((output, output[:, :, 0:1, :]), axis=2)
    truth_clean = np.concatenate(
        (trainer.truth_clean, trainer.truth_clean[:, :, :, 0:1]), axis=3)
    truth_clean = np.concatenate(
        (truth_clean, truth_clean[:, :, 0:1, :]), axis=2)
    low_res = truth_clean[:, :, ::2, ::2]
    output = output.asnumpy()

    print_log(output.shape, truth_clean.shape, low_res.shape)

    err_list = []
    img_path = []
    fig_save_path = './summary/figures_' + pattern
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    for i in range(0, 1801, 10):
        err, fig_path = plot_error_image(output, truth_clean, low_res, xmin=0, xmax=1, ymin=0, ymax=1,
                                         num=i, fig_save_path=fig_save_path)
        err_list.append([i, err])
        img_path.append(fig_path)
    gif_images = []
    for path in img_path:
        gif_images.append(imageio.imread(path))
    imageio.mimsave('results.gif', gif_images, duration=0.01)
    err_list = np.array(err_list)

    plt.figure(figsize=(6, 4))
    plt.plot(err_list[:, 0], err_list[:, 1])
    plt.xlabel('infer_step')
    plt.ylabel('relative_l2_error')
    plt.savefig("error.png")
    print_log('Infer stage, mean relative l2 error', err)
    print_log("=================================End Evaluation=================================")
    print_log("predict total time: {} s".format(time.time() - time_beg))

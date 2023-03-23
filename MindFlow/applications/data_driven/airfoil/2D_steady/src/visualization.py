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
# ==============================================================================
"""
visualization
"""
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import minorticks_on

from .utils import get_label_and_pred, save_label_and_pred, unpatchify, check_file_path


def set_font(small_size=25, medium_size=28, bigger_size=45):
    """set plot font"""
    font_legend = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18,
        'style': 'italic'
    }
    font_title = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 22
    }
    matplotlib.rcParams['mathtext.default'] = 'regular'
    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=bigger_size)  # legend fontsize
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('figure', titlesize=medium_size)  # fontsize of the figure title
    return font_title, font_legend


def plot_u_v_p(eval_dataset, model, data_params):
    """plot u v p image"""
    set_font(small_size=15)
    for data in eval_dataset.create_dict_iterator(output_numpy=False):
        inputs = data["inputs"]
        labels = data["labels"]
        pred = model(inputs)
        inputs = inputs.asnumpy()
        print("shape of inputs {} type {} max {}".format(inputs.shape, type(inputs), inputs.max()))
        print("shape of labels {} type {} max {}".format(labels.shape, type(labels), labels.max()))
        print("shape of pred {} type {} max {}".format(pred.shape, type(pred), pred.max()))
        break
    save_img_dir = os.path.join(data_params['post_dir'], 'uvp_ViT')
    check_file_path(save_img_dir)
    print(f'save img dir: {save_img_dir}')
    model_name = "ViT_"
    save_label_and_pred(labels, pred, data_params['post_dir'])
    print("save res done!")
    for i in range(data_params['batch_size']):
        print("plot {} / {} done".format(i + 1, data_params['batch_size']))
        plot_contourf(labels, pred, i, save_img_dir, data_params['grid_path'], model_name)


def plot_config(xgrid, ygrid, data, index, title_name=None):
    """plot_config"""
    set_font()
    fig_note = f"({chr(ord('a')+index-1)})"
    plt.title(title_name, y=0.8, fontsize=20, fontweight=500)
    if index in (1, 4, 7):
        plt.ylabel('y/d', fontsize=18, style='italic')
    if index >= 7:
        plt.xlabel('x/d', fontsize=18, style='italic')
        min_value, max_value = np.min(data), 0.6 * np.max(data)
    else:
        min_value, max_value = 0.9 * np.min(data), 0.9 * np.max(data)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlim((-0.5, 2))
    plt.ylim((-0.5, 2))
    box = {'facecolor': 'w', 'edgecolor': 'w'}
    plt.text(-0.43, 1.65, fig_note, bbox=box, fontsize=18)
    plt.contour(xgrid, ygrid, data, 21, vmin=min_value, vmax=max_value,
                linestyles="dashed", alpha=0.2)
    h = plt.contourf(xgrid, ygrid, data, 21,
                     vmin=min_value, vmax=max_value, cmap='jet')

    cb1 = plt.colorbar(h, fraction=0.03, pad=0.05)
    cb1.ax.tick_params(labelsize=13)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb1.locator = tick_locator
    cb1.update_ticks()
    minorticks_on()
    return h


def plot_contourf(label, pred, num, save_img_dir, grid_path, model_name="ViT"):
    """plot_contourf"""
    label = unpatchify(label).asnumpy()
    pred = unpatchify(pred).asnumpy()
    save_uvp_name = f"{save_img_dir}/{str(num).rjust(3, '0')}_UVP.png"
    save_cp_name = f"{save_img_dir}/{str(num).rjust(3, '0')}_cp.png"
    grid = np.load(grid_path)[num, ...]
    xgrid, ygrid = grid[..., 0], grid[..., 1]
    fig = plt.figure(1, figsize=(20, 14))

    for i, map_name in enumerate(['U', 'V', 'P']):
        gt = label[num][:, :, i]
        pd = pred[num][:, :, i]
        l1_error = np.abs(gt - pd)
        k1, k2, k3 = i + 1, i + 4, i + 7
        plt.subplot(3, 3, k1)
        plot_config(xgrid, ygrid, gt, k1, title_name=f'CFD-{map_name}')
        plt.subplot(3, 3, k2)
        plot_config(xgrid, ygrid, pd, k2, title_name=f'{model_name}-{map_name}')
        plt.subplot(3, 3, k3)
        plot_config(xgrid, ygrid, l1_error, k3, title_name=f'l1-{map_name}')
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    plt.show()
    fig.savefig(save_uvp_name, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    # cp plot
    label_single = label[num][:, :, -1]
    pred_single = pred[num][:, :, -1]
    fig = plt.figure(2, figsize=(5, 5))
    plt.scatter(xgrid[0, :], -label_single[0, :], s=30, facecolors='none', edgecolors='k', label='CFD')
    plt.plot(xgrid[0, :], -pred_single[0, :], 'b', linewidth=2, label=model_name)
    plt.ylabel('cp', style='italic', fontsize=18)
    plt.xlabel('x/d', style='italic', fontsize=18)
    plt.legend()
    plt.show()
    fig.savefig(save_cp_name, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def set_fig_range(xgrid, ygrid, u, min_value, max_value, title, flag=False):
    """set figure range"""
    plt.contourf(xgrid, ygrid, u, 21,
                 vmin=min_value, vmax=max_value,
                 cmap='jet')
    plt.xlim((-0.5, 2))
    plt.ylim((-0.5, 2))
    plt.xticks(())
    plt.yticks(())
    if flag:
        plt.colorbar()
        plt.title(title, y=0.7)


def plot_u_and_cp(eval_dataset, model, grid_path, save_dir):
    """plot_u_and_cp"""
    print("================================Start Plotting================================")
    time_beg = time.time()
    for data in eval_dataset.create_dict_iterator():
        label, pred = get_label_and_pred(data, model)
        break
    num = 0
    grid = np.load(grid_path)[num, ...]
    xgrid, ygrid = grid[..., 0], grid[..., 1]
    fig = plt.figure(1, figsize=(18, 5))
    for i in range(8):
        flag = i == 7
        plt.subplot(4, 8, i + 1)
        set_fig_range(xgrid, ygrid, label[i, :, :, 0], -0.2, 1.5,
                      'CFD', flag)
        plt.subplot(4, 8, 8 + i + 1)
        set_fig_range(xgrid, ygrid, pred[i, :, :, 0], -0.2, 1.5,
                      'ViT', flag)
        plt.subplot(4, 8, 16 + i + 1)
        l1_map = np.abs(label[i, :, :, 0] - pred[i, :, :, 0])
        set_fig_range(xgrid, ygrid, l1_map, 0, 0.01,
                      'l1-error', flag)
        plt.subplot(4, 8, 24 + i + 1)
        plt.scatter(xgrid[0, :], -label[i, :, :, -1][0, :],
                    s=10, facecolors='none', edgecolors='k', label='CFD')
        plt.plot(xgrid[0, :], -pred[i, :, :, -1][0, :],
                 'b', linewidth=2, label='ViT')
        plt.xticks(())
        plt.yticks(())
    save_img = f"{save_dir}/U_and_cp_compare.png"
    print(save_img)
    fig.savefig(save_img, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print("================================End Plotting================================")
    print("Plot total time: {} s".format(time.time() - time_beg))

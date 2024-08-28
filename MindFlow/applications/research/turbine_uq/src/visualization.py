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
# ============================================================================
"""visualization"""
import os
from typing import List

import numpy as np
from mindflow import load_yaml_config
import matplotlib.pyplot as plt

from .postprocess import CFDPost2D, get_grid_interp
from .utils import Record


def load_visualization_data(record: Record = None, data_type='test'):
    """load_visualization_data"""
    if data_type == 'train':
        data = np.load(record.train_save_path)
    elif data_type == 'test':
        data = np.load(record.test_save_path)
    data_true = data['true']
    data_pred = data['pred']
    return data_true, data_pred


def run_visualization(record: Record, performence_list: List[str] = None,
                      field_list: List[str] = None, save_path=None):
    """run_visualization"""
    if not save_path:
        save_path = record.image2d_dir
    if not performence_list:
        performence_list = [
            'Static_pressure_ratio',
            'Total_total_efficiency',
            'Mass_flow',
        ]
    if not field_list:
        field_list = [
            'Static Pressure',
            'Vx', 'Vy', 'Vz',
            'Absolute Total Temperature',
            'Density',
        ]
    data_true, data_pred = load_visualization_data(record, data_type='test')
    grid = get_grid_interp()
    post_true = CFDPost2D(data=data_true, grid=grid)
    post_pred = CFDPost2D(data=data_pred, grid=grid)
    plot_diagnal(post_true, post_pred, save_path=save_path, parameter_list=performence_list)
    plot_meridian(post_true, post_pred, save_path=save_path, parameter_list=field_list)


def plot_diagnal(post_true, post_pred, save_path=None, parameter_list=None):
    """draw the diagnal figure"""
    for parameter in parameter_list:
        rst_true = post_true.get_performance(parameter)
        rst_pred = post_pred.get_performance(parameter)
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))
        plot_regression_dot(axs, rst_true.squeeze(), rst_pred.squeeze(), label=parameter)
        if save_path == 'show':
            plt.show()
        else:
            save_path_jpg = os.path.join(save_path, f'diagnal_{parameter}.jpg')
            fig.savefig(save_path_jpg, bbox_inches='tight', transparent=True)


def plot_meridian(post_true, post_pred, save_path=None, parameter_list=None, draw_idx=None):
    """draw the meridian figure"""
    if draw_idx is None:
        draw_idx = range(0, 3)
    rst_true, rst_pred = [], []
    for parameter in parameter_list:
        rst_true.append(np.expand_dims(post_true.get_field(parameter), axis=3))
        rst_pred.append(np.expand_dims(post_pred.get_field(parameter), axis=3))
    rst_true = np.concatenate(rst_true, axis=-1)
    rst_pred = np.concatenate(rst_pred, axis=-1)
    name_dict = load_yaml_config(os.path.join('./configs', 'visualization.yaml'))['unit']
    field_name = [name_dict[x] for x in parameter_list]
    for i in draw_idx:
        fig, axs = plt.subplots(len(parameter_list), 3, figsize=(12, 2*len(parameter_list)))
        plot_fields_ms(fig, axs, rst_true[i], rst_pred[i],
                       post_true.grid, fields_name=field_name)
        fig.patch.set_alpha(0.)
        if save_path == 'show':
            plt.show()
        else:
            save_path_jpg = os.path.join(save_path, f'meridian_derive_{i}.jpg')
            fig.savefig(save_path_jpg, bbox_inches='tight', transparent=True)


def plot_regression_dot(axs, true, pred, label='pred'):
    """plot_regression_dot"""
    max_value, min_value = max(true), min(true)
    split_value = np.linspace(min_value, max_value, 11)
    xylabels = ('true value', 'pred value')
    split_dict = {}
    split_label = np.zeros(len(true))
    for i, val in enumerate(split_value):
        split_dict[i] = str(val)
        index = true >= val
        split_label[index] = i + 1
    axs.scatter(true, pred, marker='.', color='tomato', s=320, linewidth=1,
                facecolor='tomato', edgecolor='k', alpha=1, label=label)
    axs.plot([min_value, max_value], [min_value, max_value], 'k--', linewidth=2.0)
    axs.fill_between([0.995 * min_value, 1.005 * max_value],
                     [0.995**2 * min_value, 0.995*1.005 * max_value],
                     [1.005*0.995 * min_value, 1.005**2 * max_value],
                     alpha=0.2, color='darkcyan')
    axs.set_xlim((0.995 * min_value, 1.005 * max_value))
    axs.set_ylim((0.995 * min_value, 1.005 * max_value))
    axs.tick_params(axis='both', which='both', labelsize=1)
    axs.grid(True)
    axs.legend(loc="upper left")
    axs.set_xlabel(xylabels[0])
    axs.set_ylabel(xylabels[1])


def plot_fields_ms(fig, axs, real, pred, coord, fields_name=None):
    """plot_fields_ms"""
    fmin, fmax = real.min(axis=(0, 1)), real.max(axis=(0, 1))
    cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
    cmaps = ('RdYlBu_r', 'RdYlBu_r', 'coolwarm')
    x_pos, y_pos = coord[:, :, 0], coord[:, :, 1]
    for i, field in enumerate(fields_name):
        show_field_list = [real[..., i], pred[..., i], real[..., i] - pred[..., i]]
        limit = max(abs(show_field_list[-1].min()), abs(show_field_list[-1].max()))
        for j in range(3):
            axs[i][j].cla()
            f_true = axs[i][j].pcolormesh(x_pos, y_pos, show_field_list[j], cmap=cmaps[j],
                                          shading='gouraud', antialiased=True, snap=True)
            f_true.set_zorder(10)
            axs[i][j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
            axs[i][j].axis('equal')
            colorbar = fig.colorbar(f_true, ax=axs[i][j], shrink=0.75)
            if j < 2:
                f_true.set_clim(fmin[i], fmax[i])
                colorbar.ax.set_title(field, loc='center')
            else:
                f_true.set_clim(-limit, limit)
                colorbar.ax.set_title(r'$\mathrm{\Delta}$' + field, loc='center')
            axs[i][j].tick_params(axis='both', which='both')
            axs[i][j].spines['bottom'].set_linewidth(0)
            axs[i][j].spines['left'].set_linewidth(0)
            axs[i][j].spines['right'].set_linewidth(0)
            axs[i][j].spines['top'].set_linewidth(0)

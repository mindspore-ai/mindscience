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
"""visualization after training"""
import os

import numpy as np
import matplotlib.pyplot as plt

from .utils import Record
from .dataset import get_grid


def load_visualization_data(record: Record = None, data_type='test'):
    """load_visualization_data"""
    if data_type == 'train':
        data = np.load(record.train_save_path)
    elif data_type == 'test':
        data = np.load(record.test_save_path)
    return data


def run_visualization(record: Record, save_path: str = None):
    """run_visualization"""
    if save_path is None:
        save_path = record.image2d_dir
    data = load_visualization_data(record, data_type='test')
    for hole_idx in range(5):
        for super_times in [0, 1]:
            data_true = data[f'true_hole-{hole_idx}_super-{super_times}']
            data_pred = data[f'pred_hole-{hole_idx}_super-{super_times}']
            title = f'hole-{hole_idx}_super-{super_times}'
            if save_path == 'show':
                save_file_path = save_path
            else:
                save_file_path = os.path.join(save_path, title)
            plot_field(data_true, data_pred, get_grid(),
                       title=title, save_path=save_file_path, draw_idx=[0])


def plot_field(true, pred, grid, title=None, save_path=None, draw_idx=None):
    """draw the field figure"""
    if draw_idx is None:
        draw_idx = range(0, 3)
    for i in draw_idx:
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        plot_fields_ms(fig, axs, true[i], pred[i], grid)
        plt.title(title)
        if save_path == 'show':
            plt.show()
        else:
            save_path_jpg = os.path.join(f'{save_path}_{i}.jpg')
            fig.savefig(save_path_jpg, bbox_inches='tight', transparent=True)


def plot_fields_ms(fig, axs, real, pred, coord):
    """plot_fields_ms"""
    fmin, fmax = real.min(axis=(0, 1)), real.max(axis=(0, 1))
    cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
    cmaps = ('RdYlBu_r', 'RdYlBu_r', 'coolwarm')
    x_pos, y_pos = coord[:, :, 0], coord[:, :, 1]
    show_field_list = [real[..., 0], pred[..., 0], real[..., 0] - pred[..., 0]]
    limit = max(abs(show_field_list[-1].min()), abs(show_field_list[-1].max()))
    for j in range(3):
        axs[j].cla()
        f_true = axs[j].pcolormesh(x_pos, y_pos, show_field_list[j], cmap=cmaps[j],
                                   shading='gouraud', antialiased=True, snap=True)
        f_true.set_zorder(10)
        axs[j].axis([cmin[0], cmax[0], cmin[1], cmax[1]])
        axs[j].axis('equal')
        colorbar = fig.colorbar(f_true, ax=axs[j], shrink=0.75)
        if j < 2:
            f_true.set_clim(fmin, fmax)
            colorbar.ax.set_title('T', loc='center')
        else:
            f_true.set_clim(-limit, limit)
            colorbar.ax.set_title(r'$\mathrm{\Delta}$' + 'T', loc='center')
        axs[j].tick_params(axis='both', which='both')
        axs[j].spines['bottom'].set_linewidth(0)
        axs[j].spines['left'].set_linewidth(0)
        axs[j].spines['right'].set_linewidth(0)
        axs[j].spines['top'].set_linewidth(0)

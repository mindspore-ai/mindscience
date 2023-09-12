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
"""plot functions for phygeonet"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from .py_mesh import visualize2d, set_axis_label


def plot_train(args, ev_hist, m_res_hist, time_spent):
    """plot train result"""
    plt.figure()
    plt.plot(m_res_hist, '-*', label='Equation Residual')
    plt.xlabel('Epoch')
    plt.ylabel('Residual')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{args.figures_path}/convergence.pdf', bbox_inches='tight')
    tikzplotlib.save(f'{args.figures_path}/convergence.tikz')
    plt.figure()
    plt.plot(ev_hist, '-x', label=r'$e_v$')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{args.figures_path}/error.pdf', bbox_inches='tight')
    tikzplotlib.save(f'{args.figures_path}/error.tikz')
    ev_hist = np.asarray(ev_hist)
    m_res_hist = np.asarray(m_res_hist)
    os.makedirs(args.save_data_path, exist_ok=True)
    np.savetxt(f'{args.save_data_path}/ev_hist.txt', ev_hist)
    np.savetxt(f'{args.save_data_path}/m_res_hist.txt', m_res_hist)
    np.savetxt(f'{args.save_data_path}/time_spent.txt', np.zeros([2, 2]) + time_spent)


def plot_train_process(args, coord, epoch, ofv_sb, output_v):
    """plot train process"""
    fig1 = plt.figure()
    ax = plt.subplot(1, 2, 1)
    visualize2d(ax, (coord[0, 0, 1:-1, 1:-1].numpy(),
                     coord[0, 1, 1:-1, 1:-1].numpy(),
                     output_v[0, 0, 1:-1, 1:-1].numpy()), 'horizontal', [0, 1])
    set_axis_label(ax, 'p')
    ax.set_title('CNN ' + r'$T$')
    ax.set_aspect('equal')
    ax = plt.subplot(1, 2, 2)
    visualize2d(ax, (coord[0, 0, 1:-1, 1:-1].numpy(),
                     coord[0, 1, 1:-1, 1:-1].numpy(),
                     ofv_sb[1:-1, 1:-1]), 'horizontal', [0, 1])
    set_axis_label(ax, 'p')
    ax.set_aspect('equal')
    ax.set_title('FV ' + r'$T$')
    fig1.tight_layout(pad=1)
    fig1.savefig(f"{args.figures_path}/{epoch}T.pdf", bbox_inches='tight')
    plt.close(fig1)

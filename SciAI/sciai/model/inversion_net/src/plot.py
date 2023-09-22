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
"""Vision kit"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def plot_velocity(output, target, path, vmin=None, vmax=None):
    """plot velocity model"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    if vmin is None or vmax is None:
        vmax, vmin = np.max(target), np.min(target)
    im = ax[0].matshow(output, cmap='jet', vmin=vmin, vmax=vmax)
    ax[0].set_title('Prediction', y=1.08)
    ax[1].matshow(target, cmap='jet', vmin=vmin, vmax=vmax)
    ax[1].set_title('Ground Truth', y=1.08)

    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))

        axis.set_ylabel('Depth (m)', fontsize=12)
        axis.set_xlabel('Offset (m)', fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label='Velocity(m/s)')
    plt.savefig(path)
    plt.close('all')


def plot_seismic(output, target, path, vmin=-1e-5, vmax=1e-5):
    """plot inversion result"""
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    aspect = output.shape[1] / output.shape[0]
    im = ax[0].matshow(target, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title('Ground Truth')
    ax[1].matshow(output, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title('Prediction')
    ax[2].matshow(output - target, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_title('Difference')

    fig.colorbar(im, ax=ax, shrink=0.75, label='Amplitude')
    plt.savefig(path)
    plt.close('all')

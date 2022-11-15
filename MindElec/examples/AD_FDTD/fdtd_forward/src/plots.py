# Copyright 2021 Huawei Technologies Co., Ltd
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
#pylint: disable=W0613
"""plot S parameters"""
import numpy as np
import matplotlib.pyplot as plt


def plot_s(fs, s_parameters, save_path):
    """
    Plot complex-valued S parameters.

    Args:
        fs (numpy.ndarray, shape=(nf,)): Sampling frequency (Hz).
        s_parameters (numpy.ndarray, shape=(nf, nr, ns)): Complex-valued S parameters.
        save_path (str): Path to save the figure.

    Returns:
        None.
    """

    nrows, ncols = s_parameters.shape[-2:]
    _, axes = plt.subplots(nrows, ncols)
    if nrows == 1 & ncols == 1:
        axes.plot(fs * 1e-9, 20. * np.log10(np.abs(s_parameters[:, 0, 0])),
                  ls='-', label='|S_11|')
        axes.set_xlabel('frequency (GHz)')
        axes.set_ylabel('Magnitude (dB)')
        axes.legend()
        axes.grid()

    elif ncols == 1:
        for m in range(s_parameters.shape[-2]):
            axes[m].plot(fs * 1e-9, 20. * np.log10(np.abs(s_parameters[:, m, 0])),
                         ls='-', label=f'|S_{m+1}{ncols}|')
            axes[m].set_xlabel('frequency (GHz)')
            axes[m].set_ylabel('Magnitude (dB)')
            axes[m].legend()
            axes[m].grid()

    else:
        for m in range(s_parameters.shape[-2]):
            for n in range(s_parameters.shape[-1]):
                axes[m, n].plot(fs * 1e-9, 20. * np.log10(np.abs(s_parameters[:, m, n])),
                                ls='-', label=f'|S_{m+1}{n+1}|')
                axes[m, n].set_xlabel('frequency (GHz)')
                axes[m, n].set_ylabel('Magnitude (dB)')
                axes[m, n].legend()
                axes[m, n].grid()

    plt.suptitle('S Parameters')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compare_s(fs, s_parameters, save_path, *args):
    """
    Compare simulated complex-valued S parameters with reference.

    Args:
        fs (numpy.ndarray, shape=(nf,)): Sampling frequency (Hz).
        s_parameters (numpy.ndarray, shape=(nf, nr, ns)): Complex-valued S parameters.
        save_path (str): Path to save the figure.
        args: reference results.

    Returns:
        None.
    """

    nrows, ncols = s_parameters.shape[-2:]
    _, axes = plt.subplots(nrows, ncols)
    if nrows == 1 & ncols == 1:
        axes.plot(fs * 1e-9, 20. * np.log10(np.abs(s_parameters[:, 0, 0])),
                  ls='-', lw=2, color='black', label='|S_11|, Code')
        axes.plot(args[0][:, 0], args[0][:, 1], ls='--', lw=2,
                  color='red', label='|S_11|, Ref.')
        axes.set_xlabel('frequency (GHz)')
        axes.set_ylabel('Magnitude (dB)')
        axes.legend()
        axes.grid()

    elif ncols == 1:
        for m in range(s_parameters.shape[-2]):
            axes[m].plot(fs * 1e-9, 20. * np.log10(np.abs(s_parameters[:, m, 0])),
                         ls='-', lw=2, color='black', label=f'|S_{m+1}{ncols}|, Code')
            axes[m].plot(args[m][:, 0], args[m][:, 1], ls='--',
                         lw=2, color='red', label=f'|S_{m+1}{ncols}|, Ref.')
            axes[m].set_xlabel('frequency (GHz)')
            axes[m].set_ylabel('Magnitude (dB)')
            axes[m].legend()
            axes[m].grid()

    else:
        for m in range(s_parameters.shape[-2]):
            for n in range(s_parameters.shape[-1]):
                axes[m, n].plot(fs * 1e-9, 20. * np.log10(np.abs(s_parameters[:, m, n])),
                                ls='-', lw=2, color='black', label=f'|S_{m+1}{n+1}|, Code')
                axes[m, n].plot(args[m * ncols + n][:, 0], args[m * ncols + n][:, 1],
                                ls='--', lw=2, color='red', label=f'|S_{m+1}{n+1}|, Ref.')
                axes[m, n].set_xlabel('frequency (GHz)')
                axes[m, n].set_ylabel('Magnitude (dB)')
                axes[m, n].legend()
                axes[m, n].grid()

    plt.suptitle('S Parameters')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
"""plot utils"""
import json
import os
import sys
from argparse import Namespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sciai.utils.log_utils import print_log
from sciai.utils.time_utils import time_str


def save_result_dir(save_path, save_hp):
    """
    Save figure result in given directory.

    Args:
        save_path (str): Directory path to save figures and hyperparameters.
        save_hp (Union[dict, Namespace]): Hyperparameters to save.
    """
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    res_dir = os.path.join(save_path, f"{time_str()}-{script_name}")
    try:
        os.makedirs(res_dir)
    except FileExistsError as _:
        print_log("makedirs failed due to system error.")
        return
    print_log("Saving results to directory ", res_dir)
    try:
        savefig(os.path.join(res_dir, "graph"))
    except IOError as e:
        print_log(f"warning: failed to save results due to matplotlib latex installation, error:{e}")
    if isinstance(save_hp, Namespace):
        save_hp = vars(save_hp)
    with open(os.path.join(res_dir, "hp.json"), mode="w") as f:
        json.dump(save_hp, f)


def _figsize(scale, num_plots=1):
    """
    Figure size configuration.

    Args:
        scale (Number): Scale of width.
        num_plots (int): Number of plots. Default: 1.

    Returns:
        list, Figure size configuration.
    """
    fig_width_pt = 390.0
    resolution = 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    width = fig_width_pt * scale / resolution
    height = num_plots * width * golden_mean
    return [width, height]


# setup matplotlib to use latex for output
_pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": False,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.size": 10,
    "figure.figsize": _figsize(1.0),
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ]
}
mpl.rcParams.update(_pgf_with_latex)


def newfig(width, num_plots=1):
    """
    Plot a new figure.

    Args:
        width (Number): Figures width.
        num_plots (int): Number of plots.

    Returns:
        tuple, Matplot Figure, and axes.Axes.
    """
    fig_size = _figsize(width, num_plots)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    """
    Save figure in both pdf and png.

    Args:
        filename (str): Filename of the figure.
        crop (bool): crop or not. Default: True.
    """
    bbox = 'tight' if crop else None
    pad = 0 if crop else 0.1
    plt.savefig('{}.pdf'.format(filename), bbox_inches=bbox, pad_inches=pad)
    plt.savefig('{}.png'.format(filename), bbox_inches=bbox, pad_inches=pad)

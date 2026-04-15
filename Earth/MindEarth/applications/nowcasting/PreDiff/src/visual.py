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
"visualization"
import math
from copy import deepcopy
from typing import Optional, Sequence, Union, Dict
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np


VIL_COLORS = [
    [0, 0, 0],
    [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
    [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
    [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
    [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
    [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
    [0.9607843137254902, 0.9607843137254902, 0.0],
    [0.9294117647058824, 0.6745098039215687, 0.0],
    [0.9411764705882353, 0.43137254901960786, 0.0],
    [0.6274509803921569, 0.0, 0.0],
    [0.9058823529411765, 0.0, 1.0],
]

VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]


def vil_cmap():
    """
    Generate a ListedColormap and normalization for VIL (Vertically Integrated Liquid) visualization.

    This function creates a colormap with specific color levels for VIL data visualization. It sets under/over colors
    for values outside the defined levels and handles invalid (NaN) values.

    Returns:
        tuple: A tuple containing:
            - cmap (ListedColormap): Colormap object with defined colors.
            - norm (BoundaryNorm): Normalization object based on VIL levels.
            - vmin (None): Minimum value for colormap (set to None).
            - vmax (None): Maximum value for colormap (set to None).
    """
    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)
    nil = cols.pop(0)
    under = cols[0]
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, cmap.N)
    vmin, vmax = None, None
    return cmap, norm, vmin, vmax


def vis_sevir_seq(
        save_path,
        seq: Union[np.ndarray, Sequence[np.ndarray]],
        label: Union[str, Sequence[str]] = "pred",
        norm: Optional[Dict[str, float]] = None,
        interval_real_time: float = 10.0,
        plot_stride=2,
        label_rotation=0,
        label_offset=(-0.06, 0.4),
        label_avg_int=False,
        fs=10,
        max_cols=10,
):
    """Visualize SEVIR sequence data as a grid of images with colormap and annotations.
    Args:
        save_path (str): Path to save the output visualization figure.
        seq (Union[np.ndarray, Sequence[np.ndarray]]): Input data sequence(s) to visualize.
            Can be a single array or list of arrays.
        label (Union[str, Sequence[str]], optional): Labels for each sequence. Defaults to "pred".
        norm (Optional[Dict[str, float]], optional): Normalization parameters (scale/shift).
            Defaults to {"scale": 255, "shift": 0}.
        interval_real_time (float, optional): Time interval between frames in real time. Defaults to 10.0.
        plot_stride (int, optional): Stride for subsampling frames. Defaults to 2.
        label_rotation (int, optional): Rotation angle for y-axis labels. Defaults to 0.
        label_offset (tuple, optional): Offset for y-axis label position. Defaults to (-0.06, 0.4).
        label_avg_int (bool, optional): Append average intensity to labels. Defaults to False.
        fs (int, optional): Font size for text elements. Defaults to 10.
        max_cols (int, optional): Maximum number of columns per row. Defaults to 10.

    Raises:
        NotImplementedError: If input sequence type is not supported.

    Returns:
        None: Saves visualization to disk and closes the figure.
    """
    def cmap_dict():
        return {
            "cmap": vil_cmap()[0],
            "norm": vil_cmap()[1],
            "vmin": vil_cmap()[2],
            "vmax": vil_cmap()[3],
        }

    fontproperties = FontProperties()
    fontproperties.set_family("serif")
    fontproperties.set_size(fs)

    if isinstance(seq, Sequence):
        seq_list = [ele.astype(np.float32) for ele in seq]
        assert isinstance(label, Sequence) and len(label) == len(seq)
        label_list = label
    elif isinstance(seq, np.ndarray):
        seq_list = [
            seq.astype(np.float32),
        ]
        assert isinstance(label, str)
        label_list = [
            label,
        ]
    else:
        raise NotImplementedError
    if label_avg_int:
        label_list = [
            f"{ele1}\nAvgInt = {np.mean(ele2): .3f}"
            for ele1, ele2 in zip(label_list, seq_list)
        ]
    seq_list = [ele[::plot_stride, ...] for ele in seq_list]
    seq_in_list = [len(ele) for ele in seq_list]
    max_len = max(seq_in_list)
    max_len = min(max_len, max_cols)
    seq_list_wrap = []
    label_list_wrap = []
    seq_in_list_wrap = []
    for i, (processed_seq, processed_label, seq_in) in enumerate(zip(seq_list, label_list, seq_in_list)):
        num_row = math.ceil(seq_in / max_len)
        for j in range(num_row):
            slice_end = min(seq_in, (j + 1) * max_len)
            seq_list_wrap.append(processed_seq[j * max_len : slice_end])
            if j == 0:
                label_list_wrap.append(processed_label)
            else:
                label_list_wrap.append("")
            seq_in_list_wrap.append(min(seq_in - j * max_len, max_len))

    if norm is None:
        norm = {"scale": 255, "shift": 0}
    nrows = len(seq_list_wrap)
    fig, ax = plt.subplots(nrows=nrows, ncols=max_len, figsize=(3 * max_len, 3 * nrows))

    for i, (processed_seq, processed_label, seq_in) in enumerate(
            zip(seq_list_wrap, label_list_wrap, seq_in_list_wrap)
    ):
        ax[i][0].set_ylabel(
            ylabel=processed_label, fontproperties=fontproperties, rotation=label_rotation
        )
        ax[i][0].yaxis.set_label_coords(label_offset[0], label_offset[1])
        for j in range(0, max_len):
            if j < seq_in:
                x = processed_seq[j] * norm["scale"] + norm["shift"]
                ax[i][j].imshow(x, **cmap_dict())
                if i == len(seq_list) - 1 and i > 0:
                    ax[-1][j].set_title(
                        f"Min {int(interval_real_time * (j + 1) * plot_stride)}",
                        y=-0.25,
                        fontproperties=fontproperties,
                    )
            else:
                ax[i][j].axis("off")

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].xaxis.set_ticks([])
            ax[i][j].yaxis.set_ticks([])

    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [
        Patch(
            facecolor=VIL_COLORS[i],
            label=f"{int(VIL_LEVELS[i - 1])}-{int(VIL_LEVELS[i])}",
        )
        for i in range(1, num_thresh_legend + 1)
    ]
    ax[0][0].legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(-1.2, -0.0),
        borderaxespad=0,
        frameon=False,
        fontsize="10",
    )
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path)
    plt.close(fig)

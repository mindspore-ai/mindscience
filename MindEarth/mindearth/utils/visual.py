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
"""visual"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..data import FEATURE_DICT


def plt_global_field_data(data, feature_name, std, mean, fig_title, is_surface=False, is_error=False):
    """
    Visualization of global field weather data.

    Args:
        data (numpy.array): The global field points.
        feature_name (str): The name of the feature to be visualized.
        std (numpy.array): The standard deviation of per-varibale-level.
        mean (numpy.array): The mean value of per-varibale-level.
        fig_title (str): The title of the figure.
        is_surface (bool): Whether or not a surface feature. Default: ``False`` .
        is_error (bool): Whether or not plot error. Default: ``False`` .

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``
    """
    level_num, feat_num = FEATURE_DICT.get(feature_name)
    feature_data = data[0, level_num + feat_num * 13]
    if is_surface:
        if is_error:
            feature_data = feature_data * std[level_num]
        else:
            feature_data = feature_data * std[level_num] + mean[level_num]
    else:
        if is_error:
            feature_data = feature_data * std[level_num, 0, 0, feat_num]
        else:
            feature_data = feature_data * std[level_num, 0, 0, feat_num] + mean[level_num, 0, 0, feat_num]
    norm = matplotlib.colors.Normalize(vmin=np.min(feature_data), vmax=np.max(feature_data))
    plt.imshow(X=feature_data, cmap='RdBu', norm=norm)
    plt.axis('off')
    plt.title(fig_title + ' ' + feature_name, color='black', fontsize=80)
    cb = plt.colorbar(fraction=0.025)
    cb.ax.tick_params(labelsize=40)


def plt_metrics(x, y, fig_title, label, ylabel="", xlabel="Forecast Time (hours)", loc="upper right"):
    """
    Visualization of latitude weighted rmse or acc.

    Args:
        x (numpy.array): The x value in the figure.
        y (numpy.array): The y value in the figure.
        fig_title (str): The name of the figure.
        label (str): The label of the visualization curve.
        ylabel (str): The label of the axis y. Default: ``""`` .
        xlabel (str): The label of the axis x. Default: ``"Forecast Time (hours)"`` .
        loc (str): The position of legend in the figure. Default: ``"upper right"`` .

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``
    """
    fontdict = {"family": "serif", "fontsize": 16}
    plt.title(fig_title, fontdict={"family": 'serif', 'size': 20})
    plt.plot(x, y, 'bo-', label=label, markersize=3)
    plt.legend(loc=loc)
    plt.ylabel(ylabel, fontdict=fontdict)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.xticks(fontsize=10, fontfamily='serif')
    plt.yticks(fontsize=10, fontfamily='serif')

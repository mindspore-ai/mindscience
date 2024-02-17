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
"""
visual
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def plt_log(predicts, labels, img_dir, epoch=0):
    """plot log"""
    plt.rcParams['figure.figsize'] = (12, 4.8)
    for i in range(3):
        label = labels[0, ..., i]
        predict = predicts[0, ..., i]
        prefixes = ["U", "V", "P"]
        for prefix in prefixes:
            t, _, _ = np.shape(label)

            error = np.abs(label - predict)

            vmin_u = label.min()
            vmax_u = label.max()

            vmin_error = error.min()
            vmax_error = error.max()

            vmin = [vmin_u, vmin_u, vmin_error]
            vmax = [vmax_u, vmax_u, vmax_error]

            t = len(label)
            step = int(t // np.minimum(10, t))
            times = [i * step for i in range(np.minimum(10, t))]

            sub_titles = ["Label", "Predict", "Error"]
            items = ["$T=%d$" % (t) for t in times]

            label_2d = [label[t, ...] for t in times]
            predict_2d = [predict[t, ...] for t in times]
            error_2d = [error[t, ...] for t in times]

            fig = plt.figure()
            gs = gridspec.GridSpec(3, len(times))
            gs_idx = int(0)

            for j, data_2d in enumerate([label_2d, predict_2d, error_2d]):
                for k, data in enumerate(data_2d):
                    ax = fig.add_subplot(gs[gs_idx])
                    gs_idx += 1

                    img = ax.imshow(data.T, vmin=vmin[j], vmax=vmax[j], cmap=plt.get_cmap("turbo"), origin='lower')

                    ax.set_title(sub_titles[j] + " " + items[k], fontsize=10)
                    plt.axis('off')

                aspect = 20
                pad_fraction = 0.5
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1 / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cb = plt.colorbar(img, cax=cax)
                cb.ax.tick_params(labelsize=6)

            gs.tight_layout(fig, pad=0.2, w_pad=0.2, h_pad=0.2)

            file_name = os.path.join(img_dir, prefix + "_epoch-%d_result.png" % epoch)
            fig.savefig(file_name)

            plt.close()

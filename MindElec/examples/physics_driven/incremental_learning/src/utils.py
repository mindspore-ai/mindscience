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
"""visualization of field quantities"""
import os

import copy
import io
import cv2
import PIL
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


plt.rcParams['figure.dpi'] = 300

def visual_result(inputs, fdtd, prediction, save_path, name):
    """visulization of ex/ey/hz"""
    inputs = copy.deepcopy(inputs)
    fdtd = copy.deepcopy(fdtd)
    prediction = copy.deepcopy(prediction)

    [num_t, num_x, num_y, _] = np.shape(inputs)

    vmin_ex, vmax_ex = np.percentile(fdtd[:, :, :, 0], [0.5, 99.5])
    vmin_ey, vmax_ey = np.percentile(fdtd[:, :, :, 1], [0.5, 99.5])
    vmin_hz, vmax_hz = np.percentile(fdtd[:, :, :, 2], [0.5, 99.5])

    list_vmin = [vmin_ex, vmin_ey, vmin_hz]
    list_vmax = [vmax_ex, vmax_ey, vmax_hz]

    out_names = ["Ex", "Ey", "Hz"]

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    cv_fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video_fps = 10
    video_size = (1920, 1440)
    avi = cv2.VideoWriter(os.path.join(save_path, "EH_" + str(name) + ".avi"), cv_fcc, video_fps, video_size)

    t_set = []
    if num_t < 100:
        t_set = np.arange(num_t, dtype=np.int32)
    else:
        for t in range(num_t):
            if t % int(num_t / 20) == 0 or t == num_t - 1:
                t_set.append(t)

    for t in t_set:
        ex_label = fdtd[t, :, :, 0]
        ey_label = fdtd[t, :, :, 1]
        hz_label = fdtd[t, :, :, 2]

        ex_prediction = prediction[t, :, :, 0]
        ey_prediction = prediction[t, :, :, 1]
        hz_prediction = prediction[t, :, :, 2]

        ex_label_2d = np.reshape(np.array(ex_label), (num_x, num_y))
        ey_label_2d = np.reshape(np.array(ey_label), (num_x, num_y))
        hz_label_2d = np.reshape(np.array(hz_label), (num_x, num_y))

        ex_prediction_2d = np.reshape(np.array(ex_prediction), (num_x, num_y))
        ey_prediction_2d = np.reshape(np.array(ey_prediction), (num_x, num_y))
        hz_prediction_2d = np.reshape(np.array(hz_prediction), (num_x, num_y))

        ex_error_2d = np.abs(ex_prediction_2d - ex_label_2d)
        ey_error_2d = np.abs(ey_prediction_2d - ey_label_2d)
        hz_error_2d = np.abs(hz_prediction_2d - hz_label_2d)

        fdtd_2d = [ex_label_2d, ey_label_2d, hz_label_2d]
        prediction_2d = [ex_prediction_2d, ey_prediction_2d, hz_prediction_2d]
        abs_error_2d = [ex_error_2d, ey_error_2d, hz_error_2d]

        comp_2d = [fdtd_2d, prediction_2d, abs_error_2d]
        img_names = ["label", "prediction", "error"]

        avi_fig = plt.figure()

        grid_spec = gridspec.GridSpec(3, 3)

        title = "t={:d}".format(t)
        plt.suptitle(title, fontsize=14)

        img_idx = int(0)

        for i, data_2d in enumerate(comp_2d):
            for j, data in enumerate(data_2d):
                ax = avi_fig.add_subplot(grid_spec[img_idx])
                img_idx += 1

                if img_names[i] == "error":
                    img = ax.imshow(data.T, vmin=0, vmax=1, cmap=plt.get_cmap("jet"), origin='lower')
                else:
                    img = ax.imshow(data.T, vmin=list_vmin[j], vmax=list_vmax[j], cmap=plt.get_cmap("jet"),
                                    origin='lower')

                ax.set_title(out_names[j] + " " + img_names[i], fontsize=4)
                plt.xticks(size=4)
                plt.yticks(size=4)

                img_aspect = 20
                padding_fraction = 0.5
                divide = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1/img_aspect)
                pad = axes_size.Fraction(padding_fraction, width)
                img_cax = divide.append_axes("right", size=width, pad=pad)
                cb = plt.colorbar(img, cax=img_cax)
                cb.ax.tick_params(labelsize=4)

        grid_spec.tight_layout(avi_fig, pad=0.4, w_pad=0.4, h_pad=0.4)

        # save image to memory buffer
        buffer = io.BytesIO()
        avi_fig.savefig(buffer, format="jpg")
        buffer.seek(0)
        image = PIL.Image.open(buffer)

        avi.write(np.asarray(image))

        buffer.close()

        plt.close()

    avi.release()

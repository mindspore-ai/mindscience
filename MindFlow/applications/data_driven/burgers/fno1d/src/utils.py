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
utils
"""
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mindspore import Tensor
from mindspore import dtype as mstype

def visual(model, epochs=1, resolution=1024):
    '''Use the trained model for inference and visualize the results.'''
    x = np.linspace(-1, 1, resolution)
    input_data = -np.sin(np.pi * x).reshape(1,-1,1).astype(np.float32)
    input_tensor = Tensor(input_data, dtype=mstype.float32)
    u_predict_list = []
    u_predict_list.append(input_tensor.asnumpy())
    for i in range(3):
        u_predict = model(input_tensor)
        input_tensor = u_predict
        u_predict_list.append(u_predict.asnumpy())
    u_predict_list = np.concatenate(u_predict_list,axis=0).squeeze()
    x_flat = np.linspace(-1, 1, resolution)
    t_points = np.linspace(0, 3, 4)
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1])
    plt.subplot(gs[0, :])
    heatmap_data = np.array(u_predict_list).T
    im = plt.imshow(
        heatmap_data,
        extent=[t_points.min(), t_points.max(), x_flat.min(), x_flat.max()],
        aspect='auto',
        cmap=plt.cm.rainbow,
        origin='lower'
    )
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(im, pad=0.05, aspect=10)
    cbar.set_label('u(t, x)')
    cbar.mappable.set_clim(-1, 1)

    t_cross_sections = [1, 2, 3]
    for i, t_cs in enumerate(t_cross_sections):
        ax = plt.subplot(gs[1, i])
        ax.plot(x_flat, u_predict_list[t_cs])
        ax.set_title(f't={t_cs}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(t, x)')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(f'images/{epochs + 1}-result.jpg')

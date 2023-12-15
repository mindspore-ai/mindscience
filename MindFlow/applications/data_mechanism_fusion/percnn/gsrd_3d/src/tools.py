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
"""tools"""
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def post_process(data, fig_path, is_u=True, res_flag=0, num=None):
    """post process"""
    x = np.linspace(-50, 50, 48)
    y = np.linspace(-50, 50, 48)
    z = np.linspace(-50, 50, 48)
    x, y, z = np.meshgrid(x, y, z)

    appd = ['PeRCNN', 'Truth']
    uv = ['v', 'u']
    values = data

    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=0.3 if is_u else 0.1,
        isomax=0.5 if is_u else 0.3,
        opacity=0.2,
        colorscale='RdBu',  # 'BlueRed',
        surface_count=2,  # number of isosurfaces, 2 by default: only min and max
    ))

    file_name = os.path.join(fig_path, 'Iso_surf_%s_%s_%d.png' %
                             (uv[is_u], appd[res_flag], num))
    print(f'write image to {file_name}')
    fig.write_image(file_name)
    plt.close('all')


def count_params(params):
    """count the number of parameters"""
    total_params = 0
    for param in params:
        total_params += np.prod(param.shape)
    return total_params

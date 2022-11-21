# Copyright 2022 Huawei Technologies Co., Ltd
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
utility functions
"""
import os
import yaml
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from mindspore import Tensor
import mindspore.common.dtype as mstype


EPS = 1e-8
np.random.seed(0)


def make_paths_absolute(dir_, config):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    config : dict

    Returns
    -------
    config : dict
    """
    for key in config.keys():
        if key.endswith("_path"):
            config[key] = os.path.join(dir_, config[key])
            config[key] = os.path.abspath(config[key])
        if isinstance(config[key], dict):
            config[key] = make_paths_absolute(dir_, config[key])
    return config


def load_config(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    config : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        config = yaml.safe_load(stream)
    config = make_paths_absolute(os.path.join(os.path.dirname(yaml_filepath), ".."), config)
    return config


def visual_result(model, resolution=100):
    """visulization of ex/ey/hz"""
    t_flat = np.linspace(0, 1, resolution)
    x_flat = np.linspace(-1, 1, resolution)
    t_grid, x_grid = np.meshgrid(t_flat, x_flat)
    x = x_grid.reshape((-1, 1))
    t = t_grid.reshape((-1, 1))
    xt = Tensor(np.concatenate((x, t), axis=1), dtype=mstype.float32)
    u_predict = model(xt)
    u_predict = u_predict.asnumpy()
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.scatter(t, x, c=u_predict, cmap=plt.cm.rainbow)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    t_cross_sections = [0.25, 0.5, 0.75]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        xt = Tensor(np.stack([x_flat, np.full(x_flat.shape, t_cs)], axis=-1), dtype=mstype.float32)
        u = model(xt).asnumpy()
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    plt.savefig('result.jpg')

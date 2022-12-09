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
# ==============================================================================
"""
config
"""
from easydict import EasyDict as ed


cuboid_sampling_config = ed({
    'domain': ed({
        'random_sampling': False,
        'size': [64, 64, 64],
    }),
    'BC': ed({
        'random_sampling': True,
        'size': 65536,
        'sampler': 'uniform',
    })
})


maxwell_3d_config = ed({
    "name": "Maxwell3D",              # Model name
    "geom_name": "cuboid",            # Geometry name

    # Training parameters
    "epochs": 3000,                   # epoch, >=1000 is recommended
    "batch_size": 256,                # batch size
    "lr": 0.001,                      # learning rate
    "pretrained": False,              # if use pretrained wieght
    "param_path": "checkpoints/model_cavity_best.ckpt",  # path of pretrained weight

    # Simulation parameters
    "coord_min": [0.0, 0.0, 0.0],     # minimum x,y,z coordinates
    "coord_max": [2.0, 2.0, 2.0],     # maximum x,y,z coordinates
    "eps0": 1.0,                      # permitivity of vacuum
    "wave_number": 16,                # wave number
    "eigenmode": 1,                   # eigen mode

    # Neural network parameters
    "in_channel": 3,                  # number of input channels
    "out_channel": 3,                 # number of output channels
    "layers": 6,                      # number of layres
    "neurons": 32,                    # number of neurons each layer

    # Evaluating parameters
    "axis_size": 101,                 # number of grid
    "result_save_dir": "result",      # directory to save result
})

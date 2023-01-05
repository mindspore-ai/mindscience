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


# Sampling configuration of cubic.
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


# Model configuration.
maxwell_3d_config = ed({
    "name": "Maxwell3D",              # model name
    "geom_name": "cuboid",            # gemometry name
    "waveguide_name": "waveguide",    # waveguide name
    "waveguide_points_path": "data/sample_points_all.npy",  # path of waveguide data

    # Training parameters
    "epochs": 3000,                   # epoch. tips: >=1000 is recommended.
    "batch_size": 256,                # batch size
    "lr": 0.001,                      # learning rate
    # if use pretrained weight, if true, load weight from us `param_path`
    "pretrained": False,
    "param_path": "checkpoints/model_slab_best.ckpt",  # path to load weight

    # Simulation parameters
    "coord_min": [-0.5, -0.5, -0.5],  # minimum x,y,z coordinate
    "coord_max": [0.5, 0.5, 0.5],     # maximum x,y,z coordinate
    "slab_len": 0.2,                  # size of slab
    "eps1": 1.5,                      # permitivity of slab
    "eps0": 1.0,                      # permitivity of vacuum
    "wave_number": 32.0,              # wave number

    # Neural network parameters
    "in_channel": 3,                  # number of input channel
    "out_channel": 3,                 # number of output channel
    "layers": 6,                      # number of layers
    "neurons": 32,                    # number of neuron each layer

    # Evaluate parameter, the `param_path` is defined above.
    "axis_size": 101,                 # number of grid
    "result_save_dir": "result",      # directory to save the result
})

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
config of pdenet
"""
from easydict import EasyDict as ed

train_config = ed({
    "name": "pde_net",
    "log_path": "./logs/result/",
    "summary_dir": "./summary_dir/summary",
    "eval_interval": 10,
    "lr_scheduler_gamma": 0.5,
    "lr": 0.001,
    "save_epoch_interval": 50,
    "mesh_size": 50,
    "solver_mesh_scale": 5,
    "enable_noise": True,
    "start_noise_level": 0.015,
    "end_noise_level": 0.015,
    "variant_coe_magnitude": 1.0,
    "init_freq": 4,
    "batch_size": 16,
    "mindrecord": "src/data.mindrecord",
    "epochs": 500,
    "multi_step": 20,
    "learning_rate_reduce_times": 4,
    "dt": 0.015,
    "kernel_size": 5,
    "max_order": 4,
    "channels": 1,
    "perodic_padding": True,
    "if_frozen": False,
    "enable_moment": True,
})

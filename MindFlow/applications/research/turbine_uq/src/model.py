# Copyright 2024 Huawei Technologies Co., Ltd
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
r"""This module provides a wrapper for different models."""
import os

from omegaconf import DictConfig
from mindspore import nn, load_checkpoint, load_param_into_net, dtype as mstype
from mindflow.cell import FNO2D, UNet2D


def get_model(config: DictConfig,
              compute_type=mstype.float16) -> nn.Cell:
    r"""Get the model according to the config."""
    if config.model_type == "unet":
        model = UNet2D(config.model.in_channels,
                       config.model.out_channels,
                       config.model.channels,
                       data_format='NHWC')

    elif config.model_type == "fno":
        model = FNO2D(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            resolutions=[config.model.resolution[0], config.model.resolution[1]],
            n_modes=[config.model.modes, config.model.modes],
            hidden_channels=config.model.channels,
            n_layers=config.model.depths,
            projection_channels=4 * config.model.channels,
            fno_compute_dtype=compute_type)
    return model


def reload_model(config: DictConfig, ckpt_file_path: str = None):
    """reload model trained model"""
    model = get_model(config, mstype.float32)
    is_exist = os.path.exists(ckpt_file_path)
    if is_exist:
        param_dict = load_checkpoint(ckpt_file_path)
        load_param_into_net(model, param_dict)
    else:
        raise FileNotFoundError(f'the file {ckpt_file_path} is not existed')
    return model

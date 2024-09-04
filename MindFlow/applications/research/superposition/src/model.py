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
"""This module provides a wrapper for diforward_dataerent models."""
import os

from omegaconf import DictConfig
from mindspore import nn, ops, load_checkpoint, load_param_into_net, dtype as mstype
from mindflow.cell import FNO2D, UNet2D


def get_model(config: DictConfig,
              compute_type=mstype.float16) -> nn.Cell:
    """Get the model according to the config."""
    if config.model_type == "unet":
        model = UNet2D(config.in_channels,
                       config.out_channels,
                       config.channels,
                       data_format='NHWC')
    elif config.model_type == "fno":
        model = FNO2D(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            resolutions=[config.resolution[0], config.resolution[1]],
            n_modes=[config.modes, config.modes],
            hidden_channels=config.channels,
            n_layers=config.depths,
            projection_channels=4 * config.channels,
            fno_compute_dtype=compute_type)
    elif config.model_type == "spno":
        pred_model_name = config.spno.pred_model_name
        config_pred = config[pred_model_name].copy()
        config_pred.model_type = pred_model_name
        pred_model = get_model(config_pred, compute_type=compute_type)
        super_model_name = config.spno.super_model_name
        config_super = config[super_model_name].copy()
        config_super.model_type = super_model_name
        super_model = get_model(config_super, compute_type=compute_type)
        model = SPNO2D(pred_model, super_model,
                       in_channels=config.spno.in_channels,
                       win_split=config.spno.patch_num)
    return model


def reload_model(config: DictConfig, ckpt_file_path=None):
    """reload model trained model"""
    model = get_model(config, mstype.float32)
    if not os.path.exists(ckpt_file_path):
        raise FileNotFoundError(f'the file {ckpt_file_path} is not existed')
    param_dict = load_checkpoint(ckpt_file_path)
    load_param_into_net(model, param_dict)
    return model


class SPNO2D(nn.Cell):
    """SuperPosition-based Nerual Operator"""
    def __init__(self, pred_net, super_net, in_channels=10, win_split=1):
        super().__init__()
        self.pred_net = pred_net
        self.super_net = super_net
        self.in_channels = in_channels
        self.win_split = win_split

    def construct(self, x):
        """construct"""
        superpose = not x.shape[-1] == self.in_channels
        if superpose:
            x = ops.concat(ops.chunk(x, 2, axis=-1), axis=0)
        x = self.pred_net(x)
        if superpose:
            x = ops.concat(ops.chunk(x, 2, axis=0), axis=-1)
            x = self.little_windows(x)
            x = self.super_net(x)
            x = self.big_windows(x)
        return x

    def little_windows(self, forward_data):
        """little_windows"""
        num_rows = num_cols = self.win_split
        (b_z, f_rows, f_cols, channel) = forward_data.shape
        forward_data = (forward_data.reshape(b_z, f_rows//num_rows, num_rows, f_cols//num_rows, num_cols, channel)
                        .permute(0, 2, 4, 1, 3, 5)
                        .reshape(b_z*num_rows*num_cols, f_rows//num_rows, f_cols//num_rows, channel))
        return forward_data

    def big_windows(self, forward_data):
        """big_windows"""
        num_rows = num_cols = self.win_split
        (b_z, f_rows, f_cols, channel) = forward_data.shape
        forward_data = (forward_data.reshape(b_z//num_rows//num_cols, num_rows, num_cols, f_rows, f_cols, channel)
                        .permute(0, 3, 1, 4, 2, 5)
                        .reshape(b_z//num_rows//num_cols, f_rows*num_rows, f_cols*num_rows, channel))
        return forward_data

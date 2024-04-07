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
r"""This module provides a wrapper for different models."""
from typing import Optional

from omegaconf import DictConfig

import mindspore as ms
from mindspore import nn
from mindspore import dtype as mstype

from .pdeformer import PDEformer
from .baseline import DeepONet, FNO2D, UNet2D
from ..utils.tools import calculate_num_params
from ..utils.record import Record


def get_model(config: DictConfig,
              record: Optional[Record] = None,
              compute_type=mstype.float16) -> nn.Cell:
    r"""Get the model according to the config."""
    if record is not None:
        record.print(f"model_type: {config.model_type}")

    if config.model_type == "pdeformer":
        model = PDEformer(config.model, compute_dtype=compute_type)
        # load pre-trained checkpoint
        load_ckpt = config.model.get("load_ckpt", "none")
        if load_ckpt.lower() != "none":
            param_dict = ms.load_checkpoint(load_ckpt)
            param_not_load, _ = ms.load_param_into_net(model, param_dict)
            if len(param_not_load) > 0:  # pylint: disable=C1801
                warning_str = ("WARNING: These checkpoint parameters are not loaded: "
                               + str(param_not_load))
                if record is not None:
                    record.print(warning_str)
                else:
                    print(warning_str)
    elif config.model_type == "deeponet":
        model = DeepONet(config.deeponet.trunk_dim_in,
                         config.deeponet.trunk_dim_hidden,
                         config.deeponet.trunk_num_layers,
                         config.deeponet.branch_dim_in,
                         config.deeponet.branch_dim_hidden,
                         config.deeponet.branch_num_layers,
                         dim_out=config.deeponet.dim_out,
                         num_pos_enc=config.deeponet.num_pos_enc,
                         compute_dtype=compute_type)
    elif config.model_type == "fno":
        model = FNO2D(config.fno.in_channels,
                      config.fno.out_channels,
                      config.fno.resolution,
                      config.fno.modes,
                      channels=config.fno.channels,
                      depths=config.fno.depths,
                      mlp_ratio=config.fno.mlp_ratio,
                      compute_dtype=compute_type)
    elif config.model_type == "u-net":
        model = UNet2D(config.unet.in_channels,
                       config.unet.out_channels,
                       compute_dtype=compute_type)
    else:
        raise ValueError(f"The model_type {config.model_type} is not supported!")

    if record is not None:
        record.print("num_parameters: " + calculate_num_params(model))

    return model

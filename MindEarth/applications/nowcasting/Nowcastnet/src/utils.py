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
# ==============================================================================
"""Nowcastnet utils"""
import os

import numpy as np
from mindspore import ops, context, amp
import mindspore.communication.management as D
from mindspore.communication import init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindearth.utils import create_logger

from .generator import GenerationNet
from .discriminator import TemporalDiscriminator
from .evolution import EvolutionNet


def init_data_parallel(use_ascend):
    r"""init data parallel."""
    if use_ascend:
        init()
        device_num = D.get_group_size()
        os.environ['HCCL_CONNECT_TIMEOUT'] = "7200"
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num, parameter_broadcast=False)
    else:
        init("nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          parameter_broadcast=True)


def get_logger(config):
    """Get logger for saving log"""
    summary_params = config.get('summary')
    if not os.path.exists(summary_params.get("summary_dir")):
        os.makedirs(summary_params.get("summary_dir"))
    logger = create_logger(path=os.path.join(summary_params.get("summary_dir"), "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def init_generation_model(config, run_mode='train'):
    """Init generation model """
    train_params = config.get("train")
    data_params = config.get("data")
    g_model = GenerationNet(config)
    d_model = TemporalDiscriminator(data_params.get("t_in", 9) + data_params.get("t_out", 20))
    g_model.set_train(run_mode == 'train')
    d_model.set_train(run_mode == 'train')
    if train_params.get('mixed_precision', False):
        g_model = amp.auto_mixed_precision(g_model, amp_level=train_params.get("amp_level", 'O2'))
        d_model = amp.auto_mixed_precision(d_model, amp_level=train_params.get("amp_level", 'O2'))
    return g_model, d_model


def init_evolution_model(config, run_mode="train"):
    """Init evolution model """
    train_params = config.get("train")
    summary_params = config.get("summary")
    model = EvolutionNet(config.get('data').get("t_in", 9), config.get('data').get("t_out", 20))
    model.set_train(run_mode == 'train')
    if train_params['load_ckpt']:
        params = load_checkpoint(summary_params["evolution_ckpt_path"])
        load_param_into_net(model, params)
    return model


def make_grid(inputs):
    """get 2D grid"""
    batch_size, _, height, width = inputs.shape
    xx = np.arange(0, width).reshape(1, -1)
    xx = np.tile(xx, (height, 1))
    yy = np.arange(0, height).reshape(-1, 1)
    yy = np.tile(yy, (1, width))
    xx = xx.reshape(1, 1, height, width)
    xx = np.tile(xx, (batch_size, 1, 1, 1))
    yy = yy.reshape(1, 1, height, width)
    yy = np.tile(yy, (batch_size, 1, 1, 1))
    grid = np.concatenate((xx, yy), axis=1).astype(np.float32)
    return grid


def warp(inputs, flow, grid, mode="bilinear", padding_mode="zeros"):
    width = inputs.shape[-1]
    vgrid = grid + flow
    vgrid = 2.0 * ops.div(vgrid, max(width - 1, 1)) - 1.0
    vgrid = vgrid.transpose(0, 2, 3, 1)
    output = ops.grid_sample(inputs, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
    return output

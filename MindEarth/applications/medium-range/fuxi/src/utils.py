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
"""fuxi utils"""
import os
import matplotlib.pyplot as plt

import mindspore.common.dtype as mstype
import mindspore.communication.management as D
from mindspore.communication import init
from mindspore import context
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindearth.data import FEATURE_DICT
from mindearth.utils import make_dir, create_logger, plt_metrics

from .fuxi_net import FuXiNet


def init_data_parallel(use_ascend):
    """Init data parallel for model running"""
    if use_ascend:
        init()
        device_num = D.get_group_size()
        os.environ["HCCL_CONNECT_TIMEOUT"] = "7200"
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num, parameter_broadcast=False)
    else:
        init("nccl")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          parameter_broadcast=True)


def init_model(config, run_mode="train"):
    """Init model"""
    data_params = config.get("data")
    model_params = config.get("model")
    train_params = config.get("train")
    summary_params = config.get("summary")
    train_params["load_ckpt"] = run_mode == "test"
    model_params["recompute"] = data_params.get("grid_resolution") < 1.0
    data_params["h_size"], data_params["w_size"] = data_params.get("h_size", 720), data_params.get("w_size", 1440)
    summary_params["summary_dir"] = get_model_summary_dir(config)
    make_dir(os.path.join(summary_params.get("summary_dir"), "image"))
    model = FuXiNet(depths=model_params.get("depths", 18), in_channels=model_params.get("in_channels", 96),
                    out_channels=model_params.get("out_channels", 192), h_size=data_params.get("h_size", 720),
                    w_size=data_params.get("w_size", 1440), level_feature_size=data_params.get("level_feature_size", 5),
                    pressure_level_num=data_params.get("pressure_level_num", 13),
                    surface_feature_size=data_params.get("surface_feature_size", 4))
    if train_params.get('load_ckpt'):
        params = load_checkpoint(summary_params.get("ckpt_path"))
        load_param_into_net(model, params)
    return model


def get_model_summary_dir(config):
    """Get model summary directory"""
    model_params = config.get('model')
    model_name = model_params.get('name')
    summary_dir = model_name
    optimizer_params = config.get('optimizer')
    train_params = config.get('train')
    for k in model_params.keys():
        if k == 'name':
            continue
        summary_dir += '_{}_{}'.format(k, str(model_params[k]))
    summary_dir += '_{}'.format(optimizer_params.get('name'))
    summary_dir += '_{}'.format(train_params.get('name'))
    return summary_dir


def get_logger(config):
    """Get logger for saving log"""
    summary_params = config.get('summary')
    logger = create_logger(path=os.path.join(summary_params.get("summary_dir"), "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def amp_convert(network, black_list=None):
    """Do keep cell fp32."""
    network.to_float(mstype.float16)
    for _, cell in network.cells_and_names():
        if isinstance(cell, black_list):
            cell.to_float(mstype.float32)


def _get_absolute_idx(feature_tuple, pressure_level_num):
    """Get absolute index in metrics"""
    return feature_tuple[1] * pressure_level_num + feature_tuple[0]


def plt_key_info(key_info, config, epochs=1, metrics_type='RMSE', loc='upper right'):
    """ Visualize the rmse or acc results, metrics_type is 'Acc' or 'RMSE' """
    data_params = config.get('data')
    summary_params = config.get('summary')
    pred_lead_time = data_params.get('pred_lead_time', 6)
    x = range(pred_lead_time, data_params.get('t_out_valid', 20) * pred_lead_time + 1, pred_lead_time)
    z500_idx = _get_absolute_idx(FEATURE_DICT.get("Z500"), data_params.get('pressure_level_num'))
    t2m_idx = _get_absolute_idx(FEATURE_DICT.get("T2M"), data_params.get('pressure_level_num'))
    t850_idx = _get_absolute_idx(FEATURE_DICT.get("T850"), data_params.get('pressure_level_num'))
    u10_idx = _get_absolute_idx(FEATURE_DICT.get("U10"), data_params.get('pressure_level_num'))
    xaxis_interval = plt.MultipleLocator(24)

    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(2, 2, 1)
    plt_metrics(x, key_info[z500_idx, :], "Z500", "Z500", ylabel=metrics_type, loc=loc)
    ax1.xaxis.set_major_locator(xaxis_interval)
    ax2 = plt.subplot(2, 2, 2)
    plt_metrics(x, key_info[t2m_idx, :], "T2M", "T2M", ylabel=metrics_type, loc=loc)
    ax2.xaxis.set_major_locator(xaxis_interval)
    ax3 = plt.subplot(2, 2, 3)
    plt_metrics(x, key_info[t850_idx, :], "T850", "T850", ylabel=metrics_type, loc=loc)
    ax3.xaxis.set_major_locator(xaxis_interval)
    ax4 = plt.subplot(2, 2, 4)
    plt_metrics(x, key_info[u10_idx, :], "U10", "U10", ylabel=metrics_type, loc=loc)
    ax4.xaxis.set_major_locator(xaxis_interval)
    plt.subplots_adjust(wspace=0.25, hspace=0.6)
    plt.savefig(f"{summary_params.get('summary_dir')}/image/Eval_{metrics_type}_epoch{epochs}.png", bbox_inches="tight")

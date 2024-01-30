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
"""The utils of koopman_vit"""

import os
import numpy as np
import matplotlib.pyplot as plt

import mindspore.communication.management as D

from mindspore import Tensor
from mindspore import context
from mindspore.communication import init
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindearth.cell import ViTKNO
from mindearth.data import SIZE_DICT, FEATURE_DICT
from mindearth.utils import make_dir, create_logger, plt_metrics


def _get_absolute_idx(feature_tuple, pressure_level_num):
    return feature_tuple[1] * pressure_level_num + feature_tuple[0]


def plt_key_info(key_info, config, epochs=1, metrics_type='RMSE', loc='upper right'):
    """ Visualize the rmse or acc results, metrics_type is 'Acc' or 'RMSE' """
    make_dir(f"{config['summary']['summary_dir']}/image/")
    pred_lead_time = config['data'].get('pred_lead_time', 6)
    x = range(pred_lead_time, config['data'].get('t_out_valid', 20)*pred_lead_time + 1, pred_lead_time)
    z500_idx = _get_absolute_idx(FEATURE_DICT.get("Z500"), config['data']['pressure_level_num'])
    t2m_idx = _get_absolute_idx(FEATURE_DICT.get("T2M"), config['data']['pressure_level_num'])
    t850_idx = _get_absolute_idx(FEATURE_DICT.get("T850"), config['data']['pressure_level_num'])
    u10_idx = _get_absolute_idx(FEATURE_DICT.get("U10"), config['data']['pressure_level_num'])

    plt.figure(1, figsize=(14, 7))
    plt.tight_layout()
    plt.subplots(2, 2)
    plt.subplot(2, 2, 1)
    plt_metrics(x, key_info[z500_idx, :], metrics_type + " of Z500", "Z500", loc=loc)
    plt.subplot(2, 2, 2)
    plt_metrics(x, key_info[t2m_idx, :], metrics_type + " of T2M", "T2M", loc=loc)
    plt.subplot(2, 2, 3)
    plt_metrics(x, key_info[t850_idx, :], metrics_type + " of T850", "T850", loc=loc)
    plt.subplot(2, 2, 4)
    plt_metrics(x, key_info[u10_idx, :], metrics_type + " of U10", "U10", loc=loc)
    plt.subplots_adjust(wspace=0.25, hspace=0.6)
    plt.savefig(f"{config['summary']['summary_dir']}/image/Eval_{metrics_type}_epoch{epochs}.png", bbox_inches="tight")


def get_logger(config):
    logger = create_logger(path=os.path.join(config['summary']["summary_dir"], "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def init_data_parallel(use_ascend):
    """Init data parallel for model running"""
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


def init_model(config):
    """Init model"""
    data_params = config["data"]
    model_params = config["model"]
    compute_type = mstype.float32

    model = ViTKNO(image_size=(data_params["h_size"], data_params["w_size"]),
                   in_channels=data_params["feature_dims"],
                   out_channels=data_params["feature_dims"],
                   patch_size=data_params["patch_size"],
                   encoder_depths=model_params["encoder_depth"],
                   encoder_embed_dims=model_params["encoder_embed_dim"],
                   mlp_ratio=model_params["mlp_ratio"],
                   dropout_rate=model_params["dropout_rate"],
                   num_blocks=model_params["num_blocks"],
                   high_freq=True,
                   encoder_network=model_params["encoder_network"],
                   compute_dtype=compute_type)

    if config['train']['load_ckpt']:
        prams = load_checkpoint(config['summary']["ckpt_path"])
        load_param_into_net(model, prams)
    return model


def unpatchify(x, img_size, patch_size):
    feature_num = x.shape[-1] // (patch_size ** 2)
    batch_size = x.shape[0]
    h, w = img_size[0] // patch_size, img_size[1] // patch_size
    x = x.reshape(batch_size, h, w, patch_size, patch_size, feature_num)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    imgs = x.reshape(batch_size, patch_size * h, patch_size * w, feature_num)
    return imgs


def get_model_summary_dir(config):
    """Get model summary directory"""
    model_name = config['model']['name']
    summary_dir = model_name
    for k in config['model'].keys():
        if k == 'name':
            print(str(k))
            continue
        summary_dir += '_{}_{}'.format(k, str(config['model'][k]))
    summary_dir += '_{}'.format(config['optimizer']['name'])
    summary_dir += '_{}'.format(config['train']['name'])
    return summary_dir


def update_config(opt, config):
    """Update config by user specified args"""
    make_dir(opt.output_dir)
    config['data']['data_sink'] = opt.data_sink

    config['train']['distribute'] = opt.distribute
    config['train']['device_id'] = opt.device_id
    if opt.device_target == "GPU":
        opt.amp_level = "O0"
    config['train']['amp_level'] = opt.amp_level
    config['train']['run_mode'] = opt.run_mode
    config['train']['load_ckpt'] = opt.load_ckpt
    if config['train']['run_mode'] == 'test':
        config['train']['load_ckpt'] = True

    config['data']['num_workers'] = opt.num_workers
    config['data']['grid_resolution'] = opt.grid_resolution
    config['data']['h_size'], config['data']['w_size'] = SIZE_DICT[opt.grid_resolution]

    config['optimizer']['epochs'] = opt.epochs
    config['optimizer']['finetune_epochs'] = opt.finetune_epochs
    config['optimizer']['warmup_epochs'] = opt.warmup_epochs
    config['optimizer']['initial_lr'] = opt.initial_lr

    config['summary']["valid_frequency"] = opt.valid_frequency
    summary_dir = get_model_summary_dir(config)
    config['summary']["summary_dir"] = os.path.join(opt.output_dir, summary_dir)
    make_dir(config['summary']["summary_dir"])
    config['summary']["ckpt_path"] = opt.ckpt_path


def load_dir_data(file_dir, file_name, dtype=mstype.int32):
    path = os.path.join(file_dir, file_name)
    return Tensor(np.load(path), dtype)


def get_coe(config):
    data_params = config["data"]
    w_size = data_params["w_size"]
    coe_dir = os.path.join(data_params["root_dir"], "coe")
    sj_std = load_dir_data(coe_dir, 'sj_std.npy', mstype.float32)
    wj = load_dir_data(coe_dir, 'wj.npy', mstype.float32)
    ai = load_dir_data(coe_dir, 'ai_norm.npy', mstype.float32).repeat(
        w_size, axis=-1).reshape(
            (1, -1)).repeat(data_params['batch_size'], axis=0).reshape(-1, 1)
    return sj_std, wj, ai

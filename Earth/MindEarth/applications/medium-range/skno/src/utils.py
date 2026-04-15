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
# ==============================================================================
"""SKNO utils"""
import os

import numpy as np
import mindspore.communication.management as D
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindearth.utils import make_dir, create_logger

from .skno import SKNO


def get_logger(config):
    """Get logger for saving log"""
    logger = create_logger(path=os.path.join(config['summary']["summary_dir"], "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def init_model(config):
    """Init model"""
    data_params = config["data"]
    model_params = config["model"]
    compute_type = mstype.float32

    model = SKNO(image_size=(data_params["h_size"], data_params["w_size"]),
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
    
def get_model_summary_dir(config):
    """Get model summary directory"""
    model_name = config['model']['name']
    summary_dir = model_name
    for k in config['model'].keys():
        if k == 'name':
            continue
        summary_dir += '_{}_{}'.format(k, str(config['model'][k]))
    summary_dir += '_{}'.format(config['optimizer']['name'])
    summary_dir += '_{}'.format(config['train']['name'])
    return summary_dir

def update_config(opt, config):
    """Update config file"""
    cfg_summary = config['summary']
    cfg_train = config['train']
    make_dir(cfg_summary["output_dir"])
    cfg_train['device_id'] = opt.device_id
    cfg_train['run_mode'] = opt.run_mode

    summary_dir = get_model_summary_dir(config)
    cfg_summary["summary_dir"] = os.path.join(cfg_summary["output_dir"], summary_dir)
    make_dir(cfg_summary["summary_dir"])


def load_dir_data(file_dir, file_name, dtype=mstype.int32):
    """Load data"""
    path = os.path.join(file_dir, file_name)
    return Tensor(np.load(path), dtype)



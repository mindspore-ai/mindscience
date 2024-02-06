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
"""ensoforcast utils"""
import os

import numpy as np
import matplotlib.pyplot as plt

import mindspore.dataset as ds
from mindspore.train.serialization import load_checkpoint
from mindearth.utils import create_logger

from .ctefnet import CTEFNet
from .dataset import CMIP5Data, ReanalysisData


def get_logger(config):
    """Get logger for saving log"""
    summary_params = config.get('summary')
    logger = create_logger(path=os.path.join(summary_params.get("summary_dir"), "results.log"))
    for key in config:
        logger.info(config[key])
    return logger


def init_model(config, run_mode='train'):
    """Init model"""
    data_params = config.get("data")
    model_params = config.get("model")
    train_params = config.get("train")
    train_params['load_ckpt'] = run_mode == "test"
    model = CTEFNet(
        cov_hidden_channels=model_params.get('cov_hidden_channels'),
        cov_out_channels=model_params.get('cov_out_channels'),
        heads=model_params.get('heads'),
        num_layer=model_params.get('num_layer'),
        feedforward_dims=model_params.get('feedforward_dims'),
        dropout=model_params.get('dropout'),
        obs_time=data_params.get('obs_time'),
        pred_time=data_params.get('pred_time')
    )
    return model


def init_dataloader(config):
    """Init dataloader"""
    data_params = config.get('data')
    train_type = data_params.get('train_dataset')
    valid_type = data_params.get('valid_dataset')
    assert train_type in ['CMIP5', 'Reanalysis'], 'Unexpected Data Type %s.' % train_type
    assert valid_type in ['CMIP5', 'Reanalysis'], 'Unexpected Data Type %s.' % valid_type
    if train_type == 'CMIP5':
        train_dataset = CMIP5Data(data_params.get('root_dir'), data_params.get('train_period'),
                                  data_params.get('obs_time'), data_params.get('pred_time'))
    else:
        train_dataset = ReanalysisData(data_params.get('root_dir'), data_params.get('train_period'),
                                       data_params.get('obs_time'), data_params.get('pred_time'))
    if valid_type == 'CMIP5':
        valid_dataset = CMIP5Data(data_params.get('root_dir'), data_params.get('valid_period'),
                                  data_params.get('obs_time'), data_params.get('pred_time'))
    else:
        valid_dataset = ReanalysisData(data_params.get('root_dir'), data_params.get('valid_period'),
                                       data_params.get('obs_time'), data_params.get('pred_time'))
    train_dataloader = ds.GeneratorDataset(train_dataset, ["data", "index"], shuffle=True).batch(
        data_params.get('train_batch_size'), False)
    valid_dataloader = ds.GeneratorDataset(valid_dataset, ["data", "index"], shuffle=False).batch(
        data_params.get('valid_batch_size'), False)
    return train_dataloader, valid_dataloader


def get_param_dict(config, current_step):
    """Get param dict when load checkpoint"""
    summary_params = config.get("summary")

    ckpt_path = os.path.join(summary_params.get('summary_dir'), 'ckpt', f'step_{current_step}')
    ckpt_list = os.listdir(ckpt_path)
    ckpt_list.sort()
    ckpt_name = ckpt_list[-1]
    params_dict = load_checkpoint(os.path.join(ckpt_path, ckpt_name))
    return params_dict, ckpt_path


def plot_correlation(config, corr_list):
    """Plot model eval result"""
    n_line = len(corr_list)
    summary_params = config.get('summary')

    n = len(corr_list[0])
    x = np.arange(1, n+1, 1)
    plt.rc('font', size=16)
    plt.figure(figsize=(15, 6), dpi=150)
    plt.plot(x, corr_list[0], color='orangered', linestyle='-', marker='o', markerfacecolor='orangered', linewidth=5,
             label='CTEFNet-pretrain', markersize='8')
    if n_line > 1:
        plt.plot(x, corr_list[1], color='blue', linestyle='-', marker='o', markerfacecolor='blue', linewidth=5,
                 label='CTEFNet-finetune', markersize='8')
    plt.xlabel('Forecast Lead (months)')
    plt.ylabel('Correlation Skill')
    plt.tick_params(labelsize=18)
    my_x_ticks = np.arange(1, n+1, 1)
    my_y_ticks = np.arange(0.1, 1.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.grid(linewidth=0.1)
    plt.legend(ncol=4)
    plt.axhline(0.5, color='black')
    plt.savefig(os.path.join(summary_params.get('summary_dir'),
                             'Forecast_Correlation_Skill' + '.png'))
    plt.show()

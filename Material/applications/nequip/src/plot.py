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
"""plot"""
import logging
import math
import matplotlib.pyplot as plt
import numpy as np


def format_config_string(args, config):
    """format_config_string"""
    config_string = f'Device: {args.device_target}, dtype: {args.dtype}\n'

    config_data = config.get('data')
    for k, v in config_data.items():
        config_string += f'{k}: {v}\n'

    config_optimizer = config.get('optimizer')
    for k, v in config_optimizer.items():
        config_string += f'{k}: {v}\n'

    return config_string


def plot_loss(args, config, loss_train=None, loss_eval=None):
    """plot_loss"""
    save_path = config.get('data').get('save_path')
    num_epochs = config.get('optimizer').get('num_epoch')
    eval_steps = config.get('optimizer').get('eval_steps')
    name = config.get('data').get('path').split('/')[-1]
    epochs_train = np.arange(num_epochs)
    epochs_eval = np.arange(0, num_epochs, eval_steps)
    if loss_train is not None:
        plt.plot(epochs_train, np.array(loss_train))

    if loss_eval is not None:
        plt.plot(epochs_eval, np.array(loss_eval))

    config_str = format_config_string(args, config)

    y_max = min(loss_train[0], 10) * 1.2
    plt.xlim(0, num_epochs)
    plt.ylim(0, y_max)
    plt.grid()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training Process {name}')
    plt.text(num_epochs / 3, y_max / 3, config_str)

    plt.savefig(f'{save_path}/training_process_{name}.png')
    plt.show()


def plot_lr(args, config, learing_rate):
    """plot_lr"""
    save_path = config.get('data').get('save_path')
    optimizer_params = config.get('optimizer')
    data_params = config.get('data')

    total_steps_num = optimizer_params.get('num_epoch') * math.ceil(
        data_params.get('n_train') / data_params.get('batch_size'))

    name = config.get('data').get('path').split('/')[-1]
    steps_train = np.arange(total_steps_num)
    if learing_rate is not None:
        plt.plot(steps_train, np.array(learing_rate))

    config_str = format_config_string(args, config)

    y_max = max(learing_rate) * 1.2
    plt.xlim(0, total_steps_num)
    plt.ylim(0, y_max)
    plt.grid()

    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title(f'Training Process {name}')
    plt.text(total_steps_num / 3, y_max / 3, config_str)

    plt.savefig(f'{save_path}/learning_rate_{name}.png')
    plt.show()


def print_configuration(args, config):
    """print_configuration"""
    logging.info('---- Configuration Summary -----')
    logging.info('Device: %s, dtype: %s', args.device_target, args.dtype)
    for k, v in config.items():
        if not isinstance(v, dict):
            logging.info('%s: %s', k, v)

    logging.info('------ Data configuration ------')
    config_data = config.get('data')
    for k, v in config_data.items():
        logging.info('%s: %s', k, v)

    logging.info('---- Optimizer configuration ---')
    config_optimizer = config.get('optimizer')
    for k, v in config_optimizer.items():
        logging.info('%s: %s', k, v)
    logging.info('--------------------------------')

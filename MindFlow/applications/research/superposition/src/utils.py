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
"""utilization"""
import os
import time
import shutil

import numpy as np
from omegaconf import DictConfig, OmegaConf
from mindspore import nn, Tensor, ops, dtype as mstype
from mindflow import print_log, log_config, get_warmup_cosine_annealing_lr as get_cos_lr

from .model import reload_model
from .dataset import DataNormer, load_dataset


def get_lr(config_train: DictConfig, steps_per_epoch):
    """Get the learning rate according to the config."""
    warmup_epochs = config_train.lr_scheduler.get('warmup_epochs', 10)
    return get_cos_lr(config_train.lr_init, steps_per_epoch, config_train.epochs, warmup_epochs=warmup_epochs)


def get_optimizer(model: nn.Cell, config: dict, steps_per_epoch: int) -> nn.Cell:
    """Get the optimizer according to the config"""
    if config.train.optimizer == 'Adam':
        opt_func = nn.Adam
    elif config.train.optimizer == 'AdamW':
        opt_func = nn.AdamWeightDecay
    lr_var = get_lr(config_train=config.train, steps_per_epoch=steps_per_epoch)
    pred_weight = list(filter(lambda x: 'pred_net' in x.name, model.trainable_params()))
    super_weight = list(filter(lambda x: 'super_net' in x.name, model.trainable_params()))
    params_0 = [{'params': pred_weight, 'weight_decay': 0.01, 'lr': lr_var}]
    params_1 = [{'params': super_weight, 'weight_decay': 0.01, 'lr': [x / 10 for x in lr_var]}]
    optimizer = (opt_func(params_0), opt_func(params_1))
    return optimizer


def calculate_l2_error(label: Tensor, pred: Tensor) -> np.array:
    """Computes the relative L2 loss"""
    error_norm = np.linalg.norm(pred - label, ord=2, axis=1, keepdims=False)
    label_norm = np.linalg.norm(label, ord=2, axis=1, keepdims=False)
    l2_error = error_norm / (label_norm + 1.0e-6)
    l2_error = l2_error.clip(0, 5)
    centered_min = np.percentile(l2_error, 1)
    centered_max = np.percentile(l2_error, 99)
    centered_mean = l2_error.clip(centered_min, centered_max).mean()
    l2_error_dict = {
        "l2_error_mean": l2_error.mean(),
        "l2_error_min": l2_error.min(),
        "l2_error_max": l2_error.max(),
        "l2_error_centered_mean": centered_mean,
        "l2_error_centered_min": centered_min,
        "l2_error_centered_max": centered_max,
    }
    return l2_error_dict


class Record:
    """Record experimental results and various outputs."""
    def __init__(self, root_dir, record_name=None, is_exist=False):
        if record_name is None:
            record_name = time.strftime('%Y-%m-%d-%H-%M-%S')
        if is_exist:
            self.record_path = os.path.join(root_dir)
        else:
            self.record_path = os.path.join(root_dir, record_name)
        self.ckpt_dir = os.path.join(self.record_path, 'ckpt')
        self.npz_dir = os.path.join(self.record_path, 'npz')
        self.image2d_dir = os.path.join(self.record_path, 'image2d')
        self.config = os.path.join(self.record_path, 'config.yaml')
        self.train_save_path = os.path.join(self.npz_dir, 'train.npz')
        self.test_save_path = os.path.join(self.npz_dir, 'test.npz')
        self.ckpt_model = os.path.join(self.ckpt_dir, 'model_last.ckpt')
        os.makedirs(self.record_path, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.image2d_dir, exist_ok=True)
        os.makedirs(self.npz_dir, exist_ok=True)


def init_record(config_file_path, record_name=None):
    '''Initialize the record object'''
    config = OmegaConf.load(config_file_path)
    record = Record(config['record_path'], record_name=record_name)
    shutil.copyfile(config_file_path, os.path.join(record.record_path, "config.yaml"))
    config_str = OmegaConf.to_yaml(config)
    log_config(log_dir=record.record_path, model_name="results")
    print_log('Configuration:\n' + config_str)
    print_log(f'Pid: {os.getpid()}')
    return record


def load_record(record_path: str):
    """load_record"""
    record = Record(record_path, is_exist=True)
    return record


def padding_tensor(xx_inputs, x_norm=None, channel_num=10, shuffle=True, const=350):
    """padding tensor in the channel dim"""
    expand = int(channel_num - xx_inputs.shape[-1])
    if expand <= 0:
        return xx_inputs
    xx_fill = Tensor(np.zeros([*xx_inputs.shape[:-1], expand], dtype=np.float32) + const, dtype=xx_inputs.dtype)
    xx_fill = x_norm.norm(xx_fill)
    xx_inputs = ops.concat((xx_inputs, xx_fill), axis=-1)
    if shuffle:
        for i in range(xx_inputs.shape[0]):
            idx = np.random.permutation(xx_inputs.shape[-1])
            indices = Tensor(idx, dtype=mstype.int32)
            xx_inputs[i] = ops.gather(xx_inputs[i], indices, axis=-1)
    return xx_inputs


def run_inference(config, record):
    """run_inference"""
    model = reload_model(config, ckpt_file_path=record.ckpt_model)
    data_loader_train, data_loader_test = load_dataset(config)
    train_save_dict, test_save_dict = {}, {}
    for super_times in [0, 1]:
        train_rst_list = inference_loop(model, [data_loader_train], super_times=super_times,
                                        is_un_norm=True, is_get_loss=False)
        test_rst_list = inference_loop(model, data_loader_test, super_times=super_times,
                                       is_un_norm=True, is_get_loss=False)
        for i, (true, pred) in enumerate(train_rst_list):
            train_save_dict.update({f'true_hole-{i}_super-{super_times}': true})
            train_save_dict.update({f'pred_hole-{i}_super-{super_times}': pred})
        for i, (true, pred) in enumerate(test_rst_list):
            test_save_dict.update({f'true_hole-{i}_super-{super_times}': true})
            test_save_dict.update({f'pred_hole-{i}_super-{super_times}': pred})
    np.savez(record.train_save_path, **train_save_dict)
    np.savez(record.test_save_path, **test_save_dict)


def inference_loop(model, dataset_list, super_times=0, is_un_norm=False, is_get_loss=True):
    """inference_loop"""
    model.set_train(False)
    true_list, pred_list, rst_list = [], [], []
    x_norm = DataNormer(data_type='x_norm')
    for dataset_iter in dataset_list:
        pred, true = [], []
        for inputs, outputs in dataset_iter:
            channels = model.in_channels * (2**super_times)
            inputs = padding_tensor(inputs.astype(mstype.float32), x_norm=x_norm, channel_num=channels)
            pred.append(model(inputs).asnumpy())
            true.append(outputs.asnumpy())
        true_list.append(np.concatenate(true, axis=0))
        pred_list.append(np.concatenate(pred, axis=0))
    for pred, true in zip(true_list, pred_list):
        if is_un_norm:
            y_norm = DataNormer(data_type='y_norm')
            true = y_norm.un_norm(true)
            pred = y_norm.un_norm(pred)
            rst_list.append((true, pred))
        if is_get_loss:
            l2_error = calculate_l2_error(true, pred)['l2_error_centered_mean']
            rst_list.append(l2_error)
    return rst_list

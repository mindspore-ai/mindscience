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
r"""This module provides different learning rate scheduler."""
import os
import time
import shutil
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf
from mindspore import nn, Tensor, ops, dtype as mstype
from mindflow import log_config, print_log, get_warmup_cosine_annealing_lr as get_cos_lr

from .model import reload_model
from .dataset import DataNormer, load_dataset


def get_lr(config_train: DictConfig, steps_per_epoch: int):
    r"""Get the learning rate according to the config."""
    epochs = config_train.epochs
    lr_init = config_train.lr_init
    warmup_epochs = config_train.lr_scheduler.get('warmup_epochs', 10)
    return get_cos_lr(lr_init, steps_per_epoch, epochs, warmup_epochs=warmup_epochs, eta_min=lr_init*0.01)


def get_optimizer(model: nn.Cell, config: dict, steps_per_epoch: int,) -> nn.Cell:
    """Get the optimizer according to the config."""
    lr_var = get_lr(config_train=config.train, steps_per_epoch=steps_per_epoch)
    params = [{'params': model.trainable_params(), 'lr': lr_var, 'weight_decay': config.train.weight_decay}]
    if config.train.optimizer == 'Adam':
        optimizer = nn.Adam(params)
    elif config.train.optimizer == 'AdamW':
        optimizer = nn.AdamWeightDecay(params)
    return optimizer


def repeat_tensor(x: Tensor, grid_shape: List[int] = (64, 128)):
    """repeat_tensor"""
    x = ops.expand_dims(x, axis=1)
    x = ops.expand_dims(x, axis=2)
    x = ops.tile(x, (1, grid_shape[0], grid_shape[1], 1))
    return x


def calculate_l2_error(label: Tensor, pred: Tensor) -> np.array:
    """Computes the relative L2 loss."""
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
        if not record_name:
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
    """Initialize the record object."""
    config = OmegaConf.load(config_file_path)
    record = Record(config['record_path'],
                    record_name=record_name,
                    is_exist=False)
    shutil.copyfile(config_file_path, os.path.join(record.record_path, "config.yaml"))
    config_str = OmegaConf.to_yaml(config)
    log_config(record.record_path, model_name="results")
    print_log('Configuration:\n' + config_str)
    print_log(f'Pid: {os.getpid()}')
    return record


def load_record(record_path):
    """load_record"""
    record = Record(record_path, is_exist=True)
    return record


def run_inference(config, record):
    """run_inference"""
    model = reload_model(config, ckpt_file_path=record.ckpt_model)
    y_norm = DataNormer(data_type='y_norm')
    data_loader_train, data_loader_test = load_dataset(config)
    (true_train, pred_train) = inference_loop(model, data_loader_train, y_norm=y_norm,
                                              is_un_norm=True, is_get_loss=False)
    np.savez(record.train_save_path, **{'true': true_train, 'pred': pred_train})
    (true_test, pred_test) = inference_loop(model, data_loader_test, y_norm=y_norm,
                                            is_un_norm=True, is_get_loss=False)
    np.savez(record.test_save_path, **{'true': true_test, 'pred': pred_test})


def inference_loop(model, dataset, y_norm=None, is_un_norm=False, is_get_loss=True):
    """inference_loop"""
    pred, true = [], []
    for inputs, outputs in dataset:
        inputs = repeat_tensor(inputs).astype(mstype.float32)
        pred.append(model(inputs).asnumpy())
        true.append(outputs.asnumpy())
    true = np.concatenate(true, axis=0)
    pred = np.concatenate(pred, axis=0)
    if is_un_norm:
        true = y_norm.un_norm(true)
        pred = y_norm.un_norm(pred)
    if is_get_loss:
        l2_error = calculate_l2_error(true, pred)['l2_error_centered_mean']
        return l2_error
    return (true, pred)

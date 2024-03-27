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
r"""This module provides different learning rate scheduler."""
from typing import List, Optional
from omegaconf import DictConfig
import numpy as np
from mindspore import nn


def linear_warmup_lr(lr_list: List[float],
                     steps_per_epoch: int,
                     lr_warmup_epochs: int,
                     max_lr: float) -> List[float]:
    r"""Generates a linear warmup learning rate schedule."""
    lr_ = np.linspace(0, max_lr, num=steps_per_epoch*lr_warmup_epochs, endpoint=True)
    size = len(lr_list)
    lr_ = lr_.tolist() + lr_list
    lr_ = lr_[:size]
    return lr_


def get_lr_list(steps_per_epoch: int,
                epochs: int,
                lr_init: float,
                lr_scheduler_type: str,
                lr_milestones: Optional[List[float]] = None,
                lr_decay: float = 0.2,
                enable_warmup: bool = False,
                warmup_epochs: int = 10) -> List[float]:
    r"""
    Generates a learning rate schedule based on the specified scheduler type.
    This function supports 'multi_step' or 'cosine_decay' learning rate schedulers, and
    can also perform warmup before the learning rate scheduler.

    Args:
        steps_per_epoch (int): The number of steps per epoch.
        epochs (int): The total number of epochs.
        lr_init (float): The initial learning rate.
        lr_scheduler_type (str): The type of learning rate scheduler. It should be one of
            'multi_step', 'cosine_decay', 'warmup_cosine_decay', or 'warmup_multi_step'.
        lr_milestones (list, optional): A list of milestones at which the learning rate is
            reduced. Default: [0.6, 0.8, 1.0].
        lr_decay (float): The factor by which the learning rate is decayed at each milestone.
            Default: 0.2.
        enable_warmup (bool): Whether to enable warmup. Default: False.
        warmup_epochs (int): The number of epochs over which the learning rate warms up to `lr_init`.
            Default: 0.

    Returns:
        list: A list of learning rates corresponding to each step in the training process.
    """

    total_steps = epochs * steps_per_epoch
    if lr_milestones is None:
        lr_milestones = [0.6, 0.8, 1.0]
    if lr_scheduler_type in ['piecewise_constant', 'multi_step', 'mstep']:
        milestones = [int(total_steps * x) for x in lr_milestones]
        learning_rates = [lr_init * (lr_decay**x) for x in range(len(milestones))]
        lr_list = nn.piecewise_constant_lr(milestones, learning_rates)
    elif lr_scheduler_type in ['cosine_decay', 'cos']:
        lr_list = nn.cosine_decay_lr(min_lr=0.01 * lr_init,
                                     max_lr=lr_init,
                                     total_step=total_steps,
                                     step_per_epoch=steps_per_epoch,
                                     decay_epoch=int(epochs*0.9))
    else:
        raise ValueError(
            f"The type of lr_scheduler should be in the set of {{'piecewise_constant', 'multi_step',"
            f"'mstep', 'cosine_decay', 'cos'}}, but got {lr_scheduler_type}")

    if enable_warmup:
        lr_list = linear_warmup_lr(lr_list, steps_per_epoch, warmup_epochs, lr_init)

    return lr_list


def get_lr(steps_per_epoch: int, config_train: DictConfig) -> List[float]:
    r"""Get the learning rate according to the config."""
    epochs = config_train.epochs
    lr_init = config_train.lr_init
    lr_scheduler_type = config_train.lr_scheduler.get('type', 'cos')
    lr_milestones = config_train.lr_scheduler.get('milestones', [0.6, 0.8, 1.0])
    lr_decay = config_train.lr_scheduler.get('decay', 0.5)
    enable_warmup = config_train.lr_scheduler.get('enable_warmup', False)
    warmup_epochs = config_train.lr_scheduler.get('warmup_epochs', 10)

    lr_list = get_lr_list(steps_per_epoch,
                          epochs,
                          lr_init,
                          lr_scheduler_type,
                          lr_milestones=lr_milestones,
                          lr_decay=lr_decay,
                          enable_warmup=enable_warmup,
                          warmup_epochs=warmup_epochs)

    return lr_list

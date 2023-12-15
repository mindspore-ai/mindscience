# ============================================================================
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
"""
learning rate module
"""
import numpy as np


class _LRScheduler():
    """_LRScheduler"""
    def __init__(self, learning_rate, max_epoch, steps_per_epoch):
        self.base_lr = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = int(max_epoch * steps_per_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        raise NotImplementedError


class _WarmUp():
    """_WarmUp"""
    def __init__(self, warmup_init_lr):
        self.warmup_init_lr = warmup_init_lr

    def get_lr(self, current_step):
        """
        Get learning rate during warmup
        """
        raise NotImplementedError


class _LinearWarmUp(_WarmUp):
    """
    linear warmup function
    """
    def __init__(self, lr, warmup_epochs, steps_per_epoch, warmup_init_lr=0):
        self.base_lr = lr
        self.warmup_init_lr = warmup_init_lr
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)
        super(_LinearWarmUp, self).__init__(warmup_init_lr)

    def get_warmup_steps(self):
        """get_warmup_steps"""
        return self.warmup_steps

    def get_lr(self, current_step):
        """get_lr"""
        lr_inc = (float(self.base_lr) - float(self.warmup_init_lr)) / float(self.warmup_steps)
        learning_rate = float(self.warmup_init_lr) + lr_inc * current_step
        return learning_rate


class StepLR(_LRScheduler):
    """Decays the learning rate by gamma every epoch_size epochs.

    Args:
        lr (float): Initial learning rate which is the
            lower boundary in the cycle.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        max_epoch (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle.
        epoch_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        warmup_epochs (int): The number of epochs to Warmup.
            Default: 0
    """
    def __init__(self, lr, epoch_size, gamma, steps_per_epoch, max_epoch, warmup_epochs=0):
        self.epoch_size = epoch_size
        self.gamma = gamma
        self.warmup = _LinearWarmUp(lr, warmup_epochs, steps_per_epoch)
        super(StepLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        warmup_steps = self.warmup.get_warmup_steps()

        lr_each_step = []
        for i in range(self.total_steps):
            if i < warmup_steps:
                learning_rate = self.warmup.get_lr(i+1)
            else:
                cur_ep = i // self.steps_per_epoch
                learning_rate = self.base_lr * self.gamma**(cur_ep // self.epoch_size)

            lr_each_step.append(learning_rate)
        return np.array(lr_each_step).astype(np.float32)
    
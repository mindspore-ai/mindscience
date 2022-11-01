# Copyright 2020 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""Learning rate scheduler."""
from collections import Counter
import numpy as np


class _LRScheduler:
    """Basic class for learning rate scheduler"""

    def __init__(self, lr, max_epoch, steps_per_epoch):
        self.base_lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = int(max_epoch * steps_per_epoch)

    def get_lr(self):
        raise NotImplementedError


class MultiStepLR(_LRScheduler):
    """Multi-step learning rate scheduler"""

    def __init__(self, lr, milestones, gamma, steps_per_epoch, max_epoch):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(lr, max_epoch, steps_per_epoch)

    def get_lr(self):
        lr_each_step = []
        current_lr = self.base_lr
        for i in range(self.total_steps):
            cur_ep = i // self.steps_per_epoch
            if i % self.steps_per_epoch == 0 and cur_ep in self.milestones:
                current_lr = current_lr * self.gamma
            lr = current_lr
            lr_each_step.append(lr)
        return np.array(lr_each_step).astype(np.float32)

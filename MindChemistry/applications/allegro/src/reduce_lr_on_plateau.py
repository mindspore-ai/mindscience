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
"""ReduceLROnPlateau
"""

import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.ops import operations as P


literal_min = 'min'

class ReduceLROnPlateau:
    """ReduceLROnPlateau
    """

    def __init__(
            self,
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-8
    ):

        if factor >= 1.0:
            raise ValueError("The lr factor should be less than 1.0.")
        self.factor = factor
        self.optimizer = optimizer
        self.min_lr = Tensor(min_lr, mstype.float32)
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.eps = eps
        self.mode_worse = None
        self.assign = P.Assign()
        self.cast = P.Cast()
        self.last_epoch = Parameter(Tensor(0, dtype=mstype.int32), name='last_epoch_' + self.__class__.__name__)

        if self.mode not in {literal_min, 'max'}:
            raise ValueError(f"`mode` should be 'min' or 'max', but got {self.mode}.")
        if self.threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f"`threshold mode` should be 'rel' or 'abs', but got {self.threshold_mode}.")

        if self.mode == literal_min:
            self.mode_worse = float("inf")
        else:
            self.mode_worse = float("-inf")

        self.best = Parameter(Tensor(self.mode_worse, dtype=mstype.float32), name='best')

        self.cooldown_counter = Parameter(Tensor(0, dtype=mstype.float32), name='cooldown_counter')
        self.wait = Parameter(Tensor(0, dtype=mstype.float32), name='wait')
        self.increase_tensor = Tensor(1, mstype.int32)
        self._last_lr = optimizer.learning_rate.value()

    @property
    def in_cooldown(self):
        """ Whether in cooldown period. """
        return self.cooldown_counter > 0

    def get_last_lr(self):
        """
        Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def step(self, metrics):
        """
        Get the current learning rate and change the learning rate.

        Args:
            metrics(Union(int, float)): the evaluation metrics.
        """
        epoch = self.last_epoch + 1
        current = self.cast(metrics, mstype.float32)
        self.assign(self.last_epoch, epoch)

        if self._is_improvement(current, self.best):
            ops.assign(self.best, current)
            ops.assign(self.wait, 0)
        else:
            ops.assign_add(self.wait, self.increase_tensor)

        if self.in_cooldown:
            ops.assign_sub(self.cooldown_counter, self.increase_tensor)
            ops.assign(self.wait, 0)

        if self.wait > self.patience:
            self._reduce_lr()
            ops.assign(self.cooldown_counter, self.cooldown)
            ops.assign(self.wait, 0)

        return True

    def _is_improvement(self, current, best):
        """ Whether current metric value is better than best. """
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            benchmark = best * rel_epsilon
            return current < benchmark

        if self.mode == 'min' and self.threshold_mode == 'abs':
            benchmark = best - self.threshold
            return current < benchmark

        if self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            benchmark = best * rel_epsilon
            return current > benchmark

        benchmark = best + self.threshold
        return current > benchmark

    def _reduce_lr(self):
        old_lr = self._last_lr
        new_lr = ops.maximum(old_lr * self.factor, self.min_lr)
        if old_lr > new_lr + self.eps:
            ops.assign(self._last_lr, new_lr)
            self.optimizer.learning_rate.set_data(new_lr)
        return True

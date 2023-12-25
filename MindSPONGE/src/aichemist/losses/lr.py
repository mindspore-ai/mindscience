# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
Learning rate schedule for optimizer
"""
import numpy as np
import mindspore as ms
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import functional as F
from mindspore import _checkparam as Validator


__all__ = [
    "TransformerLR",
]


class TransformerLR(LearningRateSchedule):
    r"""A transformer type dynamic learning rate schedule.

    Args:
        learning_rate (float):  Reference learning rate. Default: 1.0

        warmup_steps (int):     Warm up steps. Default: 4000

        dimension (int):        Dimension of output Tensor. Default: 1

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 learning_rate: float = 1.0,
                 warmup_steps: int = 4000,
                 dimension: int = 1,
                 ):

        super().__init__()
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float.")
        Validator.check_non_negative_float(learning_rate, "learning_rate", self.cls_name)
        Validator.check_positive_int(warmup_steps, 'warmup_steps', self.cls_name)

        self.learning_rate = learning_rate

        self.warmup_steps = ms.Tensor(warmup_steps, ms.float32)
        self.dim_scale = dimension ** -0.5

    def construct(self, global_step: int):
        """Calculate the learning rate at current step.

        Args:
            global_step (int):  Global training step.

        Returns:
            lr (Tensor):   Current learning rate.

        """
        step_num = F.reshape(F.cast(global_step, ms.float32), ())
        warmup_scale = self.warmup_steps ** -1.5
        lr1 = step_num ** -0.5
        lr2 = step_num * warmup_scale
        lr_percent = self.dim_scale * F.minimum(lr1, lr2)
        return self.learning_rate * lr_percent


def cos_decay_lr(start_step, lr_init, lr_min, lr_max, decay_steps, warmup_steps):
    """cosine decay learning rate

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    lr_each_step = []
    for i in range(decay_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            lr = lr_min + 0.5 * (lr_max-lr_min) * \
                (1 + np.cos((i - warmup_steps) / decay_steps * np.pi))
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step[start_step:]

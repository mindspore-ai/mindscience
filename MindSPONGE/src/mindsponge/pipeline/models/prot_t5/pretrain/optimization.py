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
"""create optimizer"""
import mindspore
from mindspore import ops, nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

from mindspore.nn.optim import AdaFactor


class WarmUpPolynomialDecayLR(LearningRateSchedule):
    """Polynomia Decay LR with Warmup"""
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = max(warmup_steps, 1)
        self.end_learning_rate = end_learning_rate
        self.decay_steps = decay_steps
        self.power = power

    def construct(self, global_step):
        """construct"""
        # warmup lr
        warmup_percent = global_step.astype(mindspore.float32) / self.warmup_steps
        warmup_learning_rate = self.learning_rate * warmup_percent
        # polynomial lr
        global_step = ops.minimum(global_step, self.decay_steps)
        decayed_learning_rate = (self.learning_rate - self.end_learning_rate) * \
                                ops.pow((1 - global_step / self.decay_steps), self.power) + \
                                self.end_learning_rate
        is_warmup = (global_step < self.warmup_steps).astype(mindspore.float32)
        learning_rate = ((1.0 - is_warmup) * decayed_learning_rate + is_warmup * warmup_learning_rate)
        return learning_rate


def create_optimizer(model, init_lr, optim_type, weight_decay=0.0):
    """create optimizer"""
    if optim_type == 'adafactor':
        optim = AdaFactor(model.trainable_params())
    elif weight_decay > 0:
        optim = nn.AdamWeightDecay(model.trainable_params(), init_lr, weight_decay=weight_decay)
    else:
        optim = nn.Adam(model.trainable_params(), learning_rate=init_lr)

    return optim

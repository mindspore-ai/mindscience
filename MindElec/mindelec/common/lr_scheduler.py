# Copyright 2021 Huawei Technologies Co., Ltd
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
"""lr scheduler"""
import numpy as np

from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from ..architecture.util import check_mode


class LearningRate(LearningRateSchedule):
    r"""
    Warmup-decay learning rate.

    Args:
        learning_rate (float): positive float type number of basic learning rate.
        end_learning_rate (float): non-negtive float type number of end learning rate.
        warmup_steps (int): non-negtive int type number of warmup steps.
        decay_steps (int): A positive int value used to calculate decayed learning rate.
        power (float): A positive float value used to calculate decayed learning rate.

    Inputs:
       Tesnor. The current step number with shape `()`.

    Returns:
       Tensor. The learning rate value for the current step with shape `()`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindelec.common import LearningRate
        >>> from mindspore.common.tensor import Tensor
        >>> from mindspore.common import dtype as mstype
        >>> lr = LearningRate(0.1, 0.001, 0, 10, 0.5)
        >>> print(lr(Tensor(1000, mstype.int32)))
        0.001
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(LearningRate, self).__init__()
        check_mode("LearningRate")
        _check_type(learning_rate, "learning_rate", float, thresh_hold=0.0, restrict=True)
        _check_type(end_learning_rate, "end_learning_rate", float, thresh_hold=0.0, restrict=False)
        _check_type(warmup_steps, "warmup_steps", int, thresh_hold=0, restrict=False)
        _check_type(decay_steps, "decay_steps", int, thresh_hold=0, restrict=True)
        _check_type(power, "power", float, thresh_hold=0.0, restrict=True)

        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power):
    """
    generate learning rate array

    Args:
       global_step(int): current step number, non-negtive int value.
       lr_init(float): init learning rate, positive float value.
       lr_end(float): end learning rate, non-negtive float value.
       lr_max(float): max learning rate, positive float value.
       warmup_steps(int): number of warmup epochs, non-negtive int value.
       total_steps(int): total epoch of training, positive int value.
       poly_power(float): poly learning rate power, positive float value.

    Returns:
       Numpy.array, learning rate array

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindelec.common import get_poly_lr
        >>> learning_rate = get_poly_lr(100, 0.001, 0.1, 0.0001, 1000, 10000, 0.5)
        >>> print(learning_rate.shape)
        (9900,)
    """
    _check_type(global_step, "global_step", int, thresh_hold=0, restrict=False)
    _check_type(lr_init, "lr_init", float, thresh_hold=0.0, restrict=True)
    _check_type(lr_end, "lr_end", float, thresh_hold=0.0, restrict=False)
    _check_type(lr_max, "lr_max", float, thresh_hold=0.0, restrict=True)
    _check_type(warmup_steps, "warmup_steps", int, thresh_hold=0, restrict=False)
    _check_type(total_steps, "total_steps", int, thresh_hold=0, restrict=True)
    _check_type(poly_power, "poly_power", float, thresh_hold=0.0, restrict=True)

    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max - lr_end) * (base ** poly_power)
            lr = lr + lr_end
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)

    learning_rate = np.array(lr_each_step).astype(np.float32)
    current_step = global_step
    learning_rate = learning_rate[current_step:]
    return learning_rate


def _check_type(param, param_name, param_type, thresh_hold=0, restrict=False):
    if not isinstance(param, param_type):
        raise TypeError("the type of {} should be {}, but got {}".format(param_name, param_type, type(param)))
    if restrict:
        if param <= thresh_hold:
            raise ValueError("the value of {} should be > {}, but got: {}".format(param_name, thresh_hold, param))
    else:
        if param < thresh_hold:
            raise ValueError("the value of {} should be >= {}, but got: {}".format(param_name, thresh_hold, param))

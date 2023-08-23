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
# pylint: disable=C1801
"""lr scheduler"""
import math

import numpy as np

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator as validator
except ImportError:
    import mindspore._checkparam as validator


def _get_linear_warmup_lr(warmup_steps, lr_end, lr_init=0.0):
    """warmup lr"""
    lr_inc = (float(lr_end) - float(lr_init)) / float(warmup_steps)
    lr = [float(lr_init) + lr_inc * (i + 1) for i in range(warmup_steps)]
    return lr


def _get_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, eta_min=1e-6):
    """cosine annealing lr"""
    total_steps = last_epoch * steps_per_epoch
    delta = 0.5 * (lr_init - eta_min)
    lr = []
    for i in range(total_steps):
        tmp_epoch = min(math.floor(i / steps_per_epoch), last_epoch)
        lr.append(eta_min + delta * (1 + math.cos(math.pi * tmp_epoch / last_epoch)))
    return lr


def get_warmup_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch,
                                   warmup_epochs=0, warmup_lr_init=0.0, eta_min=1e-6):
    r"""
    Calculates learning rate base on cosine decay function. If warmup epoch is specified, the warmup epoch will be
    warmed up by linear annealing.

    For the i-th step, the formula of computing cosine decayed_learning_rate[i] is:

    .. math::
        decayed\_learning\_rate[i] = eta\_min + 0.5 * (lr\_init - eta\_min) *
        (1 + cos(\frac{current\_epoch}{last\_epoch}\pi))

    Where :math:`current\_epoch = floor(\frac{i}{steps\_per\_epoch})`.

    If warmup epoch is specified, for the i-th step in waramup epoch, the formula of computing
     warmup_learning_rate[i] is:

    .. math::
        warmup\_learning\_rate[i] = (lr\_init - warmup\_lr\_init) * i / warmup\_steps + warmup\_lr\_init

    Args:
        lr_init (float): init learning rate, positive float value.
        steps_per_epoch (int): number of steps to each epoch, positive int value.
        last_epoch (int): total epoch of training, positive int value.
        warmup_epochs (int): total epoch of warming up, default:0.
        warmup_lr_init (float): warmup init learning rate, default:0.0.
        eta_min (float): minimum learning rate, default: 1e-6.

    Returns:
        Numpy.array, learning rate array.

    Raises:
        TypeError: If `lr_init` or `warmup_lr_init` or `eta_min` is not a float.
        TypeError: If `steps_per_epoch` or `warmup_epochs` or `last_epoch` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindearth import get_warmup_cosine_annealing_lr
        >>> lr_init = 0.001
        >>> steps_per_epoch = 3
        >>> last_epoch = 5
        >>> warmup_epochs = 1
        >>> lr = get_warmup_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, warmup_epochs=warmup_epochs)
        >>> print(lr)
        [3.3333333e-04 6.6666666e-04 1.0000000e-03 9.0460398e-04 9.0460398e-04
         9.0460398e-04 6.5485400e-04 6.5485400e-04 6.5485400e-04 3.4614600e-04
         3.4614600e-04 3.4614600e-04 9.6396012e-05 9.6396012e-05 9.6396012e-05]
    """
    validator.check_positive_float(lr_init, arg_name="lr_init")
    validator.check_non_negative_float(warmup_lr_init, arg_name="warmup_lr_init")
    validator.check_non_negative_float(eta_min, arg_name="eta_min")
    validator.check_non_negative_int(warmup_epochs, arg_name="warmup_epochs")
    validator.check_positive_int(steps_per_epoch, arg_name="steps_per_epoch")
    validator.check_positive_int(last_epoch, arg_name="last_epoch")

    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_lr_list = []
    if warmup_epochs != 0:
        warmup_lr_list += _get_linear_warmup_lr(warmup_steps, lr_init, warmup_lr_init)

    cosine_lr_list = _get_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, eta_min=eta_min)

    lr_each_step = warmup_lr_list + cosine_lr_list[warmup_steps:]

    return np.array(lr_each_step).astype(np.float32)

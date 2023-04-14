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
import bisect
import math

import numpy as np

from ..utils.check_func import check_lr_param_type_value, check_param_type


def get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power):
    r"""
    Generate polynomial decay learning rate array.
    The learning rate decays in a polynomial manner as training goes along.
    it follows :math:`lr = step * (lr_max - lr_init)/warmup_steps` ,
    then :math:`lr = lr_end + (lr_max - lr_end) * [(1- i + step)/(total_steps - warmup_steps)]**poly_power`

    Args:
        global_step (int): current step number, non-negtive int value.
        lr_init (float): init learning rate, positive float value.
        lr_end (float): end learning rate, non-negtive float value.
        lr_max (float): max learning rate, positive float value.
        warmup_steps (int): number of warmup epochs, non-negtive int value.
        total_steps (int): total epoch of training, positive int value.
        poly_power (float): poly learning rate power, positive float value.

    Returns:
        Numpy.array, learning rate array.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.common import get_poly_lr
        >>> learning_rate = get_poly_lr(100, 0.001, 0.1, 0.0001, 1000, 10000, 0.5)
        >>> print(learning_rate.shape)
        (9900,)
    """
    check_lr_param_type_value(global_step, "global_step", int, thresh_hold=0, restrict=False, exclude=bool)
    check_lr_param_type_value(lr_init, "lr_init", float, thresh_hold=0.0, restrict=True)
    check_lr_param_type_value(lr_end, "lr_end", float, thresh_hold=0.0, restrict=False)
    check_lr_param_type_value(lr_max, "lr_max", float, thresh_hold=0.0, restrict=True)
    check_lr_param_type_value(warmup_steps, "warmup_steps", int, thresh_hold=0, restrict=False, exclude=bool)
    check_lr_param_type_value(total_steps, "total_steps", int, thresh_hold=0, restrict=True, exclude=bool)
    check_lr_param_type_value(poly_power, "poly_power", float, thresh_hold=0.0, restrict=True)

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


def get_multi_step_lr(lr_init, milestones, gamma, steps_per_epoch, last_epoch):
    r"""
    Generate decay learning rate array of each parameter group by gamma once the
    number of epoch reaches one of the milestones.

    Calculate learning rate by the given `milestone` and `lr_init`. Let the value of `milestone` be
    :math:`(M_1, M_2, ..., M_t, ..., M_N)` and the value of `lr_init` be :math:`(x_1, x_2, ..., x_t, ..., x_N)`.
    N is the length of `milestone`. Let the output learning rate be `y`, then for the i-th step, the formula of
    computing decayed_learning_rate[i] is:

    .. math::
        y[i] = x_t,\ for\ i \in [M_{t-1}, M_t)

    Args:
        lr_init (float): init learning rate, positive float value.
        milestones (Union[list[int], tuple[int]]): list of epoch indices, each element in the list must be greater than
            0.
        gamma (float): multiplicative factor of learning rate decay.
        steps_per_epoch (int): number of steps to each epoch, positive int value.
        last_epoch (int): total epoch of training, positive int value.

    Returns:
        Numpy.array, learning rate array.

    Raises:
        TypeError: If `lr_init` or `gamma` is not a float.
        TypeError: If `steps_per_epoch` or `last_epoch` is not an int.
        TypeError: If `milestones` is neither a tuple nor a list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindflow import get_multi_step_lr
        >>> lr_init = 0.001
        >>> milestones = [2, 4]
        >>> gamma = 0.1
        >>> steps_per_epoch = 3
        >>> last_epoch = 5
        >>> lr = get_multi_step_lr(lr_init, milestones, gamma, steps_per_epoch, last_epoch)
        >>> print(lr)
        [1.e-03 1.e-03 1.e-03 1.e-03 1.e-03 1.e-03 1.e-04 1.e-04 1.e-04 1.e-04 1.e-04 1.e-04 1.e-05 1.e-05 1.e-05]
    """

    check_lr_param_type_value(lr_init, "lr_init", float, thresh_hold=0.0, restrict=True)
    check_lr_param_type_value(gamma, "gamma", float, thresh_hold=0.0, restrict=True)
    check_lr_param_type_value(steps_per_epoch, "steps_per_epoch", int, thresh_hold=0, restrict=True)
    check_lr_param_type_value(last_epoch, "last_epoch", int, thresh_hold=0, restrict=True)
    check_param_type(milestones, "milestones", [list, tuple])

    ordered_milestones = sorted(milestones)
    idx = bisect.bisect_left(ordered_milestones, last_epoch)
    new_milestones = ordered_milestones[:idx]
    new_milestones.append(last_epoch)
    step_milestones = [it * steps_per_epoch for it in new_milestones]

    lr = []
    last_item = 0
    last_lr = lr_init / gamma
    for item in step_milestones:
        cur_lr = last_lr * gamma
        lr += [cur_lr] * (item - last_item)
        last_item = item
        last_lr = cur_lr

    return np.array(lr).astype(np.float32)


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
        warmup_epochs (int): total epoch of warming up, default: ``0``.
        warmup_lr_init (float): warmup init learning rate, default: ``0.0``.
        eta_min (float): minimum learning rate, default: ``1e-6``.

    Returns:
        Numpy.array, learning rate array.

    Raises:
        TypeError: If `lr_init` or `warmup_lr_init` or `eta_min` is not a float.
        TypeError: If `steps_per_epoch` or `warmup_epochs` or `last_epoch` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindflow import get_warmup_cosine_annealing_lr
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
    check_lr_param_type_value(lr_init, "lr_init", float, thresh_hold=0.0, restrict=True)
    check_lr_param_type_value(warmup_lr_init, "warmup_lr_init", float, thresh_hold=0.0, restrict=False)
    check_lr_param_type_value(eta_min, "eta_min", float, thresh_hold=0.0, restrict=False)
    check_lr_param_type_value(warmup_epochs, "warmup_epochs", int, thresh_hold=0, restrict=False)
    check_lr_param_type_value(steps_per_epoch, "steps_per_epoch", int, thresh_hold=0, restrict=True)
    check_lr_param_type_value(last_epoch, "last_epoch", int, thresh_hold=0, restrict=True)

    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_lr_list = []
    if warmup_epochs != 0:
        warmup_lr_list += _get_linear_warmup_lr(warmup_steps, lr_init, warmup_lr_init)

    cosine_lr_list = _get_cosine_annealing_lr(lr_init, steps_per_epoch, last_epoch, eta_min=eta_min)

    lr_each_step = warmup_lr_list + cosine_lr_list[warmup_steps:]

    return np.array(lr_each_step).astype(np.float32)

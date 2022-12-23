# Copyright 2022 Huawei Technologies Co., Ltd
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
# pylint: disable=W,E,R
# ============================================================================
"""utils"""
import stat
import time
import os
import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import Callback


class CosineSimilarity(nn.Cell):
    """cosine similarity"""

    def __init__(self, axis: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.axis = axis
        self.eps = eps

    def construct(self, x1, x2):
        """construct"""
        if x2.shape[0] == 1:
            x = ops.tensor_dot(x1, x2, (self.axis, self.axis)).squeeze(1)
            y1 = ops.tensor_dot(x1, x1, (self.axis, self.axis)).diagonal()
            y2 = ops.tensor_dot(x2, x2, (self.axis, self.axis)).squeeze(1)
        else:
            x = ops.tensor_dot(x1, x2, (self.axis, self.axis)).diagonal()
            y1 = ops.tensor_dot(x1, x1, (self.axis, self.axis)).diagonal()
            y2 = ops.tensor_dot(x2, x2, (self.axis, self.axis)).diagonal()

        y = ops.sqrt(y1) * ops.sqrt(y2)

        return x / ops.maximum(y, self.eps)


def mv(mat, vec):
    vec = ops.expand_dims(vec, 1)
    return ops.matmul(mat, vec).squeeze(1)


def squeeze(tensor):
    tensor = tensor.squeeze()
    if tensor.ndim == 0:
        return tensor.expand_dims(0)
    return tensor


# pylint: disable=invalid-name
def xavier_normal_(parameter: ms.Parameter, gain: float = 1.):
    fan_in, fan_out = ms.common.initializer._calculate_fan_in_and_fan_out(parameter.shape)
    std = gain * (2 / (fan_in + fan_out)) ** 0.5
    weight = ms.Tensor(np.random.normal(loc=0.0, scale=std, size=parameter.shape), ms.float32)
    parameter.set_data(weight)


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def warmup_cosine_lr(base_lr, steps_per_epoch, total_epochs, warmup=0.1, init_ratio=0.05):
    """warmup consine lr"""
    total_steps = steps_per_epoch * total_epochs
    if warmup < 1:
        warmup_steps = warmup * total_steps
    else:
        warmup_steps = warmup
    decay_steps = total_steps - warmup_steps
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * init_ratio))
        else:
            lr.append(a_cosine_learning_rate(i, base_lr, warmup_steps, decay_steps))

    return lr


class LossCallBack(Callback):
    """loss call back"""

    def __init__(self, run, bsz, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.run = run
        self.bsz = bsz

    def on_train_step_end(self, run_context):
        """on train step end"""
        cb_params = run_context.original_args()

        loss = np.array(cb_params.net_outputs)

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num

        if self._per_print_times != 0 and cur_num % self._per_print_times == 0:
            fd = os.open("./{}_loss_{}.record".format(self.run, self.bsz),
                         os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR)
            fo = os.fdopen(fd, "a+")
            time_now = time.strftime("%d %H:%M:%S", time.localtime())
            fo.write(f"epoch: {cb_params.cur_epoch_num} step: {cur_step_in_epoch} , loss is {loss},"
                     f" time is {time_now}\n")
            print(f"epoch: {cb_params.cur_epoch_num} step: {cur_step_in_epoch} , loss is {loss},"
                  f" time is {time_now}\n")
            os.close(fd)


def for_stack(tensor_list, axis, step=100):
    """for stack"""
    if len(tensor_list) < step:
        return ops.stack(tensor_list, axis)

    stacked_list = []
    total = len(tensor_list) // step
    for i in range(total):
        start = i * step
        end = start + step
        stacked_list.append(ops.stack(tensor_list[start:end], axis))
    if len(tensor_list) % step != 0:
        stacked_list.append(ops.stack(tensor_list[total * step:], axis))

    if len(stacked_list) < 100:
        stacked = ops.concat(stacked_list, axis)
    else:
        stacked = for_concat(stacked_list, axis)

    return stacked


def for_concat(tensor_list, axis, step=50):
    """for concat"""
    if len(tensor_list) < step:
        return ops.concat(tensor_list, axis)

    concated_list = []
    total = len(tensor_list) // step
    for i in range(total):
        start = i * step
        end = start + step
        concated_list.append(ops.concat(tensor_list[start:end], axis))
    if len(tensor_list) % step != 0:
        concated_list.append(ops.concat(tensor_list[total * step:], axis))

    return ops.concat(concated_list, axis)

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
# ============================================================================
"""util"""
import os
import time
import struct
import sys

import numpy as np

import mindspore
import mindspore.ops as ops
from mindspore.train.callback import Callback
from mindspore.common.tensor import Tensor


class StepLossTimeMonitor(Callback):
    """Callback for monitoring training progress, loss, and performance metrics

    This callback tracks and logs:
    - Training loss at each step
    - Current learning rate
    - Processing speed (FPS)
    - Epoch-level metrics

    Attributes:
        _per_print_times (int): Frequency of logging (every n steps)
        batch_size (int): Training batch size
        rank (int): Process rank in distributed training
        step_time (float): Timestamp for step timing
        epoch_start (float): Timestamp for epoch timing
        losses (list): List to store loss values per epoch
        learning_rate_func (function): Function to compute learning rate
        lr_all (np.array): Array containing all learning rate values
    """
    def __init__(
            self, lr_all, learning_rate_func, batch_size, per_print_times=1, rank=0
    ):
        super().__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.batch_size = batch_size
        self.rank = rank
        self.step_time = 0
        self.epoch_start = 0
        self.losses = []
        self.learning_rate_func = learning_rate_func
        self.lr_all = lr_all

    def step_begin(self, run_context):
        """step_begin"""
        self.step_time = time.time()
        self.run_context = run_context

    def step_end(self, run_context):
        """step_end"""
        step_seconds = time.time() - self.step_time
        step_fps = self.batch_size * 1.0 / step_seconds

        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        arr_lr = cb_params.optimizer.learning_rate.asnumpy()
        new_lr = self.learning_rate_func(self.lr_all, cb_params.cur_step_num)
        ops.assign(cb_params.optimizer.learning_rate, Tensor(new_lr, mindspore.float32))

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(
                    loss[0].asnumpy(), np.ndarray
            ):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "epoch: {} step: {}. Invalid loss, terminating training.".format(
                    cb_params.cur_epoch_num, cur_step_in_epoch
                )
            )
        self.losses.append(loss)
        if self._per_print_times != 0:
            print(
                "step: %s, loss is %s, fps is %s, lr is %s"
                % (cur_step_in_epoch, loss, step_fps, arr_lr),
                flush=True,
            )

    def epoch_begin(self, run_context):
        self.epoch_start = time.time()
        self.losses = []
        self.run_context = run_context

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_cost = time.time() - self.epoch_start
        step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        step_fps = self.batch_size * 1.0 * step_in_epoch / epoch_cost
        if self.rank == 0:
            print(
                "epoch: {:3d}, avg loss:{:.4f}, total cost: {:.3f} s, per step fps:{:5.3f}".format(
                    cb_params.cur_epoch_num, np.mean(self.losses), epoch_cost, step_fps
                ),
                flush=True,
            )


def filter_checkpoint_parameter_by_list(param_dict, filter_list):
    """remove useless parameters according to filter_list"""
    for key in list(param_dict.keys()):
        for name in filter_list:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del param_dict[key]
                break


def make_grid(inputs):
    """get 2D grid"""
    batch_size, _, height, width = inputs.shape
    xx = np.arange(0, width).reshape(1, -1)
    xx = np.tile(xx, (height, 1))
    yy = np.arange(0, height).reshape(-1, 1)
    yy = np.tile(yy, (1, width))
    xx = xx.reshape(1, 1, height, width)
    xx = np.tile(xx, (batch_size, 1, 1, 1))
    yy = yy.reshape(1, 1, height, width)
    yy = np.tile(yy, (batch_size, 1, 1, 1))
    grid = np.concatenate((xx, yy), axis=1).astype(np.float32)
    return grid


def warp(inputs, flow, grid, mode="bilinear", padding_mode="zeros"):
    width = inputs.shape[-1]
    vgrid = grid + flow
    vgrid = 2.0 * vgrid / max(width - 1, 1) - 1.0
    vgrid = vgrid.transpose(0, 2, 3, 1)
    output = ops.grid_sample(
        inputs, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True
    )
    return output


def readbin(filename, size, precision="real*4", skip=0, endianness="ieee-be"):
    """write a ndarray into binary file for MITgcm."""
    if endianness == "ieee-be":
        df_part1 = ">"
    elif endianness == "ieee-le":
        df_part1 = "<"
    else:
        print("Error endianness!")
        sys.exit(1)

    if precision == "real*4":
        df_part2 = "f"
        length = 4
    elif precision == "real*8":
        df_part2 = "d"
        length = 8
    else:
        print("Error precision!")
        sys.exit(1)

    dataformat = df_part1 + str(np.prod(size)) + df_part2
    fout = open(filename, "rb")
    if skip != 0:
        fout.seek(np.prod(size) * length * skip)
    data = struct.unpack(dataformat, fout.read(length * np.prod(size)))
    fout.close()
    return np.reshape(data, size, order="F")


def writebin(filename, ndarray, precision="real*4", skip=0, endianness="ieee-be"):
    """write a ndarray into binary file for MITgcm."""
    size = np.prod(ndarray.shape)
    arraycol = np.reshape(ndarray, (size, 1), order="F")
    if endianness == "ieee-be":
        df_part1 = ">"
    elif endianness == "ieee-le":
        df_part1 = "<"
    else:
        print("Error endianness!")
        sys.exit(1)

    if precision == "real*4":
        df_part2 = "f"
        length = 4
    elif precision == "real*8":
        df_part2 = "d"
        length = 8
    else:
        print("Error precision!")
        sys.exit(1)

    dataformat = df_part1 + str(size) + df_part2

    if os.path.isfile(filename):
        fout = open(filename, "r+b")
    else:
        fout = open(filename, "wb")

    fout.seek(size * skip * length, 0)
    fout.write(struct.pack(dataformat, *arraycol))
    fout.close()

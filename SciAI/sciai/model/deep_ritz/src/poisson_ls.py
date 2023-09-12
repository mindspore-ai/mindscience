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

"""Module for poisson ls problem"""
import math
import os

import numpy as np
import mindspore as ms
from mindspore import nn, ops
from sciai.common import TrainCellWithCallBack
from sciai.operators import grad
from sciai.utils import print_log

from .network import RitzNet, Problem
from .process import sample_from_surface, sample_from_disk


def error_fun(output, target):
    error = output - target
    error = ops.sqrt(ops.mean(error * error))
    # Calculate the L2 norm error.
    ref = ops.sqrt(ops.mean(target * target))
    return error / ref


def exact(r, data):
    expand_dim = ops.ExpandDims()
    output = r ** 2 - ops.reduce_sum(data * data, 1)
    return expand_dim(output, 1)


def rough(r, data):
    output = r ** 2 - r * ops.reduce_sum(data * data, axis=1) ** 0.5
    return output.unsqueeze(1)


def ffun(data):
    return 4.0 * ops.ones((data.shape[0], 1), dtype=ms.float32)


class LsLossNumericGrad(nn.Cell):
    """Ls loss numeric differentiation"""
    def __init__(self, args, backbone):
        super().__init__()
        self._backbone = backbone
        self.diff = args.diff
        self.penalty = args.penalty
        self.radius = args.radius

    def construct(self, *data):
        """Network forward pass"""
        data1, data2, data1_x_shift, data1_y_shift, data1_x_nshift, data1_y_nshift = data
        output1 = self._backbone(data1)
        output1_x_shift = self._backbone(data1_x_shift)
        output1_y_shift = self._backbone(data1_y_shift)
        output1_x_nshift = self._backbone(data1_x_nshift)
        output1_y_nshift = self._backbone(data1_y_nshift)

        dfdx2 = (output1_x_shift + output1_x_nshift - 2 * output1) / (self.diff ** 2)
        # Use difference to approximate derivatives.
        dfdy2 = (output1_y_shift + output1_y_nshift - 2 * output1) / (self.diff ** 2)

        # Loss function 1
        f_term = ffun(data1)
        loss1 = ops.mean((dfdx2 + dfdy2 + f_term) * (dfdx2 + dfdy2 + f_term)) * math.pi * self.radius ** 2
        target = exact(self.radius, data1)
        error = error_fun(output1, target)

        # Loss function 2
        output2 = self._backbone(data2)
        target2 = exact(self.radius, data2)
        loss2 = ops.mean(
            (output2 - target2) * (output2 - target2) * self.penalty * 2 * math.pi * self.radius)
        loss = loss1 + loss2

        return loss, error


class LsLossAutoGrad(nn.Cell):
    """Ls loss auto differentiation"""
    def __init__(self, args, backbone):
        super().__init__()
        self._backbone = backbone
        self._grad = grad(self._backbone, output_index=0, input_index=0)
        self._second_grad = grad(self._grad, output_index=0, input_index=0)
        self.penalty = args.penalty
        self.radius = args.radius

    def construct(self, *data):
        """Network forward pass"""
        data1, data2, _, _, _, _ = data
        output1 = self._backbone(data1)

        ddf = self._second_grad(data1)
        ddf_sum = ddf.sum(axis=1, keepdims=True)

        # Loss function 1
        f_term = ffun(data1)
        loss1 = ops.mean((ddf_sum + f_term) * (ddf_sum + f_term)) * math.pi * self.radius ** 2
        target = exact(self.radius, data1)
        error = error_fun(output1, target)

        # Loss function 2
        output2 = self._backbone(data2)
        target2 = exact(self.radius, data2)
        loss2 = ops.mean(
            (output2 - target2) * (output2 - target2) * self.penalty * 2 * math.pi * self.radius)
        loss = loss1 + loss2

        return loss, error


class PoissonLs(Problem):
    """Class defining the poisson ls problem"""
    def __init__(self, args):
        super().__init__(args=args)

    def init_net(self):
        net = RitzNet(self.args.layers)
        exponential_decay_lr = nn.ExponentialDecayLR(self.args.lr, self.args.gamma, self.args.step_size, is_stair=True)
        optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=exponential_decay_lr,
                                  weight_decay=self.args.decay)
        if self.args.autograd:
            print_log("Loss is calculated by auto-grad")
            loss_cell = LsLossAutoGrad(self.args, net)
        else:
            print_log("Loss is calculated by numerical grad")
            loss_cell = LsLossNumericGrad(self.args, net)
        ckpt_interval = 1000 if self.args.save_ckpt else 0
        train_net = TrainCellWithCallBack(loss_cell, optimizer, time_interval=self.args.write_step,
                                          ckpt_interval=ckpt_interval, ckpt_dir=self.args.save_ckpt_path,
                                          loss_interval=self.args.write_step, loss_names=("loss", "error"),
                                          amp_level=self.args.amp_level)
        return train_net, net

    def train(self, net, dtype):
        x_shift = ms.Tensor(np.array([self.args.diff, 0.0]), dtype)
        y_shift = ms.Tensor(np.array([0.0, self.args.diff]), dtype)
        data1 = ms.Tensor(sample_from_disk(self.args.radius, self.args.body_batch), dtype)
        data2 = ms.Tensor(sample_from_surface(self.args.radius, self.args.bdry_batch), dtype)

        data1_x_nshift = data1 - x_shift
        data1_y_nshift = data1 - y_shift
        data1_x_shift = data1 + x_shift
        data1_y_shift = data1 + y_shift

        if not os.path.exists(self.args.save_data_path):
            os.makedirs(self.args.save_data_path)

        for step in range(self.args.train_epoch - self.args.train_epoch_pre):
            _, error = net(data1, data2, data1_x_shift, data1_y_shift, data1_x_nshift, data1_y_nshift)

            if self.args.save_data and step % self.args.write_step == self.args.write_step - 1:
                with open(f"{self.args.save_data_path}/loss_data.txt", mode="a+") as file:
                    file.write(str(step + self.args.train_epoch_pre + 1) + " " + str(error) + "\n")

            if step % self.args.sample_step == self.args.sample_step - 1:
                data1 = ms.Tensor(sample_from_disk(self.args.radius, self.args.body_batch), dtype)
                data2 = ms.Tensor(sample_from_surface(self.args.radius, self.args.bdry_batch), dtype)

                data1_x_shift = data1 + x_shift
                data1_y_shift = data1 + y_shift
                data1_x_nshift = data1 - x_shift
                data1_y_nshift = data1 - y_shift
            if 10 * (step + 1) % self.args.train_epoch == 0:
                print_log("%s%% finished..." % (100 * (step + 1) // self.args.train_epoch))

    def evaluate(self, net, dtype):
        num_quad = self.args.num_quad
        data = ms.Tensor(sample_from_disk(1, num_quad), dtype)
        output = net(data)
        target = exact(self.args.radius, data)
        return error_fun(output, target)

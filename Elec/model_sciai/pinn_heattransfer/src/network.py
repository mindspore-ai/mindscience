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
"""Network architectures for PINN heat-transfer"""

import mindspore as ms
from mindspore import nn, ops, amp
from mindspore.common.initializer import XavierNormal
from mindspore.nn import optim
from sciai.architecture import MLP, Normalize
from sciai.common import TrainCellWithCallBack, lbfgs_train
from sciai.utils import amp2datatype, to_tensor


class NeuralNetwork:
    """Neural net"""

    def __init__(self, args, logger, x_f, ub, lb):
        super().__init__()
        # DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [T]
        layers = args.layers

        # Setting up the optimizers with the hyperparameters
        self.epochs = args.epochs
        self.nt_max_iter = args.nt_epochs
        self.print_interval = args.print_interval
        self.save_ckpt_path = args.save_ckpt_path
        self.ckpt_interval = 50 if args.save_ckpt else 0
        self.amp_level = args.amp_level
        self.model_name = args.model_name
        self.dtype = amp2datatype(args.amp_level)
        self.use_lbfgs = args.lbfgs
        self.x_f, self.t_f = to_tensor((x_f[:, 0:1], x_f[:, 1:2]), dtype=self.dtype)

        lb_tensor, ub_tensor = to_tensor((lb, ub), dtype=self.dtype)
        self.model = Net(layers, lb_tensor, ub_tensor)

        if self.dtype == ms.float16:
            self.model.to_float(self.dtype)
        self.ms_optimizer = optim.Adam(self.model.trainable_params(), learning_rate=args.lr, beta1=args.b1)
        self.logger = logger

    def ms_optimization(self, x_train, u_train):
        """ms_optimization"""
        loss_cell = MyWithLossCell(self.model, nn.MSELoss(), self.x_f, self.t_f, u_train)
        self.logger.log_train_opt("Adam")
        train_cell = TrainCellWithCallBack(loss_cell, self.ms_optimizer, loss_interval=self.print_interval,
                                           time_interval=self.print_interval,
                                           ckpt_interval=self.ckpt_interval,
                                           ckpt_dir=self.save_ckpt_path,
                                           amp_level=self.amp_level,
                                           model_name=self.model_name)
        for _ in range(self.epochs):
            train_cell(x_train)

    def nt_optimization(self, x_train, u_train):
        self.logger.log_train_opt("LBFGS")
        loss_cell = MyWithLossCell(self.model, nn.MSELoss(), self.x_f, self.t_f, u_train)
        amp.auto_mixed_precision(loss_cell, amp_level=self.amp_level)
        lbfgs_train(loss_cell, (x_train,), self.nt_max_iter)

    def train(self, x_train, u_train):
        """train"""
        self.logger.log_train_start(self)

        # Creating the tensors
        x_train, u_train = to_tensor((x_train, u_train), dtype=self.dtype)

        # Optimizing
        self.ms_optimization(x_train, u_train)
        if self.use_lbfgs and ms.get_context("mode") == ms.PYNATIVE_MODE:
            self.nt_optimization(x_train, u_train)

        self.logger.log_train_end(self.epochs + self.nt_max_iter)

    def predict(self, x_star):
        u_star = self.model(to_tensor(x_star, dtype=self.dtype))
        return u_star.asnumpy()


class Net(nn.Cell):
    """Net"""

    def __init__(self, layers, lb, ub):
        super().__init__()
        self.normalize = Normalize(lb, ub)
        self.sequential = MLP(layers, weight_init=XavierNormal(), bias_init="zeros", activation="tanh")

    def construct(self, x):
        """Network forward pass"""
        x = self.normalize(x)
        x = self.sequential(x)
        return x


class MyWithLossCell(nn.Cell):
    """Loss net"""

    def __init__(self, net, loss_fn, x_f, t_f, u):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.x_f = x_f
        self.t_f = t_f
        self.u = u
        self.grad = ops.grad(self.net)
        self.grad_x = NetGradX(self.net)
        self.grad_second = ops.grad(self.grad_x)

    def construct(self, x):
        """Network forward pass"""
        output = self.net(x)
        loss1 = self.loss_fn(output, self.u)
        x_f = ops.stack([self.x_f[:, 0], self.t_f[:, 0]], axis=1)
        grad_one = self.grad(x_f)
        _, t_t = ops.split(grad_one, axis=1, split_size_or_sections=1)
        grad_second = self.grad_second(x_f)
        t_xx, _ = ops.split(grad_second, axis=1, split_size_or_sections=1)
        loss2 = self.loss_fn(t_xx, t_t)
        return loss1 + loss2


class NetGradX(nn.Cell):
    """Gradient net w.r.t variable x"""

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.grad = ops.GradOperation()

    def construct(self, x):
        """Network forward pass"""
        grad_one = self.grad(self.network)(x)
        t_x, _ = ops.split(grad_one, axis=1, split_size_or_sections=1)
        return t_x

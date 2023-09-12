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

"""Network architectures for DGM"""
import os

import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, ops
import numpy as np

from sciai.architecture import MSE
from sciai.common import TrainCellWithCallBack
from sciai.utils import print_log


class MyNetWithLoss(nn.Cell):
    """Loss net"""

    def __init__(self, backbone):
        super().__init__()  # backbone is the network
        self.backbone = backbone
        self.grad = ops.GradOperation()
        self.grad_net = self.grad(self.backbone)
        self.reduce_mean = ops.ReduceMean()
        self.mse = MSE()
        self.x_ic = ops.ones((1, 1), ms.float32)

    def construct(self, x, x_initial):
        loss_domain = self.mse(self.grad_net(x) - self.grad_x(x))
        loss_ic = self.mse(self.backbone(x_initial) - self.x_ic)
        return loss_domain, loss_ic

    def grad_x(self, x):
        return 2 * np.pi * ops.cos(2 * np.pi * x) * ops.cos(4 * np.pi * x) - 4 * np.pi * ops.sin(
            4 * np.pi * x) * ops.sin(2 * np.pi * x)


class Train:
    """Class for model training"""

    def __init__(self, net, heat_equation, args, dtype, debug=False):
        self.history_mean_hooks = []

        self.history_tl = []
        self.history_dl = []
        self.history_il = []

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.ckpt_interval = 200 if args.save_ckpt else 0
        self.print_interval = args.print_interval
        self.model_name = args.model_name
        self.net = net
        self.model = heat_equation
        self.dtype = dtype

        self.debug = debug
        self.uniform_real = ops.UniformReal()

        self.save_fig = args.save_fig
        self.save_anim = args.save_anim
        self.figures_path = args.figures_path
        self.save_ckpt_path = args.save_ckpt_path
        self.amp_level = args.amp_level
        if self.debug:
            self.hooks = {}
            self.get_all_layers(self.net)

        if self.save_fig:
            folder = os.path.exists(f"{self.figures_path}/frames")
            if not folder:
                os.makedirs(f"{self.figures_path}/frames")

    def sample(self, xs=0, xe=1, size=2 ** 8):
        """sample uniformly"""
        x = xs + self.uniform_real((size, 1)) * xe
        x_initial = ms.Tensor(np.array([[0]]), dtype=self.dtype)
        return x, x_initial

    def train(self):
        """train model"""
        optimizer = nn.Adam(self.net.trainable_params(), learning_rate=self.lr)
        loss_net = MyNetWithLoss(self.net)
        train_cell = TrainCellWithCallBack(loss_net, optimizer,
                                           time_interval=self.print_interval, loss_interval=self.print_interval,
                                           ckpt_interval=self.ckpt_interval, ckpt_dir=self.save_ckpt_path,
                                           loss_names=("loss_domain", "loss_ic"), amp_level=self.amp_level,
                                           model_name=self.model_name)

        for e in range(self.epochs):
            x, x_boundry_0 = self.sample(size=self.batch_size)
            train_cell(x, x_boundry_0)

            if self.save_fig and e % 20 == 19:
                self.snapshot(e)

                x, x_boundry_0 = self.sample(size=2 ** 10)
                loss_domain, loss_ic = loss_net(x, x_boundry_0)
                loss_total = loss_domain + loss_ic

                self.history_tl.append(float(loss_total))
                self.history_dl.append(float(loss_domain))
                self.history_il.append(float(loss_ic))

                if self.debug:
                    mean = [float(ops.mean(self.hooks.get(l))) for l in self.hooks]
                    self.history_mean_hooks.append(mean)

        if self.save_anim and self.save_fig:
            cmd = f"convert -delay 10 -loop 0 $(ls {self.figures_path}/frames/*.png | sort -V) ./figures/animation.gif"
            ret = os.system(cmd)
            if ret != 0:
                print_log("Failed to animate prediction. If needed, please install ImageMagick according to README.md.")

    def snapshot(self, epoch):
        """net prediction and loss snapshot. Plot net prediction, loss, exact solution"""
        plt.ioff()
        max_x = 1
        x_range = ms.Tensor(np.linspace(0, max_x, 100, dtype=np.float), dtype=self.dtype).reshape(-1, 1)
        y = self.net(x_range)
        x_error, _ = self.model.criterion(x_range, ops.zeros((1, 1), self.dtype))
        x_error = x_error / 100
        fig, ax = plt.subplots()
        ax.set_ylim([-0.5, 2.5])
        ax.plot(x_range, y, label='Neural Net')
        ax.plot(x_range, x_error, label='Loss')
        ax.plot(x_range, self.model.exact_solution(x_range), '--', color='lightgray', label='Exact')
        ax.legend(fontsize=8, loc=2)
        plt.savefig(f"{self.figures_path}/frames/{epoch}.png")
        plt.close(fig)

    def hook_fn(self, m, _, o):
        self.hooks[m] = o

    def get_all_layers(self, net):
        for _, layer in net.cells_and_names():
            if isinstance(layer, nn.Dense):
                # Register a hook for all dense
                layer.register_forward_hook(self.hook_fn)

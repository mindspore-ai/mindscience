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

"""ENSO network"""
import mindspore.dataset as ds
from mindspore import nn, ops, Callback
from mindspore.train.callback._callback import _handle_loss
from sciai.common.dataset import DatasetGenerator
from sciai.common import TrainCellWithCallBack


class ENSO(nn.Cell):
    """ENSO network"""
    def __init__(self):
        super().__init__()
        self.tanh = ops.Tanh()
        self.conv2d1 = nn.Conv2d(6, 50, (4, 8), pad_mode="same", weight_init="XavierUniform", has_bias=True)
        self.max_pool_2d1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv2d(50, 50, (4, 8), pad_mode="same", weight_init="XavierUniform", has_bias=True)
        self.max_pool_2d2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv2d(50, 50, (4, 8), pad_mode="same", weight_init="XavierUniform", has_bias=True)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(5400, 50, activation="tanh")
        self.dense2 = nn.Dense(50, 17, activation="tanh")

    def construct(self, x):
        """Network forward pass"""
        x = self.conv2d1(x)
        x = self.tanh(x)
        x = self.max_pool_2d1(x)
        x = self.conv2d2(x)
        x = self.tanh(x)
        x = self.max_pool_2d2(x)
        x = self.conv2d3(x)
        x = self.tanh(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class ConvBlock(nn.Cell):
    """Convolution Block"""
    def __init__(self, in_channel):
        super().__init__()
        self.tanh = ops.Tanh()
        self.conv2d1 = nn.Conv2d(in_channel, 30, (4, 8), pad_mode="same", weight_init="XavierUniform", has_bias=True)
        self.batch_norm1 = nn.BatchNorm2d(num_features=30)
        self.conv2d2 = nn.Conv2d(30, 30, (4, 8), pad_mode="same", weight_init="XavierUniform", has_bias=True)
        self.batch_norm2 = nn.BatchNorm2d(num_features=30)
        self.conv2d_skip = nn.Conv2d(in_channel, 30, (1, 1), pad_mode="same", weight_init="XavierUniform",
                                     has_bias=True)
        self.add = ops.Add()
        self.max_pool_2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        """Network forward pass"""
        x_skip = x
        x = self.conv2d1(x)
        x = self.batch_norm1(x)
        x = self.tanh(x)
        x = self.conv2d2(x)
        x = self.batch_norm2(x)
        x = self.tanh(x)
        x_skip = self.conv2d_skip(x_skip)
        x = self.add(x, x_skip)
        x = self.tanh(x)
        x = self.max_pool_2d(x)
        return x


class ENSOResNet(nn.Cell):
    """ENSO ResNet"""
    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvBlock(6)
        self.conv_block2 = ConvBlock(30)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(3240, 30, activation="tanh")
        self.dense2 = nn.Dense(30, 17, activation="tanh")

    def construct(self, x):
        """Network forward pass"""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LossRecord(Callback):
    """to record loss"""
    def __init__(self):
        self._per_print_times = 1
        self._last_print_time = 0
        self.train_loss_record = []
        self.val_loss_record = []

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        if loss:
            self.train_loss_record.append(loss)

    def on_eval_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        if loss:
            self.val_loss_record.append(loss.asnumpy())


def after_train(*inputs):
    """process after train"""
    args, net, obs_ip_train, obs_nino34_train, ip_var, nino34_var = inputs

    loss_func = nn.MSELoss()

    optim = nn.optim.SGD(params=net.trainable_params(), learning_rate=args.lr_after)
    obs_dataset = ds.GeneratorDataset(source=DatasetGenerator(obs_ip_train, obs_nino34_train),
                                      shuffle=True, column_names=["data", "label"])
    obs_dataset = obs_dataset.batch(batch_size=args.batch_size_after, drop_remainder=True)
    var_dataset = ds.GeneratorDataset(source=DatasetGenerator(ip_var, nino34_var),
                                      column_names=["data", "label"])
    var_dataset = var_dataset.batch(batch_size=len(ip_var))
    loss_cell = nn.WithLossCell(net, loss_func)
    train_cell = TrainCellWithCallBack(loss_cell, optimizer=optim, loss_interval=args.print_interval,
                                       time_interval=args.print_interval, amp_level=args.amp_level)
    for _ in range(args.epochs_after):
        for x, y in obs_dataset:
            _ = train_cell(x, y)
        for x, y in var_dataset:
            _ = loss_cell(x, y)

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
"""nsf nets train"""
import os

import mindspore as ms
from mindspore import nn

from eval import evaluate
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.dataset import generate_data
from src.network import VPNSFnet
from src.process import prepare


def train(n_iter, learning_rate, net, args, *data):
    """train"""
    optim = nn.optim.Adam(net.trainable_params(), learning_rate)
    ckpt_interval = 1000 if args.save_ckpt else 0
    train_cell = TrainCellWithCallBack(net, optim,
                                       time_interval=args.print_interval, loss_interval=args.print_interval,
                                       ckpt_interval=ckpt_interval, ckpt_dir=args.save_ckpt_path,
                                       amp_level=args.amp_level, model_name=args.model_name)
    for _ in range(n_iter):
        train_cell(*data)


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    lam, ub_train, vb_train, x_train, xb_train, y_train, yb_train = generate_data(args, dtype)
    net = VPNSFnet(xb_train, yb_train, ub_train, vb_train, x_train, y_train, args.layers)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net)
    for single_lr, epoch_num in zip(args.lr, args.epochs):
        train(epoch_num, single_lr, net, args, xb_train, yb_train, ub_train, vb_train,
              x_train, y_train)
    evaluate(lam, net)
    if args.save_ckpt:
        ms.save_checkpoint(net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

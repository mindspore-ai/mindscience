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
"""DeepBSDE train script"""
import os

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.nn.dynamic_lr import piecewise_constant_lr

from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import print_time, amp2datatype, to_tensor, print_log, calc_ckpt_name

from src.net import DeepBSDE, WithLossCell
from src.config import prepare
from src.equation import get_bsde
from src.eval_utils import apply_eval


def train(args, bsde, net, net_loss):
    """Model Training"""
    dtype = amp2datatype(args.amp_level)

    args.lr_boundaries.append(args.num_iterations)
    lr = Tensor(piecewise_constant_lr(args.lr_boundaries, args.lr_values), dtype=dtype)
    opt = nn.Adam(net.trainable_params(), lr)
    train_cell = TrainCellWithCallBack(net_loss, opt,
                                       loss_interval=args.print_interval,
                                       time_interval=args.print_interval,
                                       amp_level=args.amp_level)

    for i in range(args.epochs):
        dw, x = bsde[i]
        dw, x = to_tensor((dw, x), dtype)
        train_cell(dw, x)

        if i % args.print_interval == 0:
            eval_param = {"model": net_loss, "valid_data": bsde.sample(args.valid_size)}
            eval_loss, y_init = apply_eval(eval_param, dtype)
            print_log(f"eval loss: {eval_loss}, Y0: {y_init}")

    if args.save_ckpt:
        ms.save_checkpoint(net_loss.net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))


@print_time("train")
def main(args):
    bsde = get_bsde(args)
    net = DeepBSDE(args, bsde)
    net_loss = WithLossCell(net)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net=net)
    train(args, bsde, net, net_loss)


if __name__ == '__main__':
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

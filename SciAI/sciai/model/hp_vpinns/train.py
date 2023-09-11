
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

"""hp vpinns train"""
import os

import numpy as np
import mindspore as ms
from mindspore import nn

from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import print_log, datatype2np, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import VPINN
from src.plot import plot_fig
from src.process import get_data, prepare


def train(args, model, u_train, x_u_train):
    """train"""
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=args.lr)
    train_cell = TrainCellWithCallBack(model, optimizer, time_interval=10, loss_interval=10, ckpt_interval=1000,
                                       loss_names=("loss", "lossb", "lossv"), grad_first=True,
                                       amp_level=args.amp_level)
    total_record = []
    for epoch in range(args.epochs):
        loss, _, _ = train_cell(x_u_train, u_train)
        if epoch % 10 == 0:
            total_record.append(np.array([epoch, loss.asnumpy()]))
            if loss < args.early_stop_loss:
                print_log(f"early stop since loss {loss} is less than threshold {args.early_stop_loss}")
                break
    return total_record


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    np_dtype = datatype2np(dtype)
    f_ext_total, w_quad_train, x_quad_train, x_test, x_u_train, grid, u_test, u_train \
        = get_data(args, dtype, np_dtype)

    # Model and Training
    net = VPINN(x_quad_train, w_quad_train, f_ext_total, grid, args, dtype)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net)
    total_record = train(args, net, u_train, x_u_train)
    if args.save_ckpt:
        ms.save_checkpoint(net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    u_pred = net.net_u(x_test)
    if args.save_fig:
        plt_elems = grid, u_pred, u_test, x_test
        plot_fig(x_quad_train, x_u_train, args, total_record, plt_elems)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

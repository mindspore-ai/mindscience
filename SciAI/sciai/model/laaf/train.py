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

"""laaf train"""
import os
import time

import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import Normal

from sciai.architecture.basic_block import MLPAAF
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import RecoverySlopeLoss
from src.plot import plot_train
from src.process import get_data, save_mse_hist, prepare


def train(loss_cell, args, x, y):
    """train"""
    optim = nn.Adam(loss_cell.trainable_params(), args.lr)
    train_cell = TrainCellWithCallBack(loss_cell, optim, loss_interval=10, time_interval=10, grad_first=True,
                                       amp_level=args.amp_level, model_name="laaf")
    a_hist, mse_hist, sol = [], [], []
    start_time, last_time = time.time(), time.time()
    for n in range(args.epochs):
        loss, y_pred = train_cell(x, y)
        err = loss.asnumpy()
        a_hist.append(loss_cell.a_values_np().asnumpy())
        mse_hist.append(err)
        if n in args.sol_epochs:
            sol.append(y_pred.asnumpy())
            this_time = time.time()
            print_log('steps: %d, loss: %.3e, interval: %.3e, total: %.3e' % (
                n, err, this_time - last_time, this_time - start_time))
            last_time = this_time
    return a_hist, mse_hist, sol


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    x, y = get_data(args.num_grid, dtype)
    normal = Normal(sigma=0.1, mean=0.0)
    net = MLPAAF(args.layers, weight_init=normal, bias_init=normal, activation=ops.tanh,
                 a_value=0.1, scale=10, share_type="layer_wise")
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net)
    loss_cell = RecoverySlopeLoss(net)
    _, mse_hist, sol = train(loss_cell, args, x, y)
    if args.save_ckpt:
        ms.save_checkpoint(net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    if args.save_data:
        save_mse_hist(args.save_data_path, mse_hist)
    if args.save_fig:
        plot_train(args.figures_path, sol, x, y)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

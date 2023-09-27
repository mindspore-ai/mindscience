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

"""deeponet train"""
import os
from collections import defaultdict

import numpy as np
import mindspore as ms
from mindspore import ops, nn

from eval import evaluation
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import print_log, to_tensor, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import DeepONet, SampleNet
from src.plot import save_loss_fig, plot_prediction
from src.process import save_loss_data, load_data, prepare


def train(feed_train, feed_test, feed_test0, loss_net, args, dtype):
    """train"""
    _, _, y_test = feed_test  # retain this to avoid RelErr nan
    _, _, y_test0 = feed_test0  # retain this to avoid RelErr nan
    x_u_train_ms, x_y_train_ms, y_train_ms = to_tensor(feed_train, dtype)
    x_u_test_ms, x_y_test_ms, y_test_ms = to_tensor(feed_test, ms.float32)
    x_u_test0_ms, x_y_test0_ms, y_test0_ms = to_tensor(feed_test0, ms.float32)
    n_train = x_u_train_ms.shape[0]
    batch_num = n_train // args.batch_size
    optim = nn.optim.Adam(loss_net.trainable_params(), learning_rate=args.lr)
    train_cell = TrainCellWithCallBack(loss_net, optim, grad_first=True, time_interval=args.print_interval,
                                       loss_interval=args.print_interval, batch_num=batch_num,
                                       model_name=args.model_name)
    records = defaultdict(list)
    loss_train = 1e16
    ind_range = ops.arange(n_train)
    os.makedirs(args.figures_path, exist_ok=True)
    os.makedirs(args.save_ckpt_path, exist_ok=True)
    os.makedirs(args.save_data_path, exist_ok=True)

    for i in range(args.epochs):
        ind_shuffled = ops.shuffle(ind_range)
        for j in range(batch_num):
            indexes_interval = ops.cast(ind_shuffled[j * args.batch_size: (j + 1) * args.batch_size], ms.int32)
            loss, _ = train_cell(x_u_train_ms, x_y_train_ms, y_train_ms, indexes_interval)
            del indexes_interval
        if i % args.print_interval == 0 and loss < loss_train:
            ms.save_checkpoint(loss_net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
            loss_train = loss
            loss_net.to_float(ms.float32)
            loss_test, y_pred = loss_net.net(x_u_test_ms, x_y_test_ms, y_test_ms)
            loss_test0, y_pred0 = loss_net.net(x_u_test0_ms, x_y_test0_ms, y_test0_ms)
            error = np.linalg.norm(y_pred.asnumpy() - y_test) / np.linalg.norm(y_test)
            error0 = np.linalg.norm(y_pred0.asnumpy() - y_test0) / np.linalg.norm(y_test0)
            loss_net.to_float(dtype)

            print_log(f"epoch: {i + 1}, training_loss: {loss_train}, test_loss: {loss_test}, test_loss0: "
                      f"{loss_test0}, rel_err: {error},  rel_err0: {error0}\n\n")

            ii, losst, lossv, lossv0 = update_records(i, loss_train, loss_test, loss_test0, records)
            if args.save_fig:
                save_loss_fig(args, ii, losst, lossv, lossv0)
            if args.save_data:
                save_loss_data(args.save_data_path, ii, losst, lossv, lossv0)


def update_records(i, loss_train, loss_test, loss_test0, records):
    """update records"""
    records["i_h"].append(np.float64(i))
    records["loss_train_h"].append(loss_train.asnumpy())
    records["loss_test_h"].append(loss_test.asnumpy())
    records["loss_test0_h"].append(loss_test0.asnumpy())
    ii = np.stack(records["i_h"])
    losst = np.stack(records["loss_train_h"])
    lossv = np.stack(records["loss_test_h"])
    lossv0 = np.stack(records["loss_test0_h"])
    return ii, losst, lossv, lossv0


@print_time("train")
def main(args):
    """main"""
    dtype = amp2datatype(args.amp_level)
    feed_train, feed_test, feed_test0 = load_data(args.load_data_path)
    net = DeepONet(args.layers_u, args.layers_y)
    loss_net = SampleNet(net)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, loss_net)
    if dtype == ms.float16:
        loss_net.to_float(dtype)
    train(feed_train, feed_test, feed_test0, loss_net, args, dtype)
    y_pred, y_test = evaluation(feed_test, feed_test0, loss_net)
    if args.save_fig:
        plot_prediction(y_pred, y_test, args)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

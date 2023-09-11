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

"""hfm train"""
import time

import mindspore as ms
from mindspore.nn import Adam
import numpy as np

from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import amp2datatype, to_tensor
from sciai.utils.python_utils import print_time
from src.network import MyWithLossCell
from src.process import simple_evaluate, prepare_network, prepare_training_data, obtain_data, prepare


def train(args, model, dtype, *data):
    """train"""
    t_data, x_data, y_data, c_data, t_eqns, x_eqns, y_eqns = data

    loss_cell = MyWithLossCell(model)

    ckpt_interval = 2000 if args.save_ckpt else 0
    optimizer = Adam(model.trainable_params(), learning_rate=args.lr)
    train_net = TrainCellWithCallBack(loss_cell, optimizer,
                                      time_interval=args.print_interval, loss_interval=args.print_interval,
                                      ckpt_interval=ckpt_interval, ckpt_dir=args.save_ckpt_path,
                                      amp_level=args.amp_level, model_name=args.model_name)

    x_data_num = t_data.shape[0]
    x_eqns_num = t_eqns.shape[0]
    start_time = time.time()
    running_time = 0
    it = 0
    while running_time < args.total_time and it < args.epochs:

        idx_data = np.random.choice(x_data_num, min(args.batch_size, x_data_num))
        idx_eqns = np.random.choice(x_eqns_num, args.batch_size)

        t_data_batch, x_data_batch, y_data_batch, c_data_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch = \
            to_tensor((t_data[idx_data, :], x_data[idx_data, :], y_data[idx_data, :], c_data[idx_data, :],
                       t_eqns[idx_eqns, :], x_eqns[idx_eqns, :], y_eqns[idx_eqns, :]), dtype=dtype)

        train_net(t_data_batch, x_data_batch, y_data_batch, c_data_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch)

        del t_data_batch, x_data_batch, y_data_batch, c_data_batch, t_eqns_batch, x_eqns_batch, y_eqns_batch

        if it % 10 == 0:
            elapsed = time.time() - start_time
            running_time += elapsed / 3600
            start_time = time.time()
        it += 1


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    c_star, p_star, t_star, u_star, v_star, x_star, y_star = obtain_data(args)

    t_data, x_data, y_data, c_data, t_eqns, x_eqns, y_eqns = prepare_training_data(args, c_star, t_star, x_star, y_star)
    model = prepare_network(args, dtype, t_data, x_data, y_data)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net=model)
    train(args, model, dtype, t_data, x_data, y_data, c_data, t_eqns, x_eqns, y_eqns)

    simple_evaluate(args, model, c_star, p_star, u_star, v_star, t_star, x_star, y_star)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

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
"""phygeonet train"""
import os
import time

import numpy as np
import mindspore as ms
from mindspore import nn
from sklearn.metrics import mean_squared_error as cal_mse

from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import Net, USCNN
from src.plot import plot_train, plot_train_process
from src.process import get_data, prepare
from src.py_mesh import to4_d_tensor


def train(args, dataset, net, ofv_sb):
    """train"""
    m_res_hist = []
    ev_hist = []
    total_start_time = time.time()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr)
    train_cell = TrainCellWithCallBack(net, optimizer, amp_level=args.amp_level, time_interval=1, loss_interval=1,
                                       batch_num=len(dataset), grad_first=True, model_name=args.model_name)
    for epoch in range(args.epochs):
        mres, ev = train_one_epoch(epoch, train_cell, args, dataset, ofv_sb)
        print_log(f"epoch: {epoch}, m_res_loss: {mres}, e_v_loss:{ev}")
        m_res_hist.append(mres)
        ev_hist.append(ev)
        if ev < 0.1:
            break
    time_spent = time.time() - total_start_time
    return ev_hist, m_res_hist, time_spent


def train_one_epoch(epoch, train_cell, args, dataset, ofv_sb):
    """train one epoch"""
    m_res, e_v = ms.Tensor(0), ms.Tensor(0)
    batch_num = len(dataset)
    cnnv_numpy, coord, output_v = None, None, None
    training_data_loader = dataset.create_dict_iterator()
    for batch in training_data_loader:
        _, coord, _, _, _, jinv, dxdxi, dydxi, dxdeta, dydeta = to4_d_tensor(batch.values())
        loss, output_v = train_cell(coord, jinv, dxdxi, dydxi, dxdeta, dydeta)
        m_res += loss
        cnnv_numpy = output_v[0, 0, :, :]
        e_v += np.sqrt(cal_mse(ofv_sb, cnnv_numpy) / cal_mse(ofv_sb, ofv_sb * 0))
    if epoch % 5000 == 0 or epoch % args.epochs == 0 or \
            np.sqrt(cal_mse(ofv_sb, cnnv_numpy) / cal_mse(ofv_sb, ofv_sb * 0)) < 0.1:
        if args.save_ckpt:
            ms.save_checkpoint(train_cell.train_cell.network, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
        if args.save_fig:
            plot_train_process(args, coord, epoch, ofv_sb, output_v)
    return (m_res / batch_num).asnumpy(), (e_v / batch_num).asnumpy()


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    dataset, h, nvar_input, nvar_output, nx, ny, ofv_sb = get_data(args)
    model = USCNN(h, nx, ny, nvar_input, nvar_output)
    net = Net(model, args.batch_size, h)
    net.to_float(dtype)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net)

    ev_hist, m_res_hist, time_spent = train(args, dataset, net, ofv_sb)
    if args.save_fig:
        plot_train(args, ev_hist, m_res_hist, time_spent)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

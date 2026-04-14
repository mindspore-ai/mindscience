
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

"""label free dnn surrogate train"""
import mindspore as ms
from mindspore import nn
import numpy as np
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import amp2datatype
from sciai.utils.python_utils import print_time

from src.network import WithLossCell, UVPNet
from src.process import get_data, prepare


def save_checkpoint(uvp_net, args, sigma, epochs):
    pre_fix = f"{args.model_name}_{args.amp_level}"
    ms.save_checkpoint(uvp_net.net_u, f"{args.save_ckpt_path}/{pre_fix}_sigma{sigma}_epoch{epochs}hard_u.ckpt")
    ms.save_checkpoint(uvp_net.net_v, f"{args.save_ckpt_path}/{pre_fix}_sigma{sigma}_epoch{epochs}hard_v.ckpt")
    ms.save_checkpoint(uvp_net.net_p, f"{args.save_ckpt_path}/{pre_fix}_sigma{sigma}_epoch{epochs}hard_P.ckpt")


def load_ckpt(uvp_net, args):
    ms.load_checkpoint(args.load_ckpt_path[0], uvp_net.net_u)
    ms.load_checkpoint(args.load_ckpt_path[1], uvp_net.net_v)
    ms.load_checkpoint(args.load_ckpt_path[2], uvp_net.net_p)


def train(args, dataset_iter, rho, uvp_net, sigma):
    """train"""
    loss_cell = WithLossCell(uvp_net, args.nu, rho)
    optimizer = nn.optim.Adam(params=loss_cell.trainable_params(), learning_rate=args.lr, beta1=0.9, beta2=0.99,
                              eps=1e-15)
    train_cell = TrainCellWithCallBack(loss_cell, optimizer, time_interval=args.print_interval,
                                       loss_interval=args.print_interval, batch_num=len([_ for _ in dataset_iter]),
                                       amp_level=args.amp_level)
    loss_history = []
    for epoch in range(args.epochs_train):
        for batch_idx, data in enumerate(dataset_iter):
            x_in, y_in, scale_in = data["data"], data["label"], data["scale"]
            loss = train_cell(x_in, y_in, scale_in)
            if batch_idx % args.print_interval == 0:
                loss_history.append(loss)
            if epoch % 5 == 0 and args.save_ckpt:
                save_checkpoint(uvp_net, args, sigma, epoch)
    return loss_history


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    d_p, dataset_iter, l, mu, r_inlet, rho, sigma, x_end, x_start = get_data(args)
    uvp_net = UVPNet(args.layers, sigma, mu, r_inlet, x_start, d_p, l, x_end)
    if dtype == ms.float16:
        uvp_net.to_float(ms.float16)
    if args.load_ckpt:
        load_ckpt(uvp_net, args)
    loss_history = train(args, dataset_iter, rho, uvp_net, sigma)
    if args.save_ckpt:
        save_checkpoint(uvp_net, args, sigma, args.epochs_train)
    np.savetxt('Loss_track_pipe_para.csv', np.array(loss_history))


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

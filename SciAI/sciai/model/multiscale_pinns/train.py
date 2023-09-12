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
"""mpinns train"""
import os
from collections import defaultdict

import mindspore as ms
from mindspore import nn
import numpy as np
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import to_tensor, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time

from src.network import Heat1D, NetNN, NetFF, NetSTFF
from src.plot import plot_train_val
from src.process import get_data, prepare
from eval import evaluate


def train(args, samplers, model, dtype, *data):
    """train"""
    x_star, u_star = data
    learning_rate = nn.ExponentialDecayLR(learning_rate=args.lr, decay_rate=0.9, decay_steps=1000, is_stair=True)
    optim = nn.Adam(model.trainable_params(), learning_rate)
    train_cell = TrainCellWithCallBack(model, optim,
                                       time_interval=args.print_interval,
                                       loss_interval=args.print_interval,
                                       loss_names=("bcs_loss", "ics_loss", "res_loss"),
                                       amp_level=args.amp_level)
    log_dict = defaultdict(list)
    for it in range(args.epochs):
        # Fetch boundary mini-batches
        x_ics_batch, u_ics_batch = model.fetch_minibatch(samplers["ics_sampler"], args.batch_size)
        x_bc1_batch, _ = model.fetch_minibatch(samplers["bcs_sampler1"], args.batch_size)
        x_bc2_batch, _ = model.fetch_minibatch(samplers["bcs_sampler2"], args.batch_size)

        # Fetch residual mini-batch
        x_res_batch, _ = model.fetch_minibatch(samplers["res_sampler"], args.batch_size)

        t_ics, x_ics = x_ics_batch[:, :1], x_ics_batch[:, 1:2]
        u_ics = u_ics_batch
        t_bc1, x_bc1 = x_bc1_batch[:, :1], x_bc1_batch[:, 1:2]
        t_bc2, x_bc2 = x_bc2_batch[:, :1], x_bc2_batch[:, 1:2]
        t_r, x_r = x_res_batch[:, :1], x_res_batch[:, 1:2]

        params = t_ics, x_ics, u_ics, t_bc1, x_bc1, t_bc2, x_bc2, t_r, x_r
        params = to_tensor(params, dtype)
        loss_bcs, loss_ics, loss_res = train_cell(*params)
        if it % args.print_interval == 0:
            u_pred = model.predict_u(x_star, dtype)
            error = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
            log_dict["loss_bcs_log"].append(loss_bcs.asnumpy())
            log_dict["loss_ics_log"].append(loss_ics.asnumpy())
            log_dict["loss_res_log"].append(loss_res.asnumpy())
            log_dict["l2_error_log"].append(error)
    return log_dict


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    x_star, f_star, u_star, samplers, k, sigma, t, x = get_data(args)

    net_u = {"net_nn": NetNN,  # NetNN: Plain MLP
             "net_ff": NetFF,  # NetFF: Plain Fourier feature network
             "net_st_ff": NetSTFF  # NetSTFF: Spatial-temporal Plain Fourier feature network
             }.get(args.net_type)(args.layers, sigma)
    model = Heat1D(k, samplers.get("res_sampler"), net_u)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)
    log_dict = train(args, samplers, model, dtype, x_star, u_star)
    if args.save_ckpt:
        ms.save_checkpoint(model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    u_pred = evaluate(dtype, model, u_star, x_star)
    if args.save_fig:
        plot_train_val(args.figures_path, log_dict, u_pred, x_star, f_star, u_star, t, x)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

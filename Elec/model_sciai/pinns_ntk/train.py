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
"""pinns ntk train"""
import os

import numpy as np
import mindspore as ms
from mindspore import nn

from eval import evaluate
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import PINN
from src.plot import plot_figures, plot_loss
from src.process import generate_data, prepare


def train(model, log_ntk, log_weights, args):
    """train"""
    exponential_decay_lr = nn.ExponentialDecayLR(args.lr, decay_rate=0.9, decay_steps=1000, is_stair=False)
    optimizer = nn.SGD(model.trainable_params(), learning_rate=exponential_decay_lr)
    ckpt_interval = 10000 if args.save_ckpt else 0
    train_cell = TrainCellWithCallBack(model, optimizer, amp_level=args.amp_level, time_interval=args.print_interval,
                                       loss_interval=args.print_interval, loss_names=("loss_bcs", "loss_res"),
                                       ckpt_interval=ckpt_interval, ckpt_dir=args.save_ckpt_path,
                                       model_name=args.model_name)
    for epoch in range(args.epochs):
        loss_bcs_value, loss_res_value = train_cell(model.x_u, model.y_u, model.x_r, model.y_r)
        if epoch % 100 == 0:
            model.loss_bcs_log.append(loss_bcs_value.value().asnumpy())
            model.loss_res_log.append(loss_res_value.value().asnumpy())

            # provide x, x' for NTK
            if log_ntk:
                print_log("Compute NTK...")
                k_uu, k_ur, k_rr = model.ntks(model.x_u, model.x_r)
                model.k_uu_log.append(k_uu.value().asnumpy())
                model.k_ur_log.append(k_ur.value().asnumpy())
                model.k_rr_log.append(k_rr.value().asnumpy())

            if log_weights:
                print_log("Weights stored...")
                model.weights_log.append([np.array(weight.value().asnumpy()) for weight in model.net_u.weights()])
                model.biases_log.append([bias.value().asnumpy() for bias in model.net_u.biases()])
    if args.save_ckpt:
        ms.save_checkpoint(model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    a = 4
    dom_coords = np.array([[0.0], [1.0]])
    # Training data on u(x) -- Dirichlet boundary conditions
    x_r, x_u, y_r, y_u = generate_data(a, dom_coords, args.num, dtype)
    model = PINN(args.layers, x_u, y_u, x_r, y_r, dtype)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)
    train(model, True, True, args)
    if args.save_fig:
        plot_loss(args, model)
    x_star, u_pred, u_star = evaluate(a, dom_coords, model, dtype)
    if args.save_fig:
        plot_figures(x_star, model, u_pred, u_star, args.figures_path)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

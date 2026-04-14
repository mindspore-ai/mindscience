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
"""xpinns train"""
import os
from collections import defaultdict

import numpy as np
import mindspore as ms
from mindspore import nn
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time

from src.network import XPINN
from src.plot import plot, PlotElements
from src.process import generate_data, prepare
from eval import evaluate


def train(model, inputs, stars, exacts, args):
    """train"""
    x_star1, x_star2, x_star3 = stars
    u_exact2, u_exact3 = exacts
    history = defaultdict(list)

    optimizer = nn.Adam(model.trainable_params(), args.lr)
    train_net = TrainCellWithCallBack(model, optimizer,
                                      loss_interval=args.print_interval,
                                      time_interval=args.print_interval,
                                      ckpt_interval=args.ckpt_interval,
                                      loss_names=("loss1", "loss2", "loss3"),
                                      amp_level=args.amp_level,
                                      model_name=args.model_name)
    for epoch in range(args.epochs):
        loss1, loss2, loss3 = train_net(*inputs)
        if epoch % args.print_interval == 0:
            _, u_pred2, u_pred3 = model.predict(x_star1, x_star2, x_star3)
            # Relative L2 error in subdomains 2 and 3
            l2_error2 = np.linalg.norm(u_exact2 - u_pred2, 2) / np.linalg.norm(u_exact2, 2)
            l2_error3 = np.linalg.norm(u_exact3 - u_pred3, 2) / np.linalg.norm(u_exact3, 2)
            history["mse_history1"].append(loss1)
            history["mse_history2"].append(loss2)
            history["mse_history3"].append(loss3)
            history["l2_err2"].append(l2_error2)
            history["l2_err3"].append(l2_error3)
    return history


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    layers1, layers2, layers3 = args.layers1, args.layers2, args.layers3
    x_f1_train, x_f2_train, x_f3_train, x_fi1_train, x_fi2_train, x_star1, x_star2, x_star3, x_ub_train, model_inputs, \
        u_exact, u_exact2, u_exact3, xb, xi1, xi2, yb, yi1, yi2 = generate_data(args.load_data_path, dtype)
    model = XPINN(layers1, layers2, layers3)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)
    history = train(model, model_inputs, (x_star1, x_star2, x_star3), (u_exact2, u_exact3), args)
    if args.save_ckpt:
        ms.save_checkpoint(model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    u_pred = evaluate(model, u_exact, x_star1, x_star2, x_star3)

    if args.save_fig:
        elements = PlotElements(epochs=args.epochs,
                                mse_hists=(history["mse_history1"], history["mse_history2"], history["mse_history3"]),
                                x_f_trains=(x_f1_train, x_f2_train, x_f3_train),
                                x_fi_trains=(x_fi1_train, x_fi2_train),
                                x_ub_train=x_ub_train,
                                l2_errs=(history["l2_err2"], history["l2_err3"]),
                                u=(u_exact, u_pred),
                                x_stars=(x_star1, x_star2, x_star3),
                                x=(xb, xi1, xi2),
                                y=(yb, yi1, yi2),
                                figures_path=args.figures_path)
        plot(elements)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

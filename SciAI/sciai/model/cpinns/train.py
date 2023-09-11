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

"""cpinns train"""
from collections import defaultdict

import numpy as np
import mindspore as ms
from mindspore import nn

from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import amp2datatype, datatype2np
from sciai.utils.python_utils import print_time
from src.network import PINN
from src.plot import plot
from src.process import get_model_inputs, get_data, get_star_inputs, prepare


def train(model, stars, inputs, args):
    """train"""
    x_star1, x_star2, x_star3, x_star4, u1_star, u2_star, u3_star, u4_star = stars
    u_star = np.concatenate([u1_star, u2_star, u3_star, u4_star])
    exponential_decay_lr = nn.ExponentialDecayLR(learning_rate=args.lr, decay_rate=0.5, decay_steps=3000, is_stair=True)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=exponential_decay_lr)
    train_net = TrainCellWithCallBack(model, optimizer, loss_interval=args.print_interval,
                                      time_interval=args.print_interval, ckpt_interval=args.ckpt_interval,
                                      amp_level=args.amp_level, model_name=args.model_name)

    def a_value(net):
        return net.mlp.a_value().value().asnumpy()

    history = defaultdict(list)
    for it in range(args.epochs):
        loss1, loss2, loss3, loss4 = train_net(*inputs)
        if it % args.print_interval == 0:
            u1_pred, u2_pred, u3_pred, u4_pred = model.predict(x_star1, x_star2, x_star3, x_star4)
            u_pred = np.concatenate([u1_pred, u2_pred, u3_pred, u4_pred])
            error_u = np.linalg.norm(u_star.astype(np.float) - u_pred.astype(np.float), 2) \
                      / np.linalg.norm(u_star.astype(np.float), 2)

            history["l2error_u"].append(error_u)
            history["loss1"].append(loss1)
            history["loss2"].append(loss2)
            history["loss3"].append(loss3)
            history["loss4"].append(loss4)
            history["a1"].append(a_value(model.net1))
            history["a2"].append(a_value(model.net2))
            history["a3"].append(a_value(model.net3))
            history["a4"].append(a_value(model.net4))
    return history


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    np_dtype = datatype2np(dtype)
    nu = 0.01 / np.pi  # 0.0025
    nn_layers_total, t_mesh, x_mesh, x_star, u_star, x_interface, total_dict = get_data(args, np_dtype)
    model = PINN(nn_layers_total, nu, x_interface, dtype)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)
    model_inputs = get_model_inputs(total_dict, dtype)
    stars_cast = get_star_inputs(np_dtype, total_dict)
    history = train(model, stars_cast, model_inputs, args)
    if args.save_fig:
        plot(history, t_mesh, x_mesh, x_star, args, model, u_star, x_interface, total_dict)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

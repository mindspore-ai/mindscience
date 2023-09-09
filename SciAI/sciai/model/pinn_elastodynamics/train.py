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
"""pinn elastodynamics train"""
import os

import mindspore as ms
from mindspore import nn
from mindspore.nn import optim

from sciai.architecture.basic_block import FirstOutputCell
from sciai.common import TrainCellWithCallBack, lbfgs_train
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import DeepElasticWave
from src.process import generate_data, plot_res, prepare
from eval import load_checkpoint, evaluate


def train(args, model, input_data):
    """train"""
    exponential_decay_lr = nn.ExponentialDecayLR(learning_rate=args.lr, decay_rate=0.8, decay_steps=3000, is_stair=True)
    optimizer = optim.Adam(model.trainable_params(), learning_rate=exponential_decay_lr)
    loss_cell = FirstOutputCell(model)
    train_cell = TrainCellWithCallBack(loss_cell, optimizer, time_interval=args.print_interval,
                                       loss_interval=args.print_interval,
                                       ckpt_interval=args.ckpt_interval,
                                       amp_level=args.amp_level,
                                       model_name=args.model_name)
    for _ in range(args.epochs):
        train_cell(*input_data)
    print_log("adam result:")
    evaluate(input_data, model)

    if args.use_lbfgs:
        lbfgs_train(loss_cell, input_data, args.max_iter_lbfgs)


@print_time("train")
def main(args):
    """main"""
    dtype = amp2datatype(args.amp_level)

    # Network configuration, there is only Dirichlet boundary (u,v)
    max_t, n_t, input_data, srcs = generate_data(dtype)
    model = DeepElasticWave(args.uv_layers, dtype)
    if args.load_ckpt:
        model = load_checkpoint(args, dtype)
    train(args, model, input_data)
    if args.save_ckpt:
        ms.save_checkpoint(model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    evaluate(input_data, model)
    if args.save_fig:
        plot_res(max_t, n_t, model, args.figures_path, args.load_data_path, srcs, dtype)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

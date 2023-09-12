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

"""dgm train"""
import math
import os

import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.common.initializer import HeUniform

from sciai.architecture import MLP
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.advection import Advection
from src.network import Train
from src.process import prepare
from src.plot import plot_report, plot_activation_mean, visualize


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)
    net = MLP(args.layers, weight_init=HeUniform(negative_slope=math.sqrt(5)), bias_init="zeros", activation=ops.Tanh())
    advection = Advection(net)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net=net)
    debug_mode = ms.get_context("mode") == ms.PYNATIVE_MODE
    train = Train(net, advection, args, dtype, debug=debug_mode)
    train.train()
    if args.save_ckpt:
        print_log('save checkpoint......')
        ms.save_checkpoint(net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))

    x_max = 1
    x_range = ms.Tensor(np.linspace(0, x_max, 100, dtype=float), dtype=dtype).reshape(-1, 1)
    y = net(x_range)

    print_log(f"error: {ops.mean(ops.square(y - advection.exact_solution(x_range)))}")

    if args.save_fig:
        plot_report(train)
        plot_activation_mean(train)
        visualize(advection, args.figures_path, x_range, y)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

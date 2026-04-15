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
"""sympnets train"""
import os

import mindspore as ms

from sciai.context import init_project
from sciai.utils import amp2datatype, print_time, calc_ckpt_name

from src.brain import Brain
from src.nn.hnn import HNN
from src.nn.sympnet import LASympNet, GSympNet
from src.process import prepare


@print_time("train")
def main(args, problem):
    dtype = amp2datatype(args.amp_level)
    criterion, data = problem.init_data(args)
    net = get_net(args, data.dim)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, net)
    model = Brain(args, data, net, criterion)
    model.train()
    if args.save_data:
        model.save_txt()
    if args.save_ckpt:
        ms.save_checkpoint(model.best_model, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    if args.save_fig:
        model.best_model.to_float(dtype)
        problem.plot(data, model.best_model, args.figures_path)


def get_net(args, dim):
    if args.net_type == 'LA':
        return LASympNet(dim, args.la_layers, args.la_sublayers, args.activation)
    if args.net_type == 'G':
        return GSympNet(dim, args.g_layers, args.g_width, args.activation)
    if args.net_type == 'HNN':
        return HNN(dim, args.h_layers, args.h_width, args.h_activation)
    raise Exception("net type error")


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)

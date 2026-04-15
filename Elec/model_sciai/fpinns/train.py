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

"""fpinns train"""
import os

import mindspore as ms

from sciai.context import init_project
from sciai.utils import print_log, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.process import prepare


@print_time("train")
def main(args, problem):
    """main"""
    print_log('initializing net......')
    net = problem.setup_networks(args)
    train_cell = problem.setup_train_cell(args, net)

    if args.load_ckpt:
        print_log('load checkpoint......')
        ms.load_checkpoint(args.load_ckpt_path, net)

    print_log('start training......')
    problem.train(train_cell)

    if args.save_fig:
        problem.plot_train_process()

    if args.save_ckpt:
        print_log('save checkpoint......')
        ms.save_checkpoint(net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)

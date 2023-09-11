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

"""deep ritz train"""
import os

import mindspore as ms

from sciai.context import init_project
from sciai.utils import print_log, amp2datatype, calc_ckpt_name
from sciai.utils.python_utils import print_time
from src.network import count_parameters
from src.plot import write_result, visualize
from src.utils import prepare


@print_time("train")
def main(args, problem):
    dtype = amp2datatype(args.amp_level)

    train_net, ritz_net = problem.init_net()
    print_log("The number of parameters is %s," % count_parameters(ritz_net))

    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, ritz_net)

    problem.train(train_net, dtype=dtype)

    if args.save_ckpt:
        ms.save_checkpoint(ritz_net, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))

    test_error = problem.evaluate(ritz_net, dtype=dtype)
    print_log("The test error (of the last model) is %s." % test_error)

    if args.save_data:
        n_sample = 500
        write_result(args, ritz_net, n_sample, dtype=dtype)

    if args.save_fig:
        visualize(args)


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)

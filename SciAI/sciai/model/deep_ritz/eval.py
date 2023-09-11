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

"""deep ritz eval"""
import mindspore as ms
from mindspore import amp

from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time
from src.plot import write_result, visualize
from src.utils import prepare


@print_time("eval")
def main(args, problem):
    dtype = amp2datatype(args.amp_level)

    _, ritz_net = problem.init_net()
    ritz_net = amp.auto_mixed_precision(ritz_net, args.amp_level)

    ms.load_checkpoint(args.load_ckpt_path, ritz_net)

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

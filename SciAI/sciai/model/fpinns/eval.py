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

"""fpinns eval"""
import mindspore as ms
from mindspore import nn

from sciai.context import init_project
from sciai.utils import print_log, to_tensor, amp2datatype
from sciai.utils.python_utils import print_time
from src.process import prepare


@print_time("eval")
def main(args, problem):
    """main"""
    dtype = amp2datatype(args.amp_level)

    print_log('initializing net......')
    net = problem.setup_networks(args)
    if dtype == ms.float16:
        net.to_float(ms.float16)

    print_log('load checkpoint......')
    ms.load_checkpoint(f"{args.load_ckpt_path}", net)

    x_test, t_test, y_test = problem.generate_data(args.num_test)
    x_test, t_test, y_test = to_tensor((x_test, t_test, y_test), dtype=dtype)

    criterion = nn.MSELoss()
    y_res = problem.predict(net, x_test, t_test)
    test_mse = criterion(y_res, y_test)
    print_log("MSE:" + str(test_mse))

    if args.save_fig:
        problem.plot_result(x_test, t_test, y_test, y_res)


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)

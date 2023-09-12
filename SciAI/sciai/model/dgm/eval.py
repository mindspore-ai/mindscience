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

"""dgm eval"""
import math
import numpy as np
import mindspore as ms
from mindspore import ops, amp
from mindspore.common.initializer import HeUniform

from sciai.architecture import MLP
from sciai.context import init_project
from sciai.utils import amp2datatype, print_log
from sciai.utils.python_utils import print_time
from src.advection import Advection
from src.plot import visualize
from src.process import prepare


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)

    net = MLP(args.layers, weight_init=HeUniform(negative_slope=math.sqrt(5)), bias_init="zeros", activation=ops.Tanh())
    advection = Advection(net)
    net = amp.auto_mixed_precision(net, args.amp_level)
    ms.load_checkpoint(args.load_ckpt_path, net=net)

    x_max = 1
    x_range = ms.Tensor(np.linspace(0, x_max, 100, dtype=float), dtype=dtype).reshape(-1, 1)
    y = net(x_range)

    print_log(f"error: {ops.mean(ops.square(y - advection.exact_solution(x_range)))}")

    if args.save_fig:
        visualize(advection, args.figures_path, x_range, y)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

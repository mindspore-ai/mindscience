"""dgm eval"""
import math
import numpy as np
import mindspore as ms
from mindspore import ops, amp
from mindspore.common.initializer import HeUniform

from sciai.architecture import MLP
from sciai.context import init_project
from sciai.utils import data_type_dict_amp, print_log
from sciai.utils.python_utils import print_time
from src.advection import Advection
from src.plot import visualize
from src.process import prepare


@print_time("eval")
def main(args):
    dtype = data_type_dict_amp.get(args.amp_level, ms.float32)

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

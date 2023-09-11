"""sympnets eval"""
import mindspore as ms

from sciai.context import init_project
from sciai.utils import amp2datatype, print_time

from src.brain import Brain
from src.process import prepare
from train import get_net


@print_time("eval")
def main(args, problem):
    dtype = amp2datatype(args.amp_level)
    criterion, data = problem.init_data(args)
    net = get_net(args, data.dim)
    net.to_float(dtype)
    ms.load_checkpoint(args.load_ckpt_path, net)
    model = Brain(args, data, net, criterion)
    model.evaluate()
    if args.save_fig:
        problem.plot(data, model.net, args.figures_path)


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)

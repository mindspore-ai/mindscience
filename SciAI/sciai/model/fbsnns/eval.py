"""fbsnns eval"""
import mindspore as ms

from sciai.context import init_project
from sciai.utils import data_type_dict_amp
from sciai.utils.python_utils import print_time
from src.process import prepare


@print_time("eval")
def main(args, problem):
    problem.build_param()
    data_type = data_type_dict_amp.get(args.amp_level, ms.float16)
    problem.net.to_float(data_type)
    ms.load_checkpoint(args.load_ckpt_path, problem.net)
    problem.test()


if __name__ == "__main__":
    args_problem = prepare()
    init_project(args=args_problem[0])
    main(*args_problem)

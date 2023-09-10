"""pinn elastodynamics eval"""

import mindspore as ms

from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time
from src.network import DeepElasticWave
from src.process import generate_data, plot_res, prepare


def evaluate(input_data, model):
    """evaluate"""
    loss, loss_f_uv, loss_f_s, loss_src, loss_ic, loss_fix = model(*input_data)
    print_log('loss:', loss)
    print_log('loss_f_uv:', loss_f_uv)
    print_log('loss_f_s:', loss_f_s)
    print_log('loss_src:', loss_src)
    print_log('loss_ic:', loss_ic)
    print_log('loss_fix:', loss_fix)


@print_time("eval")
def main(args):
    """main"""
    dtype = amp2datatype(args.amp_level)
    max_t, n_t, input_data, srcs = generate_data(dtype)
    model = DeepElasticWave(args.uv_layers, dtype)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    model = load_checkpoint(args, dtype)

    evaluate(input_data, model)
    if args.save_fig:
        plot_res(max_t, n_t, model, args.figures_path, args.load_data_path, srcs, dtype)


def load_checkpoint(args, dtype):
    """load checkpoint"""
    if args.load_ckpt_path.endswith("pickle"):
        model = DeepElasticWave(args.uv_layers, dtype, uv_path=args.load_ckpt_path)
    elif args.load_ckpt_path.endswith("ckpt"):
        model = DeepElasticWave(args.uv_layers, dtype)
        ms.load_checkpoint(args.load_ckpt_path, model)
    else:
        raise TypeError("checkpoint file format error, please set 'load_ckpt_path' as file path ending with "
                        "'.pickle' or '.ckpt'")
    return model


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

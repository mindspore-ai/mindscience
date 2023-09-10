"""mpinns eval"""
import mindspore as ms
import numpy as np

from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time

from src.network import Heat1D, NetNN, NetFF, NetSTFF
from src.plot import plot_train_val
from src.process import get_data, prepare


def evaluate(dtype, model, u_star, x_star):
    """evaluate"""
    u_pred = model.predict_u(x_star, dtype)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print_log('Relative L2 error_u: {:.2e}'.format(error_u))
    return u_pred


@print_time("eval")
def main(args):
    dtype = amp2datatype(args.amp_level)
    x_star, f_star, u_star, samplers, k, sigma, t, x = get_data(args)

    net_u = {"net_nn": NetNN,  # NetNN: Plain MLP
             "net_ff": NetFF,  # NetFF: Plain Fourier feature network
             "net_st_ff": NetSTFF  # NetSTFF: Spatial-temporal Plain Fourier feature network
             }.get(args.net_type)(args.layers, sigma)
    model = Heat1D(k, samplers.get("res_sampler"), net_u)
    ms.load_checkpoint(args.load_ckpt_path, model)
    model.to_float(dtype)
    u_pred = evaluate(dtype, model, u_star, x_star)
    if args.save_fig:
        plot_train_val(args.figures_path, None, u_pred, x_star, f_star, u_star, t, x)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

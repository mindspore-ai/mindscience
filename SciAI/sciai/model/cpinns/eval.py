"""cpinns eval"""
import numpy as np
import mindspore as ms
from sciai.context import init_project
from sciai.utils import data_type_dict_amp, data_type_dict_np, print_log
from sciai.utils.python_utils import print_time

from src.network import PINN
from src.plot import plot
from src.process import get_data, get_star_inputs, prepare


def evaluate(model, np_dtype, total_dict):
    """evaluate"""
    x_star1, x_star2, x_star3, x_star4, u1_star, u2_star, u3_star, u4_star = get_star_inputs(np_dtype, total_dict)
    u_star = np.concatenate([u1_star, u2_star, u3_star, u4_star])
    u1_pred, u2_pred, u3_pred, u4_pred = model.predict(x_star1, x_star2, x_star3, x_star4)
    u_pred = np.concatenate([u1_pred, u2_pred, u3_pred, u4_pred])
    error_u = np.linalg.norm(u_star.astype(np.float) - u_pred.astype(np.float), 2) \
              / np.linalg.norm(u_star.astype(np.float), 2)
    print_log("error_u: ", error_u)


@print_time("eval")
def main(args):
    dtype = data_type_dict_amp.get(args.amp_level, ms.float32)
    np_dtype = data_type_dict_np.get(dtype)
    nu = 0.01 / np.pi  # 0.0025
    nn_layers_total, t_mesh, x_mesh, x_star, u_star, x_interface, total_dict = get_data(args, np_dtype)
    model = PINN(nn_layers_total, nu, x_interface, dtype)
    if dtype == ms.float16:
        model.to_float(ms.float16)
    ms.load_checkpoint(args.load_ckpt_path, model)
    evaluate(model, np_dtype, total_dict)
    if args.save_fig:
        plot(None, t_mesh, x_mesh, x_star, args, model, u_star, x_interface, total_dict)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

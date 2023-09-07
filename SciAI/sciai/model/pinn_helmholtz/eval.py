"""pinn helmholtz eval"""
import os

import mindspore as ms
from mindspore import Tensor, ops
import scipy.io

from sciai.context import init_project
from sciai.utils import print_log, data_type_dict_amp
from sciai.utils.python_utils import print_time

from src.network import PhysicsInformedNN
from src.plot import plot_result
from src.process import generate_data, prepare


def evaluate(data, dtype, model):
    """evaluate"""
    x_pred = Tensor(data.get("x_star").tolist(), dtype=dtype)
    z_pred = Tensor(data.get("z_star").tolist(), dtype=dtype)
    u_real = Tensor(data.get("U_real").tolist(), dtype=dtype)
    u_imag = Tensor(data.get("U_imag").tolist(), dtype=dtype)
    u_pred_real, u_pred_imag = model(x_pred, z_pred)
    error_u_real = ops.norm(u_real - u_pred_real) / ops.norm(u_real)
    error_u_imag = ops.norm(u_imag - u_pred_imag) / ops.norm(u_imag)
    print_log('Error u_real: %e, Error u_imag: %e' % (error_u_real, error_u_imag))
    return u_pred_imag, u_pred_real


@print_time("eval")
def main(args):
    dtype = data_type_dict_amp.get(args.amp_level, ms.float32)

    data = scipy.io.loadmat(f'{args.load_data_path}/Marmousi_3Hz_singlesource_ps.mat')
    _, _, bounds = generate_data(args, data, dtype)
    model = PhysicsInformedNN(args.layers, bounds)
    model.to_float(dtype)

    ms.load_checkpoint(args.load_ckpt_path, model)

    u_pred_imag, u_pred_real = evaluate(data, dtype, model)

    file_real = "u_real_pred_val.mat"
    file_imag = "u_imag_pred_val.mat"

    if args.save_results:
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        scipy.io.savemat(f'{args.results_path}/{file_real}', {'u_real_pred': u_pred_real.asnumpy()})
        scipy.io.savemat(f'{args.results_path}/{file_imag}', {'u_imag_pred': u_pred_imag.asnumpy()})
    if args.save_fig:
        plot_result(args, file_real=file_real, file_imag=file_imag)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

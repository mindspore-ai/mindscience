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
"""pinn helmholtz train"""
import os

import scipy.io
import mindspore as ms
from mindspore import nn, Tensor, ops
from sciai.common import TrainCellWithCallBack, lbfgs_train
from sciai.context import init_project
from sciai.utils import print_log, amp2datatype
from sciai.utils.python_utils import print_time

from src.network import PhysicsInformedNN, LossCellHelmholtz
from src.plot import plot_result, plot_losses
from src.process import generate_data, prepare


def train(args, model, train_param, train_tensor):
    """
    training process
    Args:
        args: arguments obtained from configuration
        model: neural networks
        train_param: training parameter
        train_tensor: training data

    Returns:

    """
    loss_cell = LossCellHelmholtz(model, train_param)
    x_tensor, z_tensor = train_tensor

    ckpt_interval = 2000 if args.save_ckpt else 0
    optimizer_adam = nn.Adam(model.trainable_params(), learning_rate=args.lr)
    train_cell = TrainCellWithCallBack(loss_cell, optimizer_adam, time_interval=args.print_interval,
                                       ckpt_interval=ckpt_interval, loss_interval=args.print_interval,
                                       ckpt_dir=args.save_ckpt_path, amp_level=args.amp_level,
                                       model_name=args.model_name)

    misfit = []
    for _ in range(args.epochs):
        loss = train_cell(x_tensor, z_tensor)
        misfit.append(loss.asnumpy())

    if args.save_ckpt:
        ms.save_checkpoint(model, f"{args.save_ckpt_path}/Optim_{args.model_name}_adam_{args.amp_level}.ckpt")
    if args.save_results:
        scipy.io.savemat(f'{args.results_path}/loss_adam_{args.epochs}_{args.amp_level}.mat', {'misfit': misfit})
    if args.lbfgs:
        lbfgs_train(loss_cell, (x_tensor, z_tensor), args.epochs_lbfgs)
    if args.save_ckpt:
        ms.save_checkpoint(model, f"{args.save_ckpt_path}/Optim_{args.model_name}_lbfgs_{args.amp_level}.ckpt")


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)

    data = scipy.io.loadmat(f'{args.load_data_path}/Marmousi_3Hz_singlesource_ps.mat')
    train_param, train_tensor, bounds = generate_data(args, data, dtype)
    model = PhysicsInformedNN(args.layers, bounds)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path, model)
    train(args, model, train_param, train_tensor)

    x_pred = Tensor(data.get("x_star").tolist(), dtype=dtype)
    z_pred = Tensor(data.get("z_star").tolist(), dtype=dtype)
    u_real = Tensor(data.get("U_real").tolist(), dtype=dtype)
    u_imag = Tensor(data.get("U_imag").tolist(), dtype=dtype)

    u_pred_real, u_pred_imag = model(x_pred, z_pred)

    error_u_real = ops.norm(u_real - u_pred_real) / ops.norm(u_real)
    error_u_imag = ops.norm(u_imag - u_pred_imag) / ops.norm(u_imag)

    print_log('Error u_real: %e, Error u_imag: %e' % (error_u_real, error_u_imag))

    file_real = "u_real_pred_adam.mat"
    file_imag = "u_imag_pred_adam.mat"
    if args.save_results:
        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        scipy.io.savemat(f'{args.results_path}/{file_real}', {'u_real_pred': u_pred_real.asnumpy()})
        scipy.io.savemat(f'{args.results_path}/{file_imag}', {'u_imag_pred': u_pred_imag.asnumpy()})

    if args.save_fig:
        plot_result(args, file_real=file_real, file_imag=file_imag)
        plot_losses(args)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)

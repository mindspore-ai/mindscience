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
"""
test
"""
import time
import argparse

import numpy as np
import scipy.io as sio
import mindspore
from mindspore import nn, context, Tensor, COOTensor

from src.models import MultiScaleGNN, MultiScaleGNNStructure
from src.datasets import read_test_data
from src.datasets import second_order_derivative_matix2d
from src.visualization import losses_curve, contourf_comparison

def run_test(model_test):
    """run_test"""
    loss_all = sio.loadmat(save_dir+'/loss_all.mat')['loss_all'].tolist()
    param_dict = mindspore.load_checkpoint(save_dir+'/checkpoint/model.ckpt')
    param_not_load, _ = mindspore.load_param_into_net(model_test, param_dict)
    print(param_not_load)
    model_test.set_train(False)

    losses_curve(np.array(loss_all), save_dir+'/Figures/')

    test_p_hat = []
    start_time = time.time()
    for it in range(div_u_star_test.shape[0]//5):
        if grid_type == 'unstructure':
            b = Tensor(div_u_star_test[it * 5 : (it + 1) * 5]).reshape([-1, 1])
            p = model_test(b)
            p = p.reshape([5, nx, ny]).asnumpy()
            test_p_hat.append(p)
        if grid_type == 'structure':
            b = Tensor(div_u_star_test[it * 5 : (it + 1) * 5]).unsqueeze(1)
            p = model_test(b)
            p = p.squeeze(1).asnumpy()
            test_p_hat.append(p)

        elapsed = time.time() - start_time
        print('it: %d, Time: %.3f' % (it, elapsed))
        start_time = time.time()

    test_p_hat = np.concatenate(test_p_hat, axis=0)

    print('####################################')
    mae = np.mean((test_p_hat - p_test)**2, axis=(1, 2))
    print('mean absolute error:')
    print(mae)
    print('average mean absolute error:')
    print(mae.mean())

    print('####################################')

    rl2e = np.linalg.norm(test_p_hat - p_test, axis=(1, 2))**2 / np.linalg.norm(p_test, axis=(1, 2))**2
    print('relative L2 error:')
    print(rl2e)
    print('average relative L2 error:')
    print(rl2e.mean())

    if plot_figure:
        for k in range(div_u_star_test.shape[0]):
            contourf_comparison(p_test[k], test_p_hat[k], save_dir+f'/Figures/{k}')

    return test_p_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_type', type=str, default='unstructure')
    parser.add_argument('--activ_fun', type=str, default='swish')
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('--lambda_p', type=int, default=20)
    parser.add_argument('--lambda_eq', type=int, default=1)
    parser.add_argument('--plot_figure', type=int, default=0)
    args = parser.parse_args()

    grid_type = args.grid_type
    activ_fun = args.activ_fun
    device = args.device
    lambda_p = args.lambda_p
    lambda_eq = args.lambda_eq
    plot_figure = args.plot_figure

    context.set_context(device_target=device)

    nx = 512
    ny = 512
    dx = 2.0 * np.pi / nx
    dy = 2.0 * np.pi / ny

    t_test = np.arange(0.2, 10.2, 0.2)

    div_u_star_test, p_test = read_test_data(t_test, dx, dy)
    print('divUstar_test shape :', div_u_star_test.shape)
    print('p_test shape :', p_test.shape)
    print('divUstar_test max :', np.abs(div_u_star_test).max())
    print('p_test max :', np.abs(p_test).max())

    a0 = second_order_derivative_matix2d(nx, ny, dx, dy)

    save_dir = f'./Savers/{grid_type}/{activ_fun}_lambda_p{lambda_p}lambda_eq{lambda_eq}'

    a0_tensor = COOTensor(Tensor(np.stack((a0.row, a0.col), axis=1)),
                          Tensor(a0.data, dtype=mindspore.float32), a0.shape)

    in_channels = 1
    out_channels = 1
    if activ_fun == 'swish':
        activation = nn.SiLU()
    if activ_fun == 'elu':
        activation = nn.ELU()
    if activ_fun == 'gelu':
        activation = nn.GELU()

    if grid_type == 'structure':
        model = MultiScaleGNNStructure(in_channels, out_channels, activation)
    if grid_type == 'unstructure':
        model = MultiScaleGNN(in_channels, out_channels, activation, a0)

    p_hat = run_test(model)

    mse = np.mean((p_hat - p_test)**2)
    print(mse)

    sio.savemat(save_dir + '/results.mat', {'p_hat': p_hat, 'p_test': p_test})

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
# pylint: disable=C0103
"""ICNet train"""
import time
import argparse
import yaml
import numpy as np

import mindspore as ms
from mindspore import set_seed, context, nn
from src.network_burgers import InvarianceConstrainedNN, InvarianceConstrainedNN_STRdige
from src.datasets_burgers import read_training_data, print_pde


# Set the seed.
np.random.seed(123456)
set_seed(123456)


def train(model, NIter, lr):
    """train"""
    # Get the gradients function
    params_train = model.dnn.trainable_params()
    params_train.append(model.lambda_u)
    params_train.append(model.lambda_v)
    params_train.append(model.lambda_uux)
    params_train.append(model.lambda_vuy)
    params_train.append(model.lambda_uvx)
    params_train.append(model.lambda_vvy)

    optimizer_Adam = nn.Adam(params_train, learning_rate=lr)

    grad_fn = ms.value_and_grad(model.loss_fn, None, optimizer_Adam.parameters, has_aux=True)

    model.dnn.set_train()

    start_time = time.time()

    for epoch in range(1, 1 + NIter):
        (loss, _, _, _, _, _, _), grads = grad_fn(model.x, model.y, model.t,
                                                  model.x_f, model.y_f, model.t_f, model.u, model.v)
        optimizer_Adam(grads)
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, lambda_uux: %.3f, lambda_vuy: %.3f, lambda_uxx: %.3f, lambda_uyy: %.3f,\
            lambda_uvx: %.3f, lambda_vvy: %.3f, lambda_vxx: %.3f, lambda_vyy: %.3f, Time: %.2f' %
                  (epoch, loss.item(),
                   model.lambda_uux.item(), model.lambda_vuy.item(),
                   model.lambda_u[2].item(), model.lambda_u[4].item(),
                   model.lambda_uvx.item(), model.lambda_vvy.item(), model.lambda_v[7].item(), model.lambda_v[9].item(), elapsed))

            initial_size = 11
            loss_history_Adam_Pretrain = np.empty([0])
            lambda_u_history_Adam_Pretrain = np.zeros((initial_size, 1))
            lambda_v_history_Adam_Pretrain = np.zeros((initial_size, 1))
            lambda_uux_history_Adam_Pretrain = np.zeros((1, 1))
            lambda_vuy_history_Adam_Pretrain = np.zeros((1, 1))
            lambda_uvx_history_Adam_Pretrain = np.zeros((1, 1))
            lambda_vvy_history_Adam_Pretrain = np.zeros((1, 1))

            loss_history_Adam_Pretrain = np.append(loss_history_Adam_Pretrain, loss.numpy())
            lambda_u_history_Adam_Pretrain = np.append(
                lambda_u_history_Adam_Pretrain, model.lambda_u.numpy(), axis=1)
            lambda_v_history_Adam_Pretrain = np.append(
                lambda_v_history_Adam_Pretrain, model.lambda_v.numpy(), axis=1)

            lambda_uux_new = np.array([model.lambda_uux.numpy()])
            lambda_vuy_new = np.array([model.lambda_vuy.numpy()])
            lambda_uvx_new = np.array([model.lambda_uvx.numpy()])
            lambda_vvy_new = np.array([model.lambda_vvy.numpy()])

            lambda_uux_history_Adam_Pretrain = np.append(
                lambda_uux_history_Adam_Pretrain, lambda_uux_new, axis=1)
            lambda_vuy_history_Adam_Pretrain = np.append(
                lambda_vuy_history_Adam_Pretrain, lambda_vuy_new, axis=1)
            lambda_uvx_history_Adam_Pretrain = np.append(
                lambda_uvx_history_Adam_Pretrain, lambda_uvx_new, axis=1)
            lambda_vvy_history_Adam_Pretrain = np.append(
                lambda_vvy_history_Adam_Pretrain, lambda_vvy_new, axis=1)

            start_time = time.time()
    np.save(f'Loss-Coe/{second_path}/loss_history_Adam_Pretrain', loss_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_u_history_Adam_Pretrain',
        lambda_u_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_v_history_Adam_Pretrain',
        lambda_v_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_uux_history_Adam_Pretrain',
        lambda_uux_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_vuy_history_Adam_Pretrain',
        lambda_vuy_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_uvx_history_Adam_Pretrain',
        lambda_uvx_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_vvy_history_Adam_Pretrain',
        lambda_vvy_history_Adam_Pretrain)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="./config/ICNet_Burgers.yaml")
    args = parser.parse_known_args()[0]

    with open(args.config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    params = config.get('params', {})
    for key, value in params.items():
        setattr(args, key, value)

    model_name = args.model_name
    case = args.case
    device = args.device
    device_id = args.device_id
    network_size = args.network_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    BatchNo = args.BatchNo
    load_params = args.load_params
    second_path = args.second_path
    description_burgers_u = args.description_burgers_u
    description_burgers_v = args.description_burgers_v
    lam = args.lam
    d_tol = args.d_tol

    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=device,
        device_id=device_id)

    use_npu = (device == 'Ascend')

    if use_npu:
        msfloat_type = ms.float16
    else:
        msfloat_type = ms.float32

    X_u_train, u_train, v_train, X_f_train = read_training_data(args)

    model_pretrain = InvarianceConstrainedNN(
        X_u_train,
        u_train,
        v_train,
        X_f_train,
        network_size,
        BatchNo,
        use_npu,
        msfloat_type)
    for epoch_train, lr_train in zip(epochs, learning_rate):
        train(model_pretrain, int(float(epoch_train)), lr_train)
    ms.save_checkpoint(model_pretrain.dnn, f'model/model.ckpt')

    lambda_uux_value = model_pretrain.lambda_uux.numpy()
    lambda_vuy_value = model_pretrain.lambda_vuy.numpy()
    lambda_uvx_value = model_pretrain.lambda_uvx.numpy()
    lambda_vvy_value = model_pretrain.lambda_vvy.numpy()
    lambda_u_value = model_pretrain.lambda_u.numpy()
    lambda_v_value = model_pretrain.lambda_v.numpy()

    # Save the last coefficients
    np.save(f'Loss-Coe/{second_path}/lambda_uux_value', lambda_uux_value)
    np.save(f'Loss-Coe/{second_path}/lambda_vuy_value', lambda_vuy_value)
    np.save(f'Loss-Coe/{second_path}/lambda_uvx_value', lambda_uvx_value)
    np.save(f'Loss-Coe/{second_path}/lambda_vvy_value', lambda_vvy_value)

    np.save(f'Loss-Coe/{second_path}/lambda_u_value', lambda_u_value)
    np.save(f'Loss-Coe/{second_path}/lambda_v_value', lambda_v_value)

    if load_params:
        lambda_u_value = np.load(f'Loss-Coe/{second_path}/lambda_u_value.npy')
        lambda_v_value = np.load(f'Loss-Coe/{second_path}/lambda_v_value.npy')
        lambda_uux_value = np.load(f'Loss-Coe/{second_path}/lambda_uux_value.npy')
        lambda_vuy_value = np.load(f'Loss-Coe/{second_path}/lambda_vuy_value.npy')
        lambda_uvx_value = np.load(f'Loss-Coe/{second_path}/lambda_uvx_value.npy')
        lambda_vvy_value = np.load(f'Loss-Coe/{second_path}/lambda_vvy_value.npy')
        model_ICCO = InvarianceConstrainedNN_STRdige(X_u_train, u_train, v_train, X_f_train, network_size, BatchNo,
                                                     lambda_u_value, lambda_v_value, lambda_uux_value, lambda_vuy_value, lambda_uvx_value, lambda_vvy_value,
                                                     load_params, second_path, msfloat_type,)
    else:
        model_ICCO = InvarianceConstrainedNN_STRdige(X_u_train, u_train, v_train, X_f_train, network_size, BatchNo,
                                                     lambda_u_value, lambda_v_value, lambda_uux_value, lambda_vuy_value, lambda_uvx_value, lambda_vvy_value,
                                                     load_params, second_path, msfloat_type,)

    lam = 10**-5
    d_tol = 5
    lambda_u_STRidge, lambda_v_STRidge = model_ICCO.call_trainstridge(lam, d_tol)

    print_pde(
        lambda_uux_value,
        lambda_vuy_value,
        lambda_u_STRidge,
        description_burgers_u,
        ut='u_t')

    print_pde(
        lambda_uvx_value,
        lambda_vvy_value,
        lambda_v_STRidge,
        description_burgers_v,
        ut='v_t')

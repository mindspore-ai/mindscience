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
from ChenChao.MindSpore.ICNet.ICNet.ICNet_PR.PR_003.src.network_skg import InvarianceConstrainedNN, InvarianceConstrainedNN_STRdige
from ChenChao.MindSpore.ICNet.ICNet.ICNet_PR.PR_003.src.datasets_skg import read_training_data, print_pde


# Set seed
np.random.seed(123456)
set_seed(123456)


def train(model, NIter, lr):
    """train"""
    # Get the gradients function
    params_train = model.dnn.trainable_params()
    params_train.append(model.lambda_u)
    params_train.append(model.lambda_uxx)
    params_train.append(model.lambda_uyy)

    optimizer_Adam = nn.Adam(params_train, learning_rate=lr)

    grad_fn = ms.value_and_grad(model.loss_fn, None, optimizer_Adam.parameters, has_aux=True)

    model.dnn.set_train()

    start_time = time.time()

    for epoch in range(1, 1 + NIter):
        (loss, loss_u, loss_f_u, loss_lambda_u), grads = grad_fn(
            model.x, model.y, model.t, model.x_f, model.y_f, model.t_f, model.u)
        optimizer_Adam(grads)

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Loss: %.3e, lambda_u1: %.3f, lambda_u2: %.3f, lambda_u3: %.3f, lambda_uxx: %.3f, lambda_uyy: %.3f, Time: %.2f' %
                  (epoch, loss.item(), model.lambda_u[1].item(), model.lambda_u[2].item(), model.lambda_u[3].item(), model.lambda_uxx.item(), model.lambda_uyy.item(), elapsed))

            initial_size = 4
            loss_history_Adam_Pretrain = np.empty([0])
            loss_u_history_Adam_Pretrain = np.empty([0])
            loss_f_u_history_Adam_Pretrain = np.empty([0])
            loss_lambda_u_history_Adam_Pretrain = np.empty([0])
            lambda_u_history_Adam_Pretrain = np.zeros((initial_size, 1))
            lambda_uxx_history_Adam_Pretrain = np.zeros((1, 1))
            lambda_uyy_history_Adam_Pretrain = np.zeros((1, 1))

            loss_history_Adam_Pretrain = np.append(loss_history_Adam_Pretrain, loss.numpy())
            lambda_u_history_Adam_Pretrain = np.append(
                lambda_u_history_Adam_Pretrain, model.lambda_u.numpy(), axis=1)
            loss_u_history_Adam_Pretrain = np.append(
                loss_u_history_Adam_Pretrain, loss_u.numpy())
            loss_f_u_history_Adam_Pretrain = np.append(
                loss_f_u_history_Adam_Pretrain, loss_f_u.numpy())
            loss_lambda_u_history_Adam_Pretrain = np.append(
                loss_lambda_u_history_Adam_Pretrain, loss_lambda_u.numpy())

            lambda_uxx_new = np.array([model.lambda_uxx.numpy()])
            lambda_uyy_new = np.array([model.lambda_uyy.numpy()])
            lambda_uxx_history_Adam_Pretrain = np.append(
                lambda_uxx_history_Adam_Pretrain, lambda_uxx_new, axis=1)
            lambda_uyy_history_Adam_Pretrain = np.append(
                lambda_uyy_history_Adam_Pretrain, lambda_uyy_new, axis=1)

            start_time = time.time()

    np.save(f'Loss-Coe/{second_path}/loss_history_Adam_Pretrain', loss_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/loss_u_history_Adam_Pretrain',
        loss_u_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/loss_f_u_history_Adam_Pretrain',
        loss_f_u_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/loss_lambda_u_history_Adam_Pretrain',
        loss_lambda_u_history_Adam_Pretrain)

    np.save(
        f'Loss-Coe/{second_path}/lambda_u_history_Adam_Pretrain',
        lambda_u_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_uxx_history_Adam_Pretrain',
        lambda_uxx_history_Adam_Pretrain)
    np.save(
        f'Loss-Coe/{second_path}/lambda_uyy_history_Adam_Pretrain',
        lambda_uyy_history_Adam_Pretrain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="./config/ICNet_SKG.yaml")
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
    description_SKG = args.description_SKG
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

    X_u_train, u_train, X_f_train = read_training_data(args)

    model_pretrain = InvarianceConstrainedNN(
        X_u_train,
        u_train,
        X_f_train,
        network_size,
        BatchNo,
        use_npu,
        msfloat_type)
    for epoch_train, lr_train in zip(epochs, learning_rate):
        train(model_pretrain, int(float(epoch_train)), lr_train)
    ms.save_checkpoint(model_pretrain.dnn, f'model/model.ckpt')

    lambda_uxx_value = model_pretrain.lambda_uxx.numpy()
    lambda_uyy_value = model_pretrain.lambda_uyy.numpy()
    lambda_u_value = model_pretrain.lambda_u.numpy()

    np.save(f'Loss-Coe/{Second_path}/lambda_uxx_value', lambda_uxx_value)
    np.save(f'Loss-Coe/{Second_path}/lambda_uyy_value', lambda_uyy_value)
    np.save(f'Loss-Coe/{Second_path}/lambda_u_value', lambda_u_value)

    if load_params:
        lambda_u_value = np.load(f'Loss-Coe/{second_path}/lambda_u_value.npy')
        lambda_uxx_value = np.load(f'Loss-Coe/{second_path}/lambda_uxx_value.npy')
        lambda_uyy_value = np.load(f'Loss-Coe/{second_path}/lambda_uyy_value.npy')
        model_ICCO = InvarianceConstrainedNN_STRdige(
            X_u_train,
            u_train,
            X_f_train,
            network_size,
            BatchNo,
            lambda_u_value,
            lambda_uxx_value,
            lambda_uyy_value,
            load_params,
            second_path,
            msfloat_type)
    else:
        model_ICCO = InvarianceConstrainedNN_STRdige(X_u_train, u_train, X_f_train, network_size, BatchNo,
                                                     lambda_u_value, lambda_uxx_value, lambda_uyy_value, load_params, second_path, msfloat_type)
    lam = 10**-5
    d_tol = 5
    lambda_u_STRidge = model_ICCO.call_trainstridge(lam, d_tol)

    print_pde(lambda_uxx_value, lambda_uyy_value, lambda_u_STRidge, description_ks, ut='u_t')

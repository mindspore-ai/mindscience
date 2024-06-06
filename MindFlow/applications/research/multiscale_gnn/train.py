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
# ==============================================================================
"""
train
"""
import os
import time
import argparse

import numpy as np
import scipy.io as sio
import mindspore
from mindspore import nn, ops, context
import mindspore.dataset as ds
from mindspore.amp import DynamicLossScaler, all_finite
from mindflow.utils import load_yaml_config

from src.models import MultiScaleGNN, MultiScaleGNNStructure
from src.datasets import read_training_data
from src.datasets import second_order_derivative_matix2d
from src.visualization import losses_curve

def get_weights_norm(model_to_get):
    """get_weights_norm"""
    weights_norm = 0.0
    for name, param in model_to_get.parameters_and_names():
        if 'weight' in name:
            weights_norm += param.norm()**2
    return weights_norm

def calculate_loss_p(p_hat, p):
    """calculate_loss_p"""
    loss_p = ops.mean((p_hat - p)**2)
    return loss_p

def calculate_loss_eq(p_hat, b):
    """calculate_loss_eq"""
    p_hat = p_hat * 0.01
    p_hat = ops.cat((p_hat[:, -1:], p_hat, p_hat[:, 0:1]), axis=1)
    p_hat = ops.cat((p_hat[:, :, -1:], p_hat, p_hat[:, :, 0:1]), axis=2)
    b_hat = (p_hat[:, 0:-2, 1:-1] - 2 * p_hat[:, 1:-1, 1:-1] + p_hat[:, 2:, 1:-1]) / dx**2 + \
            (p_hat[:, 1:-1, 0:-2] - 2 * p_hat[:, 1:-1, 1:-1] + p_hat[:, 1:-1, 2:]) / dy**2
    loss_eq = ops.mean((b_hat - b)**2)
    return loss_eq

def run_train(model_train, num_epochs, restore=False):
    """run_train"""
    if not os.path.exists(save_dir+'/checkpoint/'):
        os.makedirs(save_dir+'/checkpoint/')
    if not os.path.exists(save_dir+'/Figures/'):
        os.makedirs(save_dir+'/Figures/')
    if restore:
        loss_all = sio.loadmat(save_dir+'/loss_all.mat')['loss_all'].tolist()
        param_dict = mindspore.load_checkpoint(save_dir+'/checkpoint/model.ckpt')
        param_not_load, _ = mindspore.load_param_into_net(model_train, param_dict)
        print(param_not_load)
        model_train.set_train()

    else:
        loss_all = []

    train_dataset = ds.NumpySlicesDataset(data=(div_u_star_train, p_train), shuffle=True)
    train_dataset = train_dataset.batch(batch_size=batch_size)

    batch = div_u_star_train.shape[0]//batch_size
    learning_rate = nn.piecewise_constant_lr([500*batch, 1000*batch, 1500*batch, 2000*batch],
                                             [1e-3, 1e-4, 1e-5, 1e-6])

    optimizer = nn.AdamWeightDecay(model_train.trainable_params(),
                                   learning_rate=learning_rate,
                                   weight_decay=params['weight_decay'])

    def forward_fn(b, p):
        if grid_type == 'unstructure':
            inputs = b.reshape([-1, 1])
            p_hat = model_train(inputs)
            p_hat = p_hat.reshape([batch_size, nx, ny])
        if grid_type == 'structure':
            inputs = b.unsqueeze(1)
            p_hat = model_train(inputs)
            p_hat = p_hat.squeeze(1)
        loss_p = calculate_loss_p(p_hat, p)
        loss_eq = calculate_loss_eq(p_hat, b)
        loss = lambda_p * loss_p + lambda_eq * loss_eq
        if params['device_target'] == 'Ascend':
            loss = loss_scaler.scale(loss)
        return loss, loss_p, loss_eq

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(b, p):
        (loss, loss_p, loss_eq), grads = grad_fn(b, p)
        if params['device_target'] == 'Ascend':
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
        optimizer(grads)
        return loss, loss_p, loss_eq

    print('Training......')
    model_train.set_train()
    start_time = time.time()
    weights_norm = get_weights_norm(model_train)
    for epoch in range(1, num_epochs+1):
        loss_p_it, loss_eq_it = [], []
        it = 0
        for b, p in train_dataset.create_tuple_iterator():
            _, loss_p, loss_eq = train_step(b, p)

            loss_p_it.append(loss_p.asnumpy())
            loss_eq_it.append(loss_eq.asnumpy())

            elapsed = time.time() - start_time
            print('Epoch: %d, step: %d, Time: %.3f, weights_norm: %.3e, loss_p: %.3e, loss_eq: %.3e'
                  % (epoch, it, elapsed, weights_norm.asnumpy(), np.mean(loss_p_it), np.mean(loss_eq_it)))
            start_time = time.time()
            it += 1

        weights_norm = get_weights_norm(model_train)
        print('***********************************************************')
        loss_all.append([weights_norm.asnumpy(), np.mean(loss_p_it), np.mean(loss_eq_it)])

        if epoch % 1 == 0:
            mindspore.save_checkpoint(model_train, save_dir+'/checkpoint/model.ckpt')
            sio.savemat(save_dir+'/loss_all.mat', {'loss_all': np.array(loss_all)})

        if epoch % 10 == 0:
            losses_curve(np.array(loss_all), save_dir+'/Figures/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="./config/multiscale_gnn.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config_file_path)
    params = config["params"]

    context.set_context(mode=context.GRAPH_MODE if params['mode'].upper().startswith("GRAPH") \
                        else context.PYNATIVE_MODE,
                        device_target=params['device_target'],
                        device_id=params['device_id'])

    grid_type = params['grid_type']
    activ_fun = params['activ_fun']
    in_channels = params['in_channels']
    out_channels = params['out_channels']
    batch_size = params['batch_size']
    lambda_p = params['lambda_p']
    lambda_eq = params['lambda_eq']

    nx = params['nx']
    ny = params['ny']
    dx = 2.0 * np.pi / nx
    dy = 2.0 * np.pi / ny
    t_train = np.arange(0.1, 5.1, 0.1)
    div_u_star_train, p_train = read_training_data(t_train, dx, dy)
    a0 = second_order_derivative_matix2d(nx, ny, dx, dy)

    save_dir = f'./Savers/{grid_type}/{activ_fun}_lambda_p{lambda_p}lambda_eq{lambda_eq}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if activ_fun == 'swish':
        activation = nn.SiLU()
    if activ_fun == 'elu':
        activation = nn.ELU()
    if activ_fun == 'gelu':
        activation = nn.GELU()

    if params['device_target'] == 'Ascend':
        loss_scaler = DynamicLossScaler(1024, 2, 100)

    if grid_type == 'structure':
        model = MultiScaleGNNStructure(in_channels, out_channels, activation)
    if grid_type == 'unstructure':
        model = MultiScaleGNN(in_channels, out_channels, activation, a0)
    print(model)

    run_train(model, num_epochs=2000, restore=False)

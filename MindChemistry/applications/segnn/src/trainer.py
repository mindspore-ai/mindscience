# Copyright 2024 Huawei Technologies Co., Ltd
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
User-defined wrapper for training and testing.
"""
import time
import math

import mindspore as ms
from mindspore import nn, Profiler

from mindchemistry.e3.o3 import Irreps

from src.segnn import SEGNN
from src.balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from src.dataset import create_training_dataset


def make_irreps(data_params, model_params, dtype, ncon_dtype):
    """make irreps"""
    # input_irreps
    if data_params['feature_type'] == 'one_hot':
        input_irreps = Irreps("5x0e")
    elif data_params['feature_type'] == 'cormorant':
        input_irreps = Irreps("15x0e")
    else:
        raise ValueError("input_irreps init failed!")

    output_irreps = Irreps("1x0e")
    edge_attr_irreps = Irreps.spherical_harmonics(data_params['lmax_attr'])
    node_attr_irreps = Irreps.spherical_harmonics(data_params['lmax_attr'])
    additional_message_irreps = Irreps("1x0e")
    # hidden_irreps
    if model_params['subspace_type'] == 'weight_balanced':
        hidden_irreps = WeightBalancedIrreps(
            Irreps("{}x0e".format(model_params['hidden_features'])),
            node_attr_irreps,
            sh=True,
            lmax=model_params['lmax_h'],
            dtype=dtype,
            ncon_dtype=ncon_dtype
        )
    elif data_params['feature_type'] == 'cormorant':
        hidden_irreps = BalancedIrreps(model_params['lmax_h'], model_params['hidden_features'], True)
    else:
        raise ValueError("subspace type not found!")
    return input_irreps, hidden_irreps, output_irreps, edge_attr_irreps, node_attr_irreps, additional_message_irreps


def evaluate(model_fn, loss_fn, data_loader, target_mean, target_mad):
    """model predict"""
    ds_num = 0
    total_eval = 0.0
    for x, node_attr, edge_attr, edge_dist, label, edge_index, batch, node_mask, \
            edge_mask, _, batch_size in data_loader:

        _, pred = model_fn(x, node_attr, edge_attr, edge_index, edge_dist, batch,
                           label, node_mask, edge_mask, batch_size)
        n = label.shape[0]
        ds_num += n
        total_eval += n*loss_fn(pred*target_mad + target_mean, label).asnumpy()
    return total_eval / ds_num


def train_epochs(epochs, configs, loss_fn, net, model_params, target_mean, target_mad, profiler,
                 train_loader, eval_loader, test_loader, forward, backward_step):
    """
    train the network for epochs
    """
    res_log = open(configs['log_file'], 'w')
    loss_train = []
    stat_i = 0
    num_sample = 0
    loss_steps = 0.0  # sum the train loss of steps
    train_mae_sum = 0.0  # sum the mae of steps
    best_eval_mae = 1e30
    train_print_interval = 100  # printing interval for training loss
    for epoch in range(epochs):
        total_train = 0.0
        t0 = time.time()

        step = 0
        t_step = time.time()
        for x, node_attr, edge_attr, edge_dist, label, edge_index, batch, node_mask, \
                edge_mask, _, batch_size in train_loader:

            loss, pred = backward_step(x, node_attr, edge_attr, edge_index, edge_dist, batch,
                                       label, node_mask, edge_mask, batch_size)

            stat_i += 1
            num_sample += label.shape[0]
            loss_steps += loss.asnumpy()
            total_train += loss.asnumpy()
            train_mae_sum += (loss_fn(pred*target_mad + target_mean, label) * label.shape[0]).asnumpy()

            if step % train_print_interval == 0:
                train_log = f"epoch: {epoch:3d}, step: {step:3d}, loss: {loss_steps/stat_i:<8.8f}, " \
                            f"train MAE: {train_mae_sum/num_sample:<8.4f}, time: {time.time() - t_step:.2f} "
                print(train_log)
                res_log.write(train_log + "\n")
                res_log.flush()

                stat_i = 0
                num_sample = 0
                loss_steps = 0.0
                train_mae_sum = 0.0
                t_step = time.time()
            step += 1

            if configs['profiling'] and step == configs['profiling_step']:
                profiler.analyse()
                return

        loss_train.append(total_train / step)
        print(f"epoch: {epoch:3d}, train loss: {loss_train[-1]:<8.8f}, time used: {time.time() - t0:.2f} ")

        # valid dataset
        if epoch % configs['eval_interval'] == 0:
            eval_time = time.time()
            eval_mae = evaluate(forward, loss_fn, eval_loader, target_mean, target_mad)
            eval_log = f"eval MAE:{eval_mae:<8.4f}, time used: {time.time() - eval_time:.2f}"
            res_log.write(eval_log + "\n")
            res_log.flush()
            print(eval_log)
            if eval_mae < best_eval_mae:
                best_eval_mae = eval_mae
                ms.save_checkpoint(net, model_params['ckpt_file'])

    # test dataset
    param_dict = ms.load_checkpoint(model_params['ckpt_file'])
    ms.load_param_into_net(net, param_dict)
    test_time = time.time()
    test_mae = evaluate(forward, loss_fn, test_loader, target_mean, target_mad)
    test_log = f"test MAE:{test_mae:<8.4f}, time used: {time.time() - test_time:.2f}"
    res_log.write(test_log + "\n")
    res_log.close()
    print(test_log)


def train(dtype=ms.float32, configs=None, ncon_dtype=None):
    """Train the model on the train dataset."""
    data_params = configs['data']
    model_params = configs['model']
    optimizer_params = configs['optimizer']

    # == dataset ==
    print('Loading data...                ')
    train_loader, eval_loader, test_loader, train_stats = create_training_dataset(data_params, dtype)
    target_mean, target_mad = train_stats[0], train_stats[1]
    print("train_set's mean and mad: ", target_mean, target_mad)

    # == model ==
    profiler = None
    if configs['profiling']:
        profiler = Profiler(profile_memory=True)
    print('Initializing model...                ')
    input_irreps, hidden_irreps, output_irreps, edge_attr_irreps, node_attr_irreps, \
        additional_message_irreps = make_irreps(data_params, model_params, dtype, ncon_dtype)

    net = SEGNN(
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        num_layers=model_params['num_layers'],
        norm=model_params['norm_type'],
        pool=model_params['pool_type'],
        task=model_params['task'],
        additional_message_irreps=additional_message_irreps,
        dtype=dtype,
        ncon_dtype=ncon_dtype
    )

    # loss and optimizer
    loss_fn = nn.L1Loss()
    lr_decay = 0.1
    epochs = int(optimizer_params['num_epoch'])
    lr = float(optimizer_params['learning_rate'])
    weight_decay = float(optimizer_params['weight_decay'])
    lr_rates = [lr, lr*lr_decay, lr*lr_decay*lr_decay]
    steps_epoch = math.ceil(len(train_loader) / (1.0 * data_params['batch_size']))
    lr_update_interval = [0.8, 0.9]
    first_epoch, second_epoch = int(lr_update_interval[0] * epochs), int(lr_update_interval[1] * epochs)
    first_epoch_judge = 1 if first_epoch < 1 else first_epoch
    second_epoch_judge = 1 if second_epoch < 1 else second_epoch
    milestone = [first_epoch_judge*steps_epoch, second_epoch_judge*steps_epoch, epochs*steps_epoch]
    dynamic_lr = nn.piecewise_constant_lr(milestone, lr_rates)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=dynamic_lr, weight_decay=weight_decay)

    def forward(x, node_attr, edge_attr, edge_index, edge_dist, batch,
                label, node_mask, edge_mask, batch_size):
        pred = net(x, node_attr, edge_attr, edge_index, edge_dist, batch,
                   node_mask, edge_mask, batch_size)
        norm_label = (label - target_mean) / target_mad
        loss = loss_fn(pred, norm_label)
        return loss, pred

    if configs["run_mode"] == 'infer':
        t_start = time.time()
        param_dict = ms.load_checkpoint(model_params['ckpt_file'])
        ms.load_param_into_net(net, param_dict)
        test_mae = evaluate(forward, loss_fn, test_loader, target_mean, target_mad)
        print(f"test MAE: {test_mae:<8.4f}, time used: {time.time()-t_start:.2f}")
        return

    backward = ms.value_and_grad(forward, None, optimizer.parameters, has_aux=True)

    def backward_step(x, node_attr, edge_attr, edge_index, edge_dist, batch, \
                    label, node_mask, edge_mask, batch_size):
        (loss, pred), grads = backward(x, node_attr, edge_attr, edge_index, edge_dist, batch, \
                                label, node_mask, edge_mask, batch_size)
        optimizer(grads)
        return loss, pred

    print('Initializing train...                ')
    train_epochs(epochs, configs, loss_fn, net, model_params, target_mean, target_mad, profiler,
                 train_loader, eval_loader, test_loader, forward, backward_step)

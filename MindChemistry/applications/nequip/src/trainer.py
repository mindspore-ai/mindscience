# Copyright 2022 Huawei Technologies Co., Ltd
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
import logging
import math
import os
import time

import mindspore as ms
from mindspore import Profiler
from mindspore import nn
from src.dataset import create_training_dataset, _unpack
from src.utils import training_bar

from mindchemistry.cell import EnergyNet


def generate_learning_rate(learning_rate, warmup_steps, step_num):
    warmup_scale = warmup_steps ** -1.5
    lr = []
    for s in range(1, step_num + 1):
        lr1 = s ** -0.5
        lr2 = s * warmup_scale
        lr.append(learning_rate * min(lr1, lr2))
    return lr


def train(dtype=ms.float32, configs=None):
    """Train the model on the train dataset."""
    data_params = configs.get('data')
    model_params = configs.get('model')
    optimizer_params = configs.get('optimizer')
    pred_force = configs.get('pred_force')
    is_profiling = configs.get('profiling')
    enable_mix_precision = configs.get('enable_mix_precision')
    if enable_mix_precision:
        ncon_dtype = ms.float16
    else:
        ncon_dtype = ms.float32

    load_ckpt = configs.get('load_ckpt')
    load_ckpt_path = configs.get('load_ckpt_path')
    save_ckpt = configs.get('save_ckpt')
    save_ckpt_interval = configs.get('save_ckpt_interval')
    save_ckpt_path = configs.get('save_ckpt_path')
    if save_ckpt:
        os.makedirs(save_ckpt_path, exist_ok=True)

    logging.info('Loading data...')
    trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch, num_type = create_training_dataset(
        config=data_params, dtype=dtype, pred_force=configs.get('pred_force'))
    # == Model ==
    logging.info('Initializing model...')
    net = EnergyNet(irreps_embedding_out=model_params.get('irreps_embedding_out'),
                    irreps_conv_out=model_params.get('irreps_conv_out'),
                    chemical_embedding_irreps_out=model_params.get('chemical_embedding_irreps_out'),
                    num_layers=model_params.get('num_layers'),
                    num_type=num_type,
                    r_max=model_params.get('r_max'),
                    hidden_mul=model_params.get('hidden_mul'),
                    pred_force=pred_force,
                    dtype=dtype,
                    ncon_dtype=ncon_dtype
                    )

    if load_ckpt:
        logging.info('Loading checkpoint: %s', load_ckpt_path)
        ms.load_checkpoint(load_ckpt_path, net)

    loss_fn = nn.MSELoss()
    metric_fn = nn.MAELoss()
    total_steps_num = optimizer_params.get('num_epoch') * math.ceil(
        data_params.get('n_train') / data_params.get('batch_size'))
    lr_schedule = generate_learning_rate(optimizer_params.get('learning_rate'), optimizer_params.get('warmup_steps'),
                                         total_steps_num)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr_schedule,
                        use_amsgrad=optimizer_params.get('use_amsgrad'))

    def forward(batch, x, pos, edge_src, edge_dst, energy, force, batch_size, sep):
        pred = net(batch, x, pos, edge_src, edge_dst, batch_size)
        if pred_force:
            loss_energy = loss_fn(pred[0], energy)
            loss_force = loss_fn(pred[1], force)
            if sep:
                return loss_energy, loss_force
            return loss_energy + 1000. * loss_force
        return loss_fn(pred, energy)

    backward = ms.value_and_grad(forward, None, optimizer.parameters)

    @ms.jit
    def train_step(batch, x, pos, edge_src, edge_dst, energy, force, size, sep):
        loss_, grads_ = backward(batch,
                                 x, pos,
                                 edge_src,
                                 edge_dst,
                                 energy, force,
                                 size, sep)
        optimizer(grads_)
        return loss_

    def validation(eval_batch, eval_edge_index, evalset):
        dataset_size = evalset.get_dataset_size()
        total_eval = 0
        total_eval_energy = 0
        total_eval_force = 0
        total_metric = 0
        total_metric_energy = 0
        total_metric_force = 0
        for _, data_dict_val in enumerate(evalset.create_dict_iterator()):
            batch_size_val = evalset.get_batch_size()
            inputs_val, label_val = _unpack(data_dict_val)
            pred = net(eval_batch, inputs_val[0], inputs_val[1], eval_edge_index[0], eval_edge_index[1], batch_size_val)

            if not pred_force:
                loss_val = loss_fn(pred, label_val[0]).asnumpy()
                metric = metric_fn(pred, label_val[0]).asnumpy()
                total_eval += loss_val
                total_metric += metric
            else:
                loss_val_energy = loss_fn(pred[0], label_val[0]).asnumpy()
                loss_val_force = loss_fn(pred[1], label_val[1]).asnumpy()
                metric_energy = metric_fn(pred[0], label_val[0]).asnumpy()
                metric_force = metric_fn(pred[1], label_val[1]).asnumpy()
                total_eval_energy += loss_val_energy
                total_eval_force += loss_val_force
                total_metric_energy += metric_energy
                total_metric_force += metric_force

        if not pred_force:
            loss = total_eval / dataset_size
            metric_mean = total_metric / dataset_size
            logging.info('eval loss: %8.8f   metric: %.8f', loss, metric_mean)
        else:
            loss = (total_eval_energy / dataset_size, total_eval_force / dataset_size)
            metric_mean = (total_metric_energy / dataset_size, total_metric_force / dataset_size)
            logging.info('eval energy loss:  %8.8f eval force loss: %8.8f  metric energy: %.8f  metric force: %.8f',
                         loss[0], loss[1], metric_mean[0], metric_mean[1])
        return loss, metric_mean

    # == Training ==
    t0 = time.time()

    if is_profiling:
        logging.info('Initializing profiler...')
        profiler = Profiler(output_path="profiler_data")

    logging.info('Initializing train...')
    loss_train = []
    loss_eval = []
    metric = []
    eval_steps = optimizer_params.get('eval_steps')
    for epoch in range(optimizer_params.get('num_epoch')):
        epoch_loss = 0
        ti = time.time()
        for current, data_dict in enumerate(trainset.create_dict_iterator()):
            inputs, label = _unpack(data_dict)
            batch_size_train = trainset.get_batch_size()

            loss = train_step(train_batch,
                              inputs[0], inputs[1],
                              train_edge_index[0], train_edge_index[1],
                              label[0], label[1], batch_size_train, False)

            epoch_loss += loss.asnumpy()
            training_bar(epoch, size=trainset.get_dataset_size(), current=current)

        epoch_loss = epoch_loss / trainset.get_dataset_size()
        loss_train.append(epoch_loss)
        t_now = time.time()
        t_gap = t_now - ti
        t_all = t_now - t0
        logging.info('epoch %d:  train loss: %8.8f, time gap: %.2f, total time used: %.2f', (epoch + 1), epoch_loss,
                     t_gap, t_all)

        if (epoch + 1) % eval_steps == 0:
            loss, metric_mean = validation(eval_batch, eval_edge_index, evalset)
            loss_eval.append(loss)
            metric.append(metric_mean)
        if save_ckpt and epoch % save_ckpt_interval == 0:
            logging.info('Saving checkpoint file ...')
            ms.save_checkpoint(net, f"{save_ckpt_path}/NequIP_rmd_{epoch}.ckpt")

    if is_profiling:
        profiler.analyse()
    if save_ckpt:
        ms.save_checkpoint(net, f"{save_ckpt_path}/NequIP_rmd.ckpt")

    return loss_train, loss_eval, metric, lr_schedule

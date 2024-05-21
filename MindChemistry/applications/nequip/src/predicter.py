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
User-defined wrapper for testing.
"""
import logging
import os
import numpy as np

import mindspore as ms
from mindspore import nn
from src.dataset import create_training_dataset, _unpack

from mindchemistry.cell import EnergyNet


def evaluation(dtype, configs):
    """ evaluation """
    data_params = configs.get('data')
    model_params = configs.get('model')
    pred_force = configs.get('pred_force')

    load_ckpt_path = configs.get('load_ckpt_path')

    logging.info('Loading data...')
    _, _, _, evalset, eval_edge_index, eval_batch, num_type = create_training_dataset(
        config=data_params, dtype=dtype, pred_force=configs.get('pred_force'))

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
                    # ncon_dtype=ms.float16
                    )
    loss_fn = nn.MSELoss()
    metric_fn = nn.MAELoss()

    logging.info('Loading checkpoint %s', load_ckpt_path)
    ms.load_checkpoint(load_ckpt_path, net)

    logging.info('Initializing Evaluation...')
    dataset_size = evalset.get_dataset_size()
    total_loss = 0
    total_loss_energy = 0
    total_loss_force = 0
    total_metric = 0
    total_metric_energy = 0
    total_metric_force = 0
    pred_list = []
    pred_energy_list = []
    pred_force_list = []

    for _, data_dict_val in enumerate(evalset.create_dict_iterator()):

        batch_size_val = evalset.get_batch_size()
        inputs_val, label_val = _unpack(data_dict_val)

        pred = net(eval_batch, inputs_val[0], inputs_val[1],
                   eval_edge_index[0], eval_edge_index[1],
                   batch_size_val)

        if not pred_force:
            loss = loss_fn(pred, label_val[0]).asnumpy()
            metric = metric_fn(pred, label_val[0]).asnumpy()
            total_loss += loss
            total_metric += metric
            pred_list.append(pred.asnumpy())
        else:
            loss_energy = loss_fn(pred[0], label_val[0]).asnumpy()
            loss_force = loss_fn(pred[1], label_val[1]).asnumpy()
            metric_energy = metric_fn(pred[0], label_val[0]).asnumpy()
            metric_force = metric_fn(pred[1], label_val[1]).asnumpy()
            total_loss_energy += loss_energy
            total_loss_force += loss_force
            total_metric_energy += metric_energy
            total_metric_force += metric_force
            pred_energy_list.append(pred[0].asnumpy())
            pred_force_list.append(pred[1].asnumpy())

    if not pred_force:
        np.save(os.path.join(configs.get('data').get('save_path'), 'pred.npy'), pred_list)
        loss_mean = total_loss / dataset_size
        metric_mean = total_metric / dataset_size
        logging.info('loss_mean: %8.8f   metric_mean: %.8f', loss_mean, metric_mean)
    else:
        np.save(os.path.join(configs.get('data').get('save_path'), 'pred_energy.npy'), pred_energy_list)
        np.save(os.path.join(configs.get('data').get('save_path'), 'pred_force.npy'), pred_force_list)
        pred_list = [pred_energy_list, pred_force_list]
        loss_mean = (total_loss_energy / dataset_size, total_loss_force / dataset_size)
        metric_mean = (total_metric_energy / dataset_size, total_metric_force / dataset_size)
        logging.info('loss_energy_mean: %8.8f loss_force_mean: %8.8f metric_energy_mean: %.8f metric_force_mean: %.8f',
                     loss_mean[0], loss_mean[1], metric_mean[0], metric_mean[1])

    return pred_list, loss_mean, metric_mean

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
import time

from mindspore import nn
import mindspore as ms

from mindchemistry.cell import EnergyNet

from src.dataset import create_training_dataset, _unpack
from src.utils import training_bar


def train(dtype=ms.float32, configs=None):
    """Train the model on the train dataset."""
    data_params = configs['data']
    model_params = configs['model']
    optimizer_params = configs['optimizer']
    pred_force = configs['pred_force']

    print('\rLoading data...                ', end='')
    trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch, num_type = create_training_dataset(
        config=data_params, dtype=dtype, pred_force=configs['pred_force'])
    # == Model ==
    print('\rInitializing model...              ', end='')
    net = EnergyNet(irreps_embedding_out=model_params['irreps_embedding_out'],
                    irreps_conv_out=model_params['irreps_conv_out'],
                    chemical_embedding_irreps_out=model_params['chemical_embedding_irreps_out'],
                    num_layers=model_params['num_layers'],
                    num_type=num_type,
                    r_max=model_params['r_max'],
                    hidden_mul=model_params['hidden_mul'],
                    pred_force=pred_force,
                    dtype=dtype
                    )

    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=optimizer_params['learning_rate'],
                        use_amsgrad=optimizer_params['use_amsgrad'])

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

    # == Training ==
    print('\rInitializing train...         ', end='')
    loss_eval = []
    loss_train = []
    for epoch in range(optimizer_params['num_epoch']):
        total_train = 0
        t0 = time.time()
        for current, data_dict in enumerate(trainset.create_dict_iterator()):
            batch_size_train = trainset.get_batch_size()
            inputs, label = _unpack(data_dict)

            loss, grads = backward(train_batch,
                                   *inputs,
                                   train_edge_index[0],
                                   train_edge_index[1],
                                   *label,
                                   batch_size_train,
                                   False)
            optimizer(grads)
            total_train += loss.asnumpy()
            training_bar(epoch, size=trainset.get_dataset_size(), current=current)

        loss_train += [total_train / trainset.get_dataset_size()]

        if epoch % 10 == 0:
            print('\n', end='')
            if epoch == 0:
                print('Initializing eval...       ', end='')
            if not pred_force:
                total_eval = 0
            else:
                eval_energy, eval_force = 0, 0
            for current, data_dict in enumerate(evalset.create_dict_iterator()):
                batch_size_eval = evalset.get_batch_size()
                inputs, label = _unpack(data_dict)

                loss = forward(eval_batch,
                               *inputs,
                               eval_edge_index[0],
                               eval_edge_index[1],
                               *label,
                               batch_size_eval,
                               True)
                if not pred_force:
                    total_eval += loss.asnumpy()
                else:
                    eval_energy += loss[0].asnumpy()
                    eval_force += loss[1].asnumpy()

            if not pred_force:
                loss_eval.append(total_eval / evalset.get_dataset_size())
            else:
                loss_eval.append((eval_energy / evalset.get_dataset_size(),
                                  eval_force / evalset.get_dataset_size()))

            print(
                f'\rtrain loss: {loss_train[-1]:<8.8f}, eval loss: {loss_eval[-1]}, time used: {time.time() - t0:.2f}')
        else:
            print(f'\rtrain loss: {loss_train[-1]:<8.8f}, time used: {time.time() - t0:.2f}   ')

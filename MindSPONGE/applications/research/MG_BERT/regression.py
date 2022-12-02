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
"""Regression of MG-BERT"""

import os
import argparse
import time
import numpy as np
from sklearn.metrics import r2_score
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from src.dataset import GraphRegressionDataset
from src.model import PredictModel, BertModel
from src.utils import data_rebuild_c_r

medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3', 'addH': True}
small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights', 'addH': True}
medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
medium2 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights2', 'addH': True}
large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights', 'addH': True}
medium_without_h = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_without_H',
                    'addH': False}
medium_without_pretrain = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256,
                           'path': 'medium_without_pretraining_weights', 'addH': True}


def val(val_dataset, model, value_range):
    """Validation"""
    expand_dims = ops.ExpandDims()
    y_true = []
    y_preds = []
    model.set_train(False)
    for x, adjoin_matrix, y in val_dataset:
        x, adjoin_matrix = data_rebuild_c_r(x, adjoin_matrix)
        seq = ops.Cast()(ops.equal(x, 0), ms.float32)
        mask = expand_dims(seq, 1)
        mask = expand_dims(mask, 2)

        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix)
        y_true.append(y.asnumpy())
        y_preds.append(preds.asnumpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    r2_new = r2_score(y_true, y_preds)

    val_mse = ms.nn.metrics.MSE()
    val_mse.clear()
    val_mse.update(y_true, y_preds)
    val_mse = val_mse.eval() * (value_range ** 2)
    print('val r2: {:.4f}'.format(r2_new), 'val mse:{:.4f}'.format(val_mse))
    return r2_new, y_true, y_preds


def test(test_dataset, model, task, value_range):
    """Test"""
    expand_dims = ops.ExpandDims()
    y_true = []
    y_preds = []
    param_dict_model = ms.load_checkpoint('regression_weights/{}.ckpt'.format(task))
    ms.load_param_into_net(model, param_dict_model)
    model.set_train(False)
    for x, adjoin_matrix, y in test_dataset:
        x, adjoin_matrix = data_rebuild_c_r(x, adjoin_matrix)
        seq = ops.Cast()(ops.equal(x, 0), ms.float32)
        mask = expand_dims(seq, 1)
        mask = expand_dims(mask, 2)
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix)
        y_true.append(y.asnumpy())
        y_preds.append(preds.asnumpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)

    test_r2 = r2_score(y_true, y_preds)

    test_mse = ms.nn.metrics.MSE()
    test_mse.clear()
    test_mse.update(y_true, y_preds)
    test_mse = test_mse.eval() * (value_range ** 2)
    return test_r2, test_mse


def main(seed1, trained_epoch, task, pretraining, vocab_size):
    """regression"""
    arch = medium3

    pretraining_str = 'pretraining' if pretraining else ''


    print(task)
    seed1 = seed1

    num_layers = arch.get('num_layers', "abc")
    num_heads = arch.get('num_heads', "abc")
    d_model = arch.get('d_model', "abc")
    addh = arch.get('addH', "abc")

    dff = d_model * 2

    ms.set_seed(seed=seed1)

    graph_dataset = GraphRegressionDataset('data/reg/{}.txt'.format(task), smiles_field='SMILES',
                                           label_field='Label', addh=addh)

    train_dataset, test_dataset, val_dataset = graph_dataset.get_data()

    value_range = graph_dataset.value_range

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))

    x, adjoin_matrix = data_rebuild_c_r(x, adjoin_matrix)

    seq = ops.Cast()(ops.equal(x, 0), ms.float32)
    expand_dims = ops.ExpandDims()
    mask = expand_dims(seq, 1)
    mask = expand_dims(mask, 2)

    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff,
                         num_heads=num_heads, vocab_size=vocab_size, dense_dropout=0.15)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff,
                         num_heads=num_heads, vocab_size=vocab_size)
        param_dict = ms.load_checkpoint(arch.get('path', "path") +
                                        '/bert_weights{}_{}.ckpt'.format(arch.get('name', "name"), trained_epoch))
        ms.load_param_into_net(temp, param_dict)
        ms.save_checkpoint(temp.encoder, arch.get('path', "path") +
                           '/bert_weights_encoder{}_{}.ckpt'.format(arch.get('name', "name"), trained_epoch))
        del temp

        ms.load_param_into_net(model.encoder, param_dict)
        print('load_weights')

    r2 = -10
    stopping_monitor = 0

    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=10e-5)
    loss_fn = nn.MSELoss()

    def forward_fn(x, adjoin_matrix, y):
        seq = ops.Cast()(ops.equal(x, 0), ms.float32)
        mask = ops.ExpandDims()(seq, 1)
        mask = ops.ExpandDims()(mask, 2)
        logits = model(x, adjoin_matrix, mask)
        loss = loss_fn(logits, y)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(x, adjoin_matrix, y):
        (loss, logits), grads = grad_fn(x, adjoin_matrix, y)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    for epoch in range(trained_epoch):
        start1 = time.time()
        model.set_train()
        for x, adjoin_matrix, y in train_dataset:
            x, adjoin_matrix = data_rebuild_c_r(x, adjoin_matrix)

            loss, _ = train_step(x, adjoin_matrix, y)
        print('epoch: ', epoch, 'loss: {:.4f}'.format(loss.asnumpy()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start1))

        r2_new, y_true, y_preds = val(val_dataset, model, value_range)
        if r2_new > r2:
            r2 = r2_new
            stopping_monitor = 0
            if not os.path.exists('{}/'.format(arch.get('path', "path"))):
                os.makedirs('{}/'.format(arch.get('path', "path")))
            np.save('{}/{}{}{}{}{}'.format(arch.get('path', "path"), task, seed, arch.get('name', "name"),
                                           trained_epoch, pretraining_str), [y_true, y_preds])
            ms.save_checkpoint(model, 'regression_weights/{}.ckpt'.format(task))
        else:
            stopping_monitor += 1
        print('best r2: {:.4f}'.format(r2))
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 20:
            break

    test_r2, test_mse = test(test_dataset, model, task, value_range)
    print('test r2:{:.4f}'.format(test_r2), 'test mse:{:.4f}'.format(test_mse))

    return r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_epoch', metavar='N', type=int, default=100,
                        help='training epochs, default=150')
    parser.add_argument('--task', metavar='TASK', type=str, default='logS',
                        help='Task')
    parser.add_argument('--pretraining', default=0, help='pretrain_flag, 0 or 1')
    parser.add_argument('--device_id', help='device id', type=int, default=2)
    parser.add_argument('--device', help='device', type=str, default='GPU')
    parser.add_argument('--vocab_size', metavar='N', type=int, default=17,
                        help='vocab size, default=17')
    args = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device,
                   device_id=args.device_id, pynative_synchronize=True)
    start = time.time()
    r2_list = []
    for seed in [7, 17, 27, 37, 47]:
        print(seed)
        m_r2 = main(seed, args.trained_epoch, args.task, args.pretraining, args.vocab_size)
        r2_list.append(m_r2)
    end = time.time()
    cost_time = end - start
    print(r2_list)
    print('cost time:', cost_time)

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
"""Classification of MG-BERT"""

import os
import argparse
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from src.dataset import GraphClassificationDataset
from src.model import BertModel, ClassificationModel
from src.utils import data_rebuild_c_r


medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3', 'addH': True}
small = {'name': 'Small', 'num_layers': 3, 'num_heads': 2, 'd_model': 128, 'path': 'small_weights', 'addH': True}
medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 512, 'path': 'large_weights', 'addH': True}


def val(val_dataset, model):
    """Validation"""
    expand_dims = ops.ExpandDims()
    sigmoid = ops.Sigmoid()
    y_true = []
    y_preds = []
    y_preds_label = []
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
    y_preds = sigmoid(ms.Tensor(y_preds, ms.float32)).asnumpy()
    auc_new = roc_auc_score(y_true, y_preds)
    for _, yp_item in enumerate(y_preds):
        if yp_item - 0.5 > 0:
            y_preds_label.append(1)
        else:
            y_preds_label.append(0)

    val_acc = np.count_nonzero(np.equal(np.array(y_preds_label), y_true)) / len(y_preds)
    output = [auc_new, val_acc, y_true, y_preds]
    return output


def test(test_dataset, model, task):
    """Test"""
    expand_dims = ops.ExpandDims()
    sigmoid = ops.Sigmoid()
    y_true = []
    y_preds = []
    y_preds_label = []
    param_dict_model = ms.load_checkpoint('classification_weights/{}_{}.ckpt'.format(task, seed))
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
    y_preds = sigmoid(ms.Tensor(y_preds, ms.float32)).asnumpy()
    test_auc = roc_auc_score(y_true, y_preds)

    for _, yp_item in enumerate(y_preds):
        if yp_item - 0.5 > 0:
            y_preds_label.append(1)
        else:
            y_preds_label.append(0)

    test_acc = np.count_nonzero(np.equal(np.array(y_preds_label), y_true)) / len(y_preds)

    print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_acc))
    return test_auc


def classification(seed1, trained_epoch, task, pretraining, vocab_size):
    """Classification"""
    def forward_fn(x, adjoin_matrix, y):
        seq = ops.Cast()(ops.equal(x, 0), ms.float32)
        mask = ops.ExpandDims()(seq, 1)
        mask = ops.ExpandDims()(mask, 2)
        logits = model(x, adjoin_matrix, mask)
        logits = ops.reshape(logits, (logits.shape[0] * logits.shape[1],))
        y = ops.Cast()(ops.reshape(y, (y.shape[0] * y.shape[1],)), ms.float32)
        loss = loss_fn(logits, y)
        return loss, logits

    print(task)

    arch = medium3
    pretraining_str = 'pretraining' if pretraining else ''


    num_layers = arch.get('num_layers', "abc")
    num_heads = arch.get('num_heads', "abc")
    d_model = arch.get('d_model', "abc")

    dff = d_model * 2

    seed1 = seed1
    np.random.seed(seed=seed1)
    ms.set_seed(seed=seed1)

    train_dataset, test_dataset, val_dataset = GraphClassificationDataset('data/clf/{}.txt'.format(task),
                                                                          smiles_field='SMILES',
                                                                          label_field='Label', addh=True).get_data()

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    x, adjoin_matrix = data_rebuild_c_r(x, adjoin_matrix)

    seq = ops.Cast()(ops.equal(x, 0), ms.float32)
    expand_dims = ops.ExpandDims()
    mask = expand_dims(seq, 1)
    mask = expand_dims(mask, 2)

    model = ClassificationModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads,
                                vocab_size=vocab_size, dense_dropout=0.15)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)

        param_dict = ms.load_checkpoint(arch.get('path', "path") +
                                        '/bert_weights{}_{}.ckpt'.format(arch.get('name', "name"), trained_epoch))
        ms.load_param_into_net(temp, param_dict)
        ms.save_checkpoint(temp.encoder, arch.get('path', "path") +
                           '/bert_weights_encoder{}_{}.ckpt'.format(arch.get('name', "name"), trained_epoch))
        del temp

        ms.load_param_into_net(model.encoder, param_dict)
        print('load_weights')

    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    auc1 = -10
    stopping_monitor = 0
    for epoch in range(trained_epoch):
        start1 = time.time()
        model.set_train()
        for x, adjoin_matrix, y in train_dataset:
            x, adjoin_matrix = data_rebuild_c_r(x, adjoin_matrix)
            grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, _), grads = grad_fn(x, adjoin_matrix, y)
            loss = ops.depend(loss, optimizer(grads))

        print('epoch: ', epoch, 'loss: {:.4f}'.format(loss.asnumpy()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start1))

        auc_new, val_acc, y_true, y_preds = val(val_dataset, model)
        print('val auc:{:.4f}'.format(auc_new), 'val accuracy:{:.4f}'.format(val_acc))
        if auc_new > auc1:
            auc1 = auc_new
            stopping_monitor = 0
            if not os.path.exists('{}/'.format(arch.get('path', "path"))):
                os.makedirs('{}/'.format(arch.get('path', "path")))
            np.save('{}/{}{}{}{}{}'.format(arch.get('path', "path"), task, seed, arch.get('name', "name"),
                                           trained_epoch, pretraining_str),
                    [y_true, y_preds])
            ms.save_checkpoint(model, 'classification_weights/{}_{}.ckpt'.format(task, seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc1))
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 20:
            break

    test_auc = test(test_dataset, model, task)
    return test_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_epoch', metavar='N', type=int, default=100,
                        help='training epochs, default=150')
    parser.add_argument('--task', metavar='TASK', type=str, default='Pgp_sub',
                        help='Task')
    parser.add_argument('--pretraining', default=0, help='pretrain_flag, 0 or 1')
    parser.add_argument('--device_id', help='device id', type=int, default=1)
    parser.add_argument('--device', help='device', type=str, default='GPU')
    parser.add_argument('--vocab_size', metavar='N', type=int, default=17,
                        help='vocab size, default=17')
    args = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device,
                   device_id=args.device_id, pynative_synchronize=True)
    start = time.time()
    auc_list = []
    for seed in [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]:
        print(seed)
        auc = classification(seed, args.trained_epoch, args.task, args.pretraining, args.vocab_size)
        auc_list.append(auc)
    end = time.time()
    cost_time = end - start
    print(auc_list)
    print('cost time:', cost_time)

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
"""Pretrain of MG-BERT"""

import argparse
import time
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from src.utils import SampleLoss, data_rebuild_pre
from src.model import BertModel
from src.dataset import GraphBertDataset


small = {'name': 'Small', 'num_layers': 3, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights', 'addH': True}
medium = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights', 'addH': True}
medium3 = {'name': 'Medium', 'num_layers': 6, 'num_heads': 4, 'd_model': 256, 'path': 'medium_weights3', 'addH': True}
large = {'name': 'Large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights', 'addH': True}
medium_balanced = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_balanced',
                   'addH': True}
medium_without_h = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_without_H',
                    'addH': False}


class CustomWithLossCell(nn.Cell):
    """Training"""
    def __init__(self, backbone, loss_fun):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fun

    def construct(self, t_x, t_adjoin_matrix, t_y, t_char_weight, t_mask):
        output = self._backbone(t_x, t_adjoin_matrix, t_mask)
        output = ops.reshape(output, (-1, output.shape[2]))
        t_y = ops.reshape(t_y, (t_y.shape[0] * t_y.shape[1],))
        t_char_weight = ops.reshape(t_char_weight, (t_char_weight.shape[0] * t_char_weight.shape[1],))
        return self._loss_fn(output, t_y, t_char_weight)


def main(trained_epoch, data_path, vocab_size):
    arch = medium3
    num_layers = arch.get('num_layers', "abc")
    num_heads = arch.get('num_heads', "abc")
    d_model = arch.get('d_model', "abc")
    addh = arch.get('addH', "abc")

    dff = d_model * 2

    train_dataset, _ = GraphBertDataset(path=data_path, smiles_field='canonical_smiles',
                                        addh=addh).get_data()


    model = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
    loss_fn = SampleLoss()
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=1e-4)
    net_with_loss = CustomWithLossCell(model, loss_fn)
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)

    start1 = time.time()
    for epoch in range(trained_epoch):
        print(epoch)
        start = time.time()
        train_net.set_train()
        for (batch, (x, adjoin_matrix, y, char_weight)) in enumerate(train_dataset):
            x, adjoin_matrix, y, char_weight = data_rebuild_pre(x, adjoin_matrix, y, char_weight)
            seq = ops.Cast()(ops.equal(x, 0), ms.float32)
            mask = ops.ExpandDims()(seq, 1)
            mask = ops.ExpandDims()(mask, 2)
            loss = train_net(x, adjoin_matrix, y, char_weight, mask)

            if batch % 100 == 0:
                loss, _ = loss.asnumpy(), batch
                print(f"loss: {loss:>7f}")
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        ms.save_checkpoint(model, arch.get('path', "path") + '/bert_weights{}_{}.ckpt'.format(arch.get('name', "name"),
                                                                                              epoch + 1))
        print('Saving checkpoint')
    print('cost time:', time.time() - start1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_epoch', metavar='N', type=int, default=100,
                        help='training epochs, default=150')
    parser.add_argument('--path', metavar='PATH', type=str, default='data/1chem.txt',
                        help='location of CATH dataset, default=data/1chem.txt')
    parser.add_argument('--device_id', help='device id', type=int, default=3)
    parser.add_argument('--device', help='device', type=str, default='GPU')
    parser.add_argument('--vocab_size', metavar='N', type=int, default=17,
                        help='vocab size, default=17')
    args = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device,
                   pynative_synchronize=True, device_id=args.device_id)
    main(args.trained_epoch, args.path, args.vocab_size)

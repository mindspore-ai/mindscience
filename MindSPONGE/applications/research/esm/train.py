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
"""Training of esm"""

import argparse
from argparse import Namespace
import json
import math
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from src import data
from src.util import CoordBatchConverter
from src.gvp_transformer import GVPTransformerModel


class DatasetGenerator:
    """自定义数据集类"""
    def __init__(self, path, alphabet):
        with open(path) as f:
            traindata = json.load(f)
        trainset = []
        for seq in traindata:
            trainset.append(self.mask(0.15, seq, p=0.05))
        batch = [(e["coords"], None, e["seq"]) for e in trainset]
        self.coords, self.confidence, self.padding_mask, self.prev_output_tokens, self.target\
            = self.data_generation(batch, alphabet)

    def __getitem__(self, item):
        output = [self.coords[item], self.confidence[item], self.padding_mask[item],
                  self.prev_output_tokens[item], self.target[item]]
        return output

    def __len__(self):
        return len(self.coords)

    @staticmethod
    def data_generation(batch, alphabet):
        """Data generation"""

        batch_converter = CoordBatchConverter(alphabet)
        coords, confidence, _, tokens, padding_mask = (
            batch_converter(batch)
        )
        prev_output_tokens = tokens[:, :-1]
        target = tokens[:, 1:]
        output = [coords.asnumpy(), confidence.asnumpy(), padding_mask.asnumpy(),
                  prev_output_tokens.asnumpy(), target.asnumpy()]
        return output

    @staticmethod
    def mask(mask_ratio, sentence, lower=1, upper=10, p=0.05):
        """Span masking"""

        sent_length = len(sentence['coords'])
        mask_num = math.ceil(sent_length * mask_ratio)
        mask = set()
        while len(mask) < mask_num:
            lens = list(range(lower, upper + 1))
            len_distrib = [p * (1 - p) ** (i - lower) for i in
                           range(lower, upper + 1)] if p >= 0 else None
            len_distrib = [x / (sum(len_distrib)) for x in len_distrib]
            span_len = np.random.choice(lens, p=len_distrib)
            anchor = np.random.choice(sent_length)
            if anchor in mask:
                continue
            for i in range(anchor, anchor + span_len):
                if len(mask) >= mask_num or i >= sent_length:
                    break
                mask.add(i)

        for num in mask:
            rand = np.random.random()
            if rand < 0.8:
                sentence['coords'][num - 1] = [[float('inf'), float('inf'), float('inf')],
                                               [float('inf'), float('inf'), float('inf')],
                                               [float('inf'), float('inf'), float('inf')]]
            elif rand < 0.9:
                # sample random token according to input distribution
                sentence['coords'][num - 1] = sentence['coords'][np.random.choice(sent_length)]
        return sentence


class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""
    def __init__(self, network, optimizer):
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ms.ops.GradOperation(get_by_list=True)

    def construct(self, inputs):
        """Train net construction"""
        loss = self.network((inputs['coords'], inputs['padding_mask'], inputs['confidence'],
                             inputs['prev_output_tokens']), label=inputs['target'])
        grads = \
            self.grad(self.network, self.weights)((inputs['coords'],
                                                   inputs['padding_mask'],
                                                   inputs['confidence'],
                                                   inputs['prev_output_tokens']),
                                                  inputs['target'])
        self.optimizer(grads)
        return loss


def main():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', metavar='N', type=int, default=150,
                        help='training epochs, default=150')
    parser.add_argument('--batch_size', metavar='N', type=int, default=2,
                        help='training epochs, default=1')
    parser.add_argument('--cath_data', metavar='PATH', default='data/chain_set.jsonl',
                        help='location of CATH dataset, default=data/chain_set.jsonl')
    parser.add_argument('--cath_splits', metavar='PATH', default='data/splits.json',
                        help='location of CATH split file, default=data/splits.json')
    parser.add_argument('--device_id', help='device id', type=int, default=2)
    args = parser.parse_args()
    # whether to ran with cuda
    ms.set_context(device_target='GPU', device_id=args.device_id)
    tf = open("src/args.json", "r")
    params = json.load(tf)
    params = Namespace(**params)
    alphabet = data.Alphabet.from_architecture(params.arch)
    train_dataset = ds.GeneratorDataset(DatasetGenerator('data/train_chain_set.jsonl', alphabet),
                                        ['coords', 'confidence', 'padding_mask', 'prev_output_tokens',
                                         'target'], shuffle=False)
    train_dataset = train_dataset.batch(batch_size=args.batch_size, drop_remainder=True)
    model = GVPTransformerModel(params, alphabet)
    loss = nn.CrossEntropyLoss()
    net_with_loss = nn.WithLossCell(model, loss)
    opt = nn.Adam(model.trainable_params())
    train_net = CustomTrainOneStepCell(net_with_loss, opt)
    train_net.set_train()
    for epoch in range(args.epochs):
        total_loss, total_count = 0, 0
        batch_epoch = 0
        for d in train_dataset.create_dict_iterator():
            d['target'] = ms.ops.Cast()(d['target'], ms.int32)
            result = train_net(d)
            coord_mask = ms.ops.IsFinite()(d['coords']).all(axis=-1).all(axis=-1)
            coord_mask = coord_mask[:, 1:-1]
            avgloss = \
                ms.ops.ReduceSum()(result * coord_mask) / ms.ops.ReduceSum()(ms.ops.Cast()(coord_mask, ms.float32))
            total_loss += float(avgloss)
            total_count += 1
            batch_epoch += 1
            print(f"Batch: {batch_epoch}, " f"loss: {avgloss}")
        print(f'EPOCH {epoch} TRAIN loss: {total_loss/total_count:.4f}')


if __name__ == '__main__':
    main()

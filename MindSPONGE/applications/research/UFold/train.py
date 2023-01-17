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
"""train UFold and get checkpoint files."""
import collections
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindspore.dataset import GeneratorDataset
from mindspore import context
from mindspore import dtype as mstype
from src.model import Unet as FCNNet
from src.config import process_config
from src.utils import get_args
from src.data_generator import RNASSDataGenerator
from src.data_generator import DatasetCutConcatNewMergeMulti as Dataset_FCN_merge


class MyWithLossCell(nn.Cell): # In order to use function 'TrainOneStepCell' to replace optimizer.step()
    def __init__(self, network, loss_fn):
        super(MyWithLossCell, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, x, y, label):
        out = self.network(x)
        return self.loss_fn(out*y, label)


def train(contact_net, train_merge_generator, epoches_first):
    """train model"""
    epoch = 0
    pos_weight = ms.Tensor([300], mstype.float32)
    criterion_bce_weighted = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight)
    u_optimizer = nn.Adam(params=contact_net.trainable_params(), learning_rate=1e-4)
    cast = ops.Cast()
    zeroslike = ops.ZerosLike()
    loss_net = MyWithLossCell(contact_net, criterion_bce_weighted)
    train_net = nn.TrainOneStepCell(loss_net, u_optimizer)

    print('start training...')

    for epoch in range(epoches_first):
        train_net.set_train()
        for contacts, seq_embeddings, _, seq_lens, _, _ in train_merge_generator:
            contacts_batch = ms.Tensor(cast(contacts, mstype.float32))
            seq_embedding_batch = ms.Tensor(cast(seq_embeddings, mstype.float32))

            pred_contacts = contact_net(seq_embedding_batch)
            contact_masks = zeroslike(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
            loss_ms = train_net(seq_embedding_batch, contact_masks, contacts_batch)

        print('Training log: epoch: {}, loss: {}.'.format(epoch, loss_ms))
        if epoch > -1:
            model_saved = f'./ckpt_models/ufold_train_{epoch}.ckpt'
            ms.save_checkpoint(train_net, model_saved)
            param_dict = ms.load_checkpoint(model_saved)
            new_params_list = []
            for name in param_dict:
                param_dict_new = {}
                parameter = param_dict[name]
                if name.startswith('network.'):
                    name = name[8:]
                param_dict_new['name'] = name
                param_dict_new['data'] = ms.Tensor(parameter.asnumpy(), mstype.float32)
                new_params_list.append(param_dict_new)
            ms.save_checkpoint(new_params_list, model_saved)
            param_dict = ms.load_checkpoint(model_saved)
            print("model_saved!")


def main():
    args = get_args()

    config_file = args.config

    config = process_config(config_file)
    print('Here is the configuration of this run: ')
    print(config)

    batch_size_1 = config.batch_size_stage_1
    epoches_first = config.epoches_first
    train_files = args.train_files

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    train_data_list = []
    for file_item in train_files:
        print('Loading dataset: ', file_item)
        if file_item == 'ArchiveII':
            train_data_list.append(RNASSDataGenerator('data/', file_item+'.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('data/', file_item+'.cPickle'))
    print('Data Loading Done!!!')

    ms.set_seed(1)
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = GeneratorDataset(train_merge,
                                             column_names=['contacts', 'seq_embeddings',
                                                           'matrix_reps', 'seq_lens', 'seq_ori', 'seq_name'],
                                             num_parallel_workers=16,
                                             shuffle=True).batch(batch_size=batch_size_1, drop_remainder=True)

    contact_net = FCNNet(img_ch=17)

    train(contact_net, train_merge_generator, epoches_first)

if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    main()

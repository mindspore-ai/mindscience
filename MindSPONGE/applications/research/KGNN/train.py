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
"""GCN training script."""
import os
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from src.dataset import create_dataset
from src.kgcn import KGCN
from src.loss import BinaryCrossEntropyLoss
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.utils import format_filename, download_qizhi, pickle_load, upload_qizhi
from src.data_loader import ENTITY_VOCAB_TEMPLATE, RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, \
    ADJ_RELATION_TEMPLATE, DRUG_VOCAB_TEMPLATE, DRUG_EXAMPLE, process_data

if config.device_target == 'Ascend':
    device_id = get_device_id()
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                        device_id=device_id)
else:
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
mindspore.set_seed(1)

if config.run_distribute:
    init()
    if config.device_target in ['Ascend', 'GPU']:
        RANK_ID = get_device_id()
        RANK_SIZE = get_device_num()
    else:
        RANK_ID = get_rank()
        RANK_SIZE = get_group_size()
    parallel_mode = ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                      device_num=RANK_SIZE,
                                      gradients_mean=True)
else:
    RANK_ID = 0
    RANK_SIZE = 1


def get_optimizer(params, lr):
    """get_optimizer"""
    if config.optimizer == 'sgd':
        return nn.SGD(params=params, learning_rate=lr, weight_decay=config.l2_weight)
    if config.optimizer == 'rmsprop':
        return nn.RMSProp(params=params, learning_rate=lr, weight_decay=config.l2_weight)
    if config.optimizer == 'adagrad':
        return nn.Adagrad(params=params, learning_rate=lr, weight_decay=config.l2_weight)
    if config.optimizer == 'adadelta':
        return nn.Adadelta(params=params, learning_rate=lr, weight_decay=config.l2_weight)
    if config.optimizer == 'adam':
        return nn.Adam(params=params, learning_rate=lr, weight_decay=config.l2_weight)
    raise ValueError('Optimizer Not Understood: {}'.format(config.optimizer))


def train_net():
    """train_net"""
    train_data = np.load(format_filename(config.PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=config.dataset, type="train"))
    train_dataset = create_dataset(train_data, config.batch_size, rank_size=RANK_SIZE, rank_id=RANK_ID)
    train_data_size = train_dataset.get_dataset_size()
    network = KGCN()

    loss = BinaryCrossEntropyLoss()
    optimizer = get_optimizer(network.trainable_params(), config.lr)
    network.set_train()

    model = Model(network, loss_fn=loss, optimizer=optimizer)

    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(RANK_ID) + '/')
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix="KGNN", directory=save_ckpt_path, config=ckpt_config)

    callbacks_list = [loss_cb, time_cb, ckpt_cb]
    print(f"============== Starting Training aggregator_type: {config.aggregator_type} ==============")
    model.train(config.n_epoch, train_dataset, callbacks=callbacks_list, dataset_sink_mode=False)
    print("============== End Training ==============")


if __name__ == '__main__':
    if config.enable_modelarts:
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        if not os.path.exists(config.train_dir):
            os.makedirs(config.train_dir)
        # Initialize and copy data
        download_qizhi(config.data_url, config.data_dir)
        config.save_checkpoint_path = config.train_dir
        config.RAW_DATA_DIR = config.data_dir

    NEIGHBOR_SIZE = {'drug': 4, 'kegg': 4}
    if not os.path.isdir(config.PROCESSED_DATA_DIR):
        os.makedirs(config.PROCESSED_DATA_DIR)
    config.KG_FILE = os.path.join(config.RAW_DATA_DIR, config.dataset, 'train2id.txt')
    config.ENTITY2ID_FILE = os.path.join(config.RAW_DATA_DIR, config.dataset, 'entity2id.txt')
    config.EXAMPLE_FILE = os.path.join(config.RAW_DATA_DIR, config.dataset, 'approved_example.txt')
    process_data(config.dataset, NEIGHBOR_SIZE.get(config.dataset))

    config.drug_vocab_size = len(pickle_load(format_filename(config.PROCESSED_DATA_DIR,
                                                             DRUG_VOCAB_TEMPLATE,
                                                             dataset=config.dataset)))
    config.entity_vocab_size = len(pickle_load(format_filename(config.PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=config.dataset)))
    config.relation_vocab_size = len(pickle_load(format_filename(config.PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=config.dataset)))
    config.adj_entity = np.load(format_filename(config.PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=config.dataset))
    config.adj_relation = np.load(format_filename(config.PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                  dataset=config.dataset))

    aggregator_types = ['sum', 'concat', 'neigh']
    for t in aggregator_types:
        config.aggregator_type = t
        train_net()
    if config.enable_modelarts:
        upload_qizhi(config.train_dir, config.train_url)

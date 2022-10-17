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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import os
import numpy as np

import mindspore as ms
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint
from src.kgcn import KGCN
from src.model_utils.config import config
from src.utils import pickle_load, format_filename
from src.dataset import create_dataset
from src.data_loader import ENTITY_VOCAB_TEMPLATE, RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, \
    ADJ_RELATION_TEMPLATE, DRUG_VOCAB_TEMPLATE

DRUG_EXAMPLE = '{dataset}_{type}_examples.npy'


def run_export():
    """run_export"""
    devid = int(os.getenv('DEVICE_ID', '0'))
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=devid)

    network = KGCN()
    load_checkpoint(config.checkpoint_path, network)
    network.set_train(False)
    test_data = np.load(format_filename(config.PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=config.dataset, type="test"))
    test_dataset = create_dataset(test_data, 1, rank_size=1, rank_id=0, is_training=False)

    shape = [test_dataset.get_dataset_size(), 2]
    inputs = Tensor(np.zeros(shape), ms.int64)

    export(network, inputs, file_name=config.file_name, file_format=config.file_format)


if __name__ == "__main__":
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
    run_export()

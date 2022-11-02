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
Evaluation script
"""
import os

import numpy as np
import sklearn.metrics as m
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
from mindspore import Tensor, Model, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.kgcn import KGCN
from src.model_utils.config import config
from src.utils import pickle_load, format_filename, obs_env, obs_url_env, env_obs
from src.data_loader import ENTITY_VOCAB_TEMPLATE, RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, \
    ADJ_RELATION_TEMPLATE, DRUG_VOCAB_TEMPLATE, DRUG_EXAMPLE, process_data

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=device_id)


def eval_net():
    """
    run eval
    """
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

    # Create network
    test_data = np.load(format_filename(config.PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=config.dataset, type="test"))
    test_dataset = create_dataset(test_data, config.batch_size, rank_size=1, rank_id=0)
    test_data_size = test_dataset.get_dataset_size()
    network = KGCN()
    network.set_train(False)
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(network, param_dict)
    model = Model(network)
    print('Logging Info - Evaluate over test data:')
    threshold = 0.5
    auc = 0.0
    aupr = 0.0
    acc = 0.0
    f1 = 0.0
    for _, data in enumerate(test_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        y_true = data["label"]
        test_data = Tensor(data["data"])
        y_pred = model.predict(test_data)
        auc_m = roc_auc_score(y_true=y_true, y_score=y_pred)
        p, r, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr_m = m.auc(r, p)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc_m = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1_m = f1_score(y_true=y_true, y_pred=y_pred)
        auc += auc_m
        acc += acc_m
        f1 += f1_m
        aupr += aupr_m
    auc /= test_data_size
    acc /= test_data_size
    f1 /= test_data_size
    aupr /= test_data_size
    print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}')


if __name__ == "__main__":
    if config.enable_modelarts:

        # Initialize the data and result directories in the inference image
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)

        # Copy dataset from obs to inference image
        obs_env(config.data_url, config.data_dir)
        # Copy ckpt file from obs to inference image
        obs_url_env(config.ckpt_url, config.ckpt_dir)

        config.RAW_DATA_DIR = config.data_dir
        config.checkpoint_path = config.ckpt_dir
        config.acclog_path = os.path.join(config.result_dir, config.acclog_path)

        NEIGHBOR_SIZE = {'drug': 4, 'kegg': 4}
        if not os.path.isdir(config.PROCESSED_DATA_DIR):
            os.makedirs(config.PROCESSED_DATA_DIR)
        config.KG_FILE = os.path.join(config.RAW_DATA_DIR, config.dataset, 'train2id.txt')
        config.ENTITY2ID_FILE = os.path.join(config.RAW_DATA_DIR, config.dataset, 'entity2id.txt')
        config.EXAMPLE_FILE = os.path.join(config.RAW_DATA_DIR, config.dataset, 'approved_example.txt')
        process_data(config.dataset, NEIGHBOR_SIZE.get(config.dataset))

    eval_net()

    # Copy result data from the local running environment back to obs,
    # and download it in the inference task corresponding to the Qizhi platform
    if config.enable_modelarts:
        env_obs(config.result_dir, config.result_url)

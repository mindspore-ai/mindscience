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
"""predictor file"""

import os
import time
import logging
import pickle
import yaml
import numpy as np
import mindspore as ms
from mindspore import set_seed
from data.generate import get_prop_model
from models.matformer import Matformer
from models.utils import LossRecord
from mindchemistry.graph.loss import L1LossMask, L2LossMask
from mindchemistry.graph.dataloader import DataLoaderBase as DataLoader

logging.basicConfig(level=logging.INFO)

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

ms.set_context(device_target=config['train']["device"], device_id=config['train']["device_id"])

dataset_dir = config['train']["dataset_dir"]
ckpt_dir = config['train']["ckpt_dir"]

checkpoint_dir = config['predictor']['checkpoint_path']

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

set_seed(config['train']["seed"])

get_prop_model(prop=config['train']["props"], use_lattice=True, dataset_path=config["dataset"])

BATCH_SIZE_MAX = config['train']['batch_size']

matformer = Matformer(config['model'])

model_parameters = filter(lambda p: p.requires_grad, matformer.get_parameters())
params = sum(np.prod(p.shape) for p in model_parameters)
logging.info("The model you built has %s parameters.", params)

loss_func_mse = L2LossMask(reduction='mean')
loss_func_mae = L1LossMask(reduction='mean')


def forward_eval(data_x, data_edge_attr, data_edge_index, data_batch, data_label, node_mask, edge_mask, node_num,
                 batch_size):
    """forward eval"""
    pred = matformer(data_x, data_edge_attr, data_edge_index, data_batch, node_mask, edge_mask, node_num)

    mseloss = loss_func_mse(pred, data_label, num=batch_size)
    maeloss = loss_func_mae(pred, data_label, num=batch_size)

    return mseloss, maeloss, pred


@ms.jit
def eval_step(data_x, data_edge_attr, data_edge_index, data_batch, data_label, node_mask, edge_mask, node_num,
              batch_size):
    """eval step"""
    mseloss, maeloss, pred = forward_eval(data_x, data_edge_attr, data_edge_index, data_batch, data_label,
                                          node_mask, edge_mask, node_num, batch_size)
    return mseloss, maeloss, pred


################load from pickle file
with open(config['dataset']['x_val_path'], 'rb') as f:
    x_val = pickle.load(f)
with open(config['dataset']['edge_index_val_path'], 'rb') as f:
    edge_index_val = pickle.load(f)
with open(config['dataset']['edge_attr_val_path'], 'rb') as f:
    edge_attr_val = pickle.load(f)
with open(config['dataset']['label_val_path'], 'rb') as f:
    label_val = pickle.load(f)

if os.path.exists(checkpoint_dir):
    logging.info("load checkpoint from specified path................")
    param_dict = ms.load_checkpoint(checkpoint_dir)
    epoch = int(param_dict["epoch"]) + 1
    param_not_load, _ = ms.load_param_into_net(matformer, param_dict)
    logging.info("finish loading checkpoint from specified path, start evaluating from epoch: %s", str(epoch))

    current_step = int(param_dict["current_step"])
else:
    raise FileNotFoundError

EPOCH = 0
EPOCH_SIZE = config['predictor']["epoch_size"]
BEST_EPOCH_EVAL_MSE_LOSS = 10000

logging.info("Start to initialise eval_loader")
eval_loader = DataLoader(BATCH_SIZE_MAX, edge_index_val, node_attr=x_val, edge_attr=edge_attr_val, label=label_val,
                         dynamic_batch_size=False, shuffle_dataset=False)

while EPOCH < EPOCH_SIZE:
    epoch_starttime = time.time()

    eval_mseloss_record = LossRecord()
    eval_maeloss_record = LossRecord()

    #################################################### validation #####################################################
    logging.info("+++++++++++++++ start validation +++++++++++++++++++++")
    matformer.set_train(False)

    starttime = time.time()
    for node_attr_step, edge_attr_step, label_step, edge_index_step, node_batch_step, \
            node_mask_step, edge_mask_step, node_num_step, batch_size_step in eval_loader:
        logging.info("==============================step: %s ,epoch: %s", eval_loader.step - 1, EPOCH)

        mseloss_step, maeloss_step, _ = eval_step(node_attr_step, edge_attr_step, edge_index_step, node_batch_step,
                                                  label_step, node_mask_step, edge_mask_step, node_num_step,
                                                  batch_size_step)

        logging.info("validation mse loss: %s", mseloss_step)
        logging.info("validation mae loss: %s", maeloss_step)

        eval_mseloss_record.update(mseloss_step)
        eval_maeloss_record.update(maeloss_step)
        starttime0 = starttime
        starttime = time.time()
        logging.info("validation time: %s", starttime - starttime0)
    #################################################### finish validation #####################################################

    epoch_endtime = time.time()

    logging.info("epoch %s running time: %s", EPOCH, epoch_endtime - epoch_starttime)
    logging.info("epoch %s average validation mse loss: %s", EPOCH, eval_mseloss_record.avg)
    logging.info("epoch %s average validation mae loss: %s", EPOCH, eval_maeloss_record.avg)

    EPOCH = EPOCH + 1

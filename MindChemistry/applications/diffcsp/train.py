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
"""train file"""
import os
import time
import logging
import argparse
import yaml
import numpy as np
import mindspore as ms
from mindspore import nn, set_seed
from mindspore.amp import all_finite
from mindchemistry.graph.loss import L2LossMask
from models.cspnet import CSPNet
from models.diffusion import CSPDiffusion
from models.train_utils import LossRecord
from data.dataset import fullconnect_dataset
from data.crysloader import Crysloader as DataLoader

logging.basicConfig(level=logging.INFO)

def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help="The config file path")
    parser.add_argument('--device_id', type=int, default=0,
                        help="ID of the target device")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    input_args = parser.parse_args()
    return input_args

if __name__ == '__main__':
    args = parse_args()
    ms.set_context(device_target=args.device_target, device_id=args.device_id)

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    ckpt_dir = config['train']["ckpt_dir"]

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    set_seed(config['train']["seed"])

    batch_size_max = config['train']['batch_size']

    cspnet = CSPNet(num_layers=config['model']['num_layers'], hidden_dim=config['model']['hidden_dim'],
                    num_freqs=config['model']['num_freqs'])

    if os.path.exists(config['checkpoint']['last_path']):
        logging.info("load from existing check point................")
        param_dict = ms.load_checkpoint(config['checkpoint']['last_path'])
        ms.load_param_into_net(cspnet, param_dict)
        logging.info("finish load from existing checkpoint")
    else:
        logging.info("Starting new training process")

    diffcsp = CSPDiffusion(cspnet)

    model_parameters = filter(lambda p: p.requires_grad, diffcsp.get_parameters())
    params = sum(np.prod(p.shape) for p in model_parameters)
    logging.info("The model you built has %s parameters.", params)

    optimizer = nn.Adam(params=diffcsp.trainable_params())
    loss_func_mse = L2LossMask(reduction='mean')

    def forward(atom_types_step, frac_coords_step, _, lengths_step, angles_step, edge_index_step, batch_node2graph, \
                node_mask_step, edge_mask_step, batch_mask, node_num_valid, batch_size_valid):
        """forward"""
        pred_l, rand_l, pred_x, tar_x = diffcsp(batch_size_valid, atom_types_step, lengths_step,
                                                angles_step, frac_coords_step, batch_node2graph, edge_index_step,
                                                node_mask_step, edge_mask_step, batch_mask)
        mseloss_l = loss_func_mse(pred_l, rand_l, mask=batch_mask, num=batch_size_valid)
        mseloss_x = loss_func_mse(pred_x, tar_x, mask=node_mask_step, num=node_num_valid)
        mseloss = mseloss_l + mseloss_x

        return mseloss, mseloss_l, mseloss_x

    backward = ms.value_and_grad(forward, None, weights=diffcsp.trainable_params(), has_aux=True)

    @ms.jit
    def train_step(atom_types_step, frac_coords_step, property_step, lengths_step, angles_step,
                   edge_index_step, batch_node2graph, node_mask_step, edge_mask_step, batch_mask,
                   node_num_valid, batch_size_valid):
        """train step"""
        (mseloss, mseloss_l, mseloss_x), grads = backward(atom_types_step, frac_coords_step, property_step,
                                                          lengths_step, angles_step, edge_index_step, batch_node2graph,
                                                          node_mask_step, edge_mask_step, batch_mask, node_num_valid,
                                                          batch_size_valid)

        is_finite = all_finite(grads)
        if is_finite:
            optimizer(grads)

        return mseloss, is_finite, mseloss_l, mseloss_x

    @ms.jit
    def eval_step(atom_types_step, frac_coords_step, property_step, lengths_step, angles_step,
                  edge_index_step, batch_node2graph,
                  node_mask_step, edge_mask_step, batch_mask, node_num_valid, batch_size_valid):
        """eval step"""
        mseloss, mseloss_l, mseloss_x = forward(atom_types_step, frac_coords_step, property_step, lengths_step,
                                                angles_step, edge_index_step, batch_node2graph,
                                                node_mask_step, edge_mask_step, batch_mask, node_num_valid,
                                                batch_size_valid)
        return mseloss, mseloss_l, mseloss_x

    epoch = 0
    epoch_size = config['train']["epoch_size"]

    logging.info("Start to initialise train_loader")
    train_datatset = fullconnect_dataset(name=config['dataset']["data_name"], path=config['dataset']["train"]["path"],
                                         save_path=config['dataset']["train"]["save_path"])
    train_loader = DataLoader(batch_size_max, *train_datatset, shuffle_dataset=True)
    logging.info("Start to initialise eval_loader")
    val_datatset = fullconnect_dataset(name=config['dataset']["data_name"], path=config['dataset']["val"]["path"],
                                       save_path=config['dataset']["val"]["save_path"])
    eval_loader = DataLoader(batch_size_max, *val_datatset,
                             dynamic_batch_size=False, shuffle_dataset=True)

    while epoch < epoch_size:
        epoch_starttime = time.time()

        train_mseloss_record = LossRecord()
        eval_mseloss_record = LossRecord()

        #################################################### train ###################################################
        logging.info("+++++++++++++++ start traning +++++++++++++++++++++")
        diffcsp.set_train(True)

        starttime = time.time()
        record_iter = 0
        for atom_types_batch, frac_coords_batch, property_batch, lengths_batch, angles_batch,\
            edge_index_batch, batch_node2graph_, node_mask_batch, edge_mask_batch, batch_mask_batch,\
                node_num_valid_, batch_size_valid_ in train_loader:

            result = train_step(atom_types_batch, frac_coords_batch, property_batch,
                                lengths_batch, angles_batch, edge_index_batch, batch_node2graph_,
                                node_mask_batch, edge_mask_batch, batch_mask_batch, node_num_valid_,
                                batch_size_valid_)

            mseloss_step, _, mseloss_l_, mseloss_x_ = result

            if record_iter % 50 == 0:
                logging.info("==============================step: %s ,epoch: %s", train_loader.step - 1, epoch)
                logging.info("learning rate: %s", optimizer.learning_rate.value())
                logging.info("train mse loss: %s", mseloss_step)
                logging.info("train mse_lattice loss: %s", mseloss_l_)
                logging.info("train mse_coords loss: %s", mseloss_x_)
                starttime0 = starttime
                starttime = time.time()
                logging.info("traning time: %s", starttime - starttime0)

            record_iter += 1

            train_mseloss_record.update(mseloss_step)

        #################################################### finish train ########################################
        epoch_endtime = time.time()
        logging.info("epoch %s running time: %s", epoch, epoch_endtime - epoch_starttime)
        logging.info("epoch %s average train mse loss: %s", epoch, train_mseloss_record.avg)

        ms.save_checkpoint(diffcsp.decoder, config['checkpoint']['last_path'])

        if epoch % 5 == 0:
            #################################################### validation ##########################################
            logging.info("+++++++++++++++ start validation +++++++++++++++++++++")
            diffcsp.set_train(False)

            starttime = time.time()
            for atom_types_batch, frac_coords_batch, property_batch, lengths_batch, angles_batch,\
                edge_index_batch, batch_node2graph_, node_mask_batch, edge_mask_batch, batch_mask_batch,\
                    node_num_valid_, batch_size_valid_ in eval_loader:

                result_e = eval_step(atom_types_batch, frac_coords_batch, property_batch,
                                     lengths_batch, angles_batch, edge_index_batch, batch_node2graph_,
                                     node_mask_batch, edge_mask_batch, batch_mask_batch, node_num_valid_,
                                     batch_size_valid_)

                mseloss_step, mseloss_l_, mseloss_x_ = result_e

                eval_mseloss_record.update(mseloss_step)

            #################################################### finish validation #################################

            starttime0 = starttime
            starttime = time.time()
            logging.info("validation time: %s", starttime - starttime0)
            logging.info("epoch %s average validation mse loss: %s", epoch, eval_mseloss_record.avg)

        epoch = epoch + 1

# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Train
"""

import os
import logging
import argparse
import time
import numpy as np
import mindspore as ms
from mindspore.experimental import optim
from mindchemistry.utils.load_config import load_yaml_config_from_path
from mindchemistry.cell.cdvae import CDVAE
from mindchemistry.cell.gemnet.data_utils import StandardScalerMindspore
from create_dataset import create_dataset
from src.dataloader import DataLoaderBaseCDVAE


def train_epoch(epoch, model, optimizer, scheduler, train_dataset):
    """Train the model for one epoch"""
    model.set_train()
    # Define forward function

    def forward_fn(data):
        (atom_types, dist, _, idx_kj, idx_ji,
         edge_j, edge_i, batch, lengths, num_atoms,
         angles, frac_coords, y, batch_size, sbf, total_atoms) = data
        loss = model(atom_types, dist, idx_kj, idx_ji, edge_j, edge_i,
                     batch, lengths, num_atoms, angles, frac_coords,
                     y, batch_size, sbf, total_atoms, True, True)
        return loss
    # Get gradient function
    grad_fn = ms.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    # Define function of one-step training
    def train_step(data):
        loss, grads = grad_fn(data)
        scheduler.step(loss)
        optimizer(grads)
        return loss

    start_time_step = time.time()
    for batch, data in enumerate(train_dataset):
        loss = train_step(data)
        time_step = time.time() - start_time_step
        start_time_step = time.time()
        if batch % 10 == 0:
            logging.info("Train Epoch: %d [%d]\tLoss: %4f,\t time_step: %4f",
                         epoch, batch, loss, time_step)


def test_epoch(model, val_dataset):
    """test for one epoch"""
    model.set_train(False)
    test_loss = 0
    i = 1
    for i, data in enumerate(val_dataset):
        (atom_types, dist, _, idx_kj, idx_ji,
         edge_j, edge_i, batch, lengths, num_atoms,
         angles, frac_coords, y, batch_size, sbf, total_atoms) = data
        output = model(atom_types, dist,
                       idx_kj, idx_ji, edge_j, edge_i,
                       batch, lengths, num_atoms,
                       angles, frac_coords, y, batch_size,
                       sbf, total_atoms, False, True)
        test_loss += float(output)
    test_loss /= (i+1)
    logging.info("Val Loss: %4f", test_loss)
    return test_loss

def get_scaler(args):
    """get scaler"""
    lattice_scaler_mean = ms.Tensor(np.loadtxt(
        f"./data/{args.dataset}/train/lattice_scaler_mean.csv"), ms.float32)
    lattice_scaler_std = ms.Tensor(np.loadtxt(
        f"./data/{args.dataset}/train/lattice_scaler_std.csv"), ms.float32)
    scaler_std = ms.Tensor(np.loadtxt(
        f"./data/{args.dataset}/train/scaler_std.csv"), ms.float32)
    scaler_mean = ms.Tensor(np.loadtxt(
        f"./data/{args.dataset}/train/scaler_mean.csv"), ms.float32)
    lattice_scaler = StandardScalerMindspore(
        lattice_scaler_mean, lattice_scaler_std)
    scaler = StandardScalerMindspore(scaler_mean, scaler_std)
    return lattice_scaler, scaler

def train_net(args):
    """training process"""
    folder_path = os.path.dirname(args.name_ckpt)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info("%s has been created", folder_path)
    config_path = "./conf/configs.yaml"
    data_config_path = f"./conf/data/{args.dataset}.yaml"

    model = CDVAE(config_path, data_config_path)

    # load checkpoint
    if args.load_ckpt:
        model_path = args.name_ckpt
        param_dict = ms.load_checkpoint(model_path)
        param_not_load, _ = ms.load_param_into_net(model, param_dict)
        logging.info("%s have not been loaded", param_not_load)

    # create dataset when running the model first-time or when dataset is not exist
    if args.create_dataset or not os.path.exists(f"./data/{args.dataset}/train/processed_data.npy"):
        logging.info("Creating dataset......")
        create_dataset(args) # dataset created will be save to the dir based on args.dataset as npy

    # read dataset from processed_data
    batch_size = load_yaml_config_from_path(data_config_path).get("batch_size")
    train_dataset = DataLoaderBaseCDVAE(
        batch_size, args.dataset, shuffle_dataset=True, mode="train")
    val_dataset = DataLoaderBaseCDVAE(
        batch_size, args.dataset, shuffle_dataset=False, mode="val")
    lattice_scaler, scaler = get_scaler(args)
    model.lattice_scaler = lattice_scaler
    model.scaler = scaler

    config_opt = load_yaml_config_from_path(config_path).get("Optimizer")
    learning_rate = config_opt.get("learning_rate")
    min_lr = config_opt.get("min_lr")
    factor = config_opt.get("factor")
    patience = config_opt.get("patience")

    optimizer = optim.Adam(model.trainable_params(), learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=factor, patience=patience, min_lr=min_lr)

    min_test_loss = float("inf")
    for epoch in range(args.epoch_num):
        train_epoch(epoch, model, optimizer, scheduler, train_dataset)
        if epoch % 10 == 0:
            test_loss = test_epoch(model, val_dataset)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                ms.save_checkpoint(model, args.name_ckpt)
                logging.info("Updata best acc: %f", test_loss)

    logging.info('Finished Training')

def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="perov_5", help="dataset name")
    parser.add_argument("--create_dataset", default=False,
                        type=bool, help="whether create dataset again or not")
    parser.add_argument("--num_samples_train", default=500, type=int,
                        help="number of samples for training,\
                        only valid when create_dataset is True")
    parser.add_argument("--num_samples_val", default=300, type=int,
                        help="number of samples for validation,\
                        only valid when create_dataset is True")
    parser.add_argument("--num_samples_test", default=300, type=int,
                        help="number of samples for test,\
                        only valid when create_dataset is True")
    parser.add_argument("--name_ckpt", default="./loss/loss.ckpt",
                        help="the path to save checkpoint")
    parser.add_argument("--load_ckpt", default=False, type=bool,
                        help="whether load checkpoint or not")
    parser.add_argument("--device_target", default="Ascend", help="device target")
    parser.add_argument("--device_id", default=3, type=int, help="device id")
    parser.add_argument("--epoch_num", default=100, type=int, help="number of epoch")
    return parser.parse_args()

if __name__ == "__main__":
    main_args = get_args()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    ms.context.set_context(device_target=main_args.device_target,
                           device_id=main_args.device_id,
                           mode=1)
    train_net(main_args)

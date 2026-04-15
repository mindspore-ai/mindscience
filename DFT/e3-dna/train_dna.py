# Copyright 2021 Huawei Technologies Co., Ltd
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
"""main"""
import os
import time
import logging
import argparse

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, context, Tensor, save_checkpoint, set_seed
from mindchemistry.e3 import o3

from src.models import Network
from src.data import get_iso_permuted_dataset
from src.utils import load_cfg, get_scalar_density_comparisons, collate_list_of_dicts


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Train e3 network for DNA", fromfile_prefix_chars="+"
    )
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config file")
    return load_cfg(parser.parse_args().config)

class ms_train():
    def __init__(self, loss_fn, Rs, b):
        self.loss_fn = loss_fn
        self.Rs = Rs
        self.b = b
        self.minMAE = np.inf
        self.minMUE = np.inf
        self.minloss = np.inf
    
    def train_epoch(self, epoch, model, optimizer, data_loader):
        model.set_train()
        loss_cum      = 0.0
        mae_cum       = 0.0
        mue_cum       = 0.0
        load_time     = 0
        train_time    = 0
        post_time     = 0
        train_num_ele = []

        # Define forward function
        def forward_fn(data, label):
            mask = ops.select(label == 0, ops.zeros_like(label), ops.ones_like(label))
            batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded = data
            logits = model(batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)*mask
            loss = self.loss_fn(logits, label)
            return loss, logits

        # Get gradient function
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(data, label):
            (loss, logits), grads = grad_fn(data, label)
            optimizer(grads)
            return loss, logits

        dataset_size = data_loader.get_dataset_size()
        start_time = time.time()
        for batch_idx, trainset in enumerate(data_loader.create_dict_iterator()):
            data = trainset['data']
            target = data['y']
            batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded = model.preprocess(data)
            load_time += time.time()-start_time
            start_time = time.time()
            
            loss, logits = train_step((batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded), target)
            train_time += time.time()-start_time
            start_time = time.time()
            ms.ms_memory_recycle()

            for mul, l in self.Rs:
                if l == 0:
                    num_ele = ops.sum(logits[:, :mul]).asnumpy().item()

            train_num_ele.append(num_ele)
            mue_cum  += num_ele
            mae_cum  += np.abs(num_ele)
            loss_cum += abs(loss.asnumpy())
            post_time += time.time()-start_time
            start_time = time.time()

        train_tot = len(train_num_ele)
        train_stdev = np.std(train_num_ele)
        
        logging.info("Epoch: %d - Train loss: %s", epoch, float(loss_cum)/len(data_loader))
        logging.info("      MAE:         %g", mae_cum/(len(data_loader)*self.b))
        logging.info("      MUE:         %g", mue_cum/(len(data_loader)*self.b))
        logging.info("      Train STDEV: %g", train_stdev)
        logging.info("      Train tot:   %g", train_tot)
        logging.info("      Load time: %s s, Train time: %s s, Post time: %s s", load_time, train_time, post_time)
        return loss_cum, mae_cum, mue_cum, train_num_ele


    def test_epoch(self, epoch, model, data_loader, is_train=True):
        model.set_train(False)
        test_loss_cum = 0.0
        test_mae_cum = 0.0
        test_mue_cum = 0.0
        bigIs_cum = 0.0
        eps_cum = 0.0
        ele_diff_cum = 0.0
        metrics = []
        test_num_ele = []

        dataset_size = data_loader.get_dataset_size()
        start_time = time.time()
        for batch_idx, testset in enumerate(data_loader.create_dict_iterator()):
            data = testset["data"]
            label = data['y']
            batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded = model.preprocess(data)
            mask = ops.select(label == 0, ops.zeros_like(label), ops.ones_like(label))
            logits = model(batch, x, z, edge_src, edge_dst, edge_attr, edge_length_embedded) * mask
            err = self.loss_fn(logits, label).asnumpy().item()
            ms.ms_memory_recycle()

            for mul, l in self.Rs:
                if l == 0:
                    num_ele = ops.sum(logits[:, :mul]).asnumpy().item()

            test_num_ele.append(num_ele)
            test_mue_cum  += num_ele
            test_mae_cum  += abs(num_ele)
            test_loss_cum += abs(err)
            metrics.append([test_loss_cum, test_mae_cum, test_mue_cum])

            if epoch%10 == 0:
                num_ele_target, num_ele_ml, bigI, ep = get_scalar_density_comparisons(data, logits, self.Rs, spacing=0.2, buffer=4.0)
                n_ele = np.sum(data['z'].asnumpy())
                ele_diff_cum += np.abs(n_ele-num_ele_target)
                bigIs_cum += bigI
                eps_cum += ep
 
        test_stdev = np.std(test_num_ele)
        logging.info("      Test Loss:   %g", float(metrics[0][0])/len(data_loader))
        logging.info("      Test MAE:    %g", metrics[0][1]/len(data_loader))
        logging.info("      Test MUE:    %g", metrics[0][2]/len(data_loader))
        logging.info("      Test STDEV:  %g", test_stdev)
        logging.info("      Test time:   %g", time.time()-start_time)
        if epoch%10 == 0:
            logging.info("      Test electron difference: %g", ele_diff_cum/len(data_loader))
            logging.info("      Test big I:               %g", bigIs_cum/len(data_loader))
            logging.info("      Test epsilon:             %g", eps_cum/len(data_loader))

        if is_train and float(metrics[0][0])/len(data_loader) < self.minloss:
            self.minloss = float(metrics[0][0])/len(data_loader)
            save_checkpoint(model, os.path.join("./checkpoints/best_model.ckpt"))
            logging.info("      Model saved with min loss at epoch %d", epoch)

        # if metrics[0][1]/len(data_loader) < self.minMAE:
        #     self.minMAE = metrics[0][1]/len(data_loader)
        #     save_checkpoint(model, os.path.join("./checkpoints/best_MAE.ckpt"))
        #     logging.info("      Model saved with min MAE at epoch %d", epoch)

        # if metrics[0][2]/len(data_loader) < self.minMUE:
        #     self.minMUE = metrics[0][2]/len(data_loader)
        #     save_checkpoint(model, os.path.join("./checkpoints/best_MUE.ckpt"))
        #     logging.info("      Model saved with min MUE at epoch %d", epoch)
    
        return metrics, test_stdev, ele_diff_cum, bigIs_cum, eps_cum
    
def data_generator(data):
    for d in data:
        yield d

def count_parameters(model):
    return sum(p.size for p in model.trainable_params() if p.requires_grad)


def main():
    args = get_arguments()

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "printlog.txt"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device, device_id=args.device_id, max_call_depth=3000)

    # first, get dataset
    hhh = "./data/h_s_only_augccpvdz_density.out"
    ooo = "./data/o_s_only_augccpvdz_density.out"
    ccc = "./data/c_s_only_augccpvdz_density.out"
    nnn = "./data/n_s_only_augccpvdz_density.out"
    ppp = "./data/p_s_only_augccpvdz_density.out"

    # second, check if num_ele is correct
    # def2 basis set max irreps
    Rs = [(14, 0), (5, 1), (5, 2), (2, 3), (1, 4)]

    checkpoint_dir = args.ckpt_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    b = args.batch
    logging.info("Batch size: %d", b)
    train_split = args.split
    num_epochs = args.epochs

    train_datafile = args.datapath
    test_datafile = args.testpath

    train_dataset = get_iso_permuted_dataset(train_datafile, h_iso=hhh, c_iso=ccc, n_iso=nnn, o_iso=ooo, p_iso=ppp)[:train_split[0]]
    train_loader = ds.GeneratorDataset(lambda: data_generator(train_dataset), ["data"], shuffle=True)
    train_loader = train_loader.batch(b, per_batch_map=collate_list_of_dicts)

    test_dataset = get_iso_permuted_dataset(test_datafile, h_iso=hhh, c_iso=ccc, n_iso=nnn, o_iso=ooo, p_iso=ppp)[:train_split[1]]
    test_loader = ds.GeneratorDataset(lambda: data_generator(test_dataset), ["data"], shuffle=True)
    test_loader = test_loader.batch(b, per_batch_map=collate_list_of_dicts)
    
    model_kwargs = {
        "irreps_in": "5x 0e", 
        "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([125,40,25,15]) for p in [-1, 1]], 
        "irreps_out": "14x0e + 5x1o + 5x2e + 2x3o + 1x4e", 
        "irreps_node_attr": None,
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3), 
        "layers": 3,
        "max_radius": 3.5,
        "num_neighbors": 12.666666,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_nodes": 24,
        "reduce_output": False,
    }

    model = Network(**model_kwargs)
    logging.info("Number of parameters: %d", count_parameters(model))
    optim = nn.Adam(model.trainable_params(), learning_rate=1e-2)
    loss_fn = nn.MSELoss()
    trainer = ms_train(loss_fn, Rs, b)

    if not args.is_train:
        model_dict = ms.load_checkpoint(args.ckpt_path)
        ms.load_param_into_net(model, model_dict)
        logging.info("Load checkpoint from %s", args.ckpt_path)

        model.set_train(False)
        trainer.test_epoch(0, model, test_loader, args.is_train)
        return

    for epoch in range(num_epochs):
        trainer.train_epoch(epoch, model, optim, train_loader)
        trainer.test_epoch(epoch, model, test_loader)
    save_checkpoint(model, os.path.join("./checkpoints/model.ckpt"))


if __name__ == '__main__':
    set_seed(1234)
    np.random.seed(1234)
    main()
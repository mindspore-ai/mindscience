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
# ==============================================================================
"""trainer
"""

import logging
import os
import sys
import time
import yaml

import numpy as np
import mindspore as ms
from mindspore import nn

from mindchemistry.cell.geonet import Allegro
from src.allegro_embedding import AllegroEmbedding
from src.dataset import create_training_dataset
from src.potential import Potential
from src.reduce_lr_on_plateau import ReduceLROnPlateau


def build(num_type, configs):
    """ Build Potential model

    Args:
        num_atom (int): number of atoms

    Returns:
        net (Potential): Potential model
    """
    literal_hidden_dims = 'hidden_dims'
    literal_activation = 'activation'
    literal_weight_init = 'weight_init'
    literal_uniform = 'uniform'

    emb = AllegroEmbedding(
        num_type=num_type,
        cutoff=configs['CUTOFF']
    )

    model = Allegro(
        l_max=configs['L_MAX'],
        irreps_in={
            "pos": "1x1o",
            "edge_index": None,
            "node_attrs": f"{num_type}x0e",
            "node_features": f"{num_type}x0e",
            "edge_embedding": f"{configs['NUM_BASIS']}x0e"
        },
        avg_num_neighbor=configs['AVG_NUM_NEIGHBOR'],
        num_layers=configs['NUM_LAYERS'],
        env_embed_multi=configs['ENV_EMBED_MULTI'],
        two_body_kwargs={
            literal_hidden_dims: configs['two_body_latent_mlp_latent_dimensions'],
            literal_activation: 'silu',
            literal_weight_init: literal_uniform
        },
        latent_kwargs={
            literal_hidden_dims: configs['latent_mlp_latent_dimensions'],
            literal_activation: 'silu',
            literal_weight_init: literal_uniform
        },
        env_embed_kwargs={
            literal_hidden_dims: configs['env_embed_mlp_latent_dimensions'],
            literal_activation: None,
            literal_weight_init: literal_uniform
        },
    )

    net = Potential(
        embedding=emb,
        model=model,
        avg_num_neighbor=configs['AVG_NUM_NEIGHBOR'],
        edge_eng_mlp_latent_dimensions=configs['edge_eng_mlp_latent_dimensions']
    )

    return net


def load_yaml_config_from_path(file_path):
    """load_yaml_config_from_path"""
    # Read YAML experiment definition file
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    return config


def log_config(outdir):
    """log_config

    Args:
        outdir: outdir
    """
    logger = logging.getLogger()
    logger.setLevel('INFO')
    logger.handlers = []
    formatter = logging.Formatter(fmt='%(asctime)s (%(levelname)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 创建文件输出器
    log_file = os.path.join(outdir, 'log.log')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def train(configs, dtype=ms.float32, parallel_mode="NONE"):
    """Train the model on the train dataset."""
    n_epoch = configs['N_EPOCH']
    batch_size = configs['BATCH_SIZE']
    batch_size_eval = configs['BATCH_SIZE_EVAL']
    learning_rate = configs['LEARNING_RATE']
    is_profiling = configs['IS_PROFILING']
    shuffle = configs['SHUFFLE']
    split_random = configs['SPLIT_RANDOM']
    lrdecay = configs['LRDECAY']
    n_train = configs['N_TRAIN']
    n_eval = configs['N_EVAL']
    patience = configs['PATIENCE']
    factor = configs['FACTOR']

    # logdir
    outdir = os.path.join(sys.path[0], 'test_allegro_output/log/6')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    log_config(outdir)

    flags = os.O_RDWR | os.O_CREAT
    with os.fdopen(os.open(os.path.join(outdir, 'input.yaml'), flags, 777), 'w') as f:
        yaml.dump(configs, f)
        logging.info("Dump config file to: %s", (os.path.join(outdir, 'input.yaml')))

    # generate dataset
    logging.info("Loading data...                ")
    data_path = configs['DATA_PATH']
    ds_train, edge_index, batch, ds_test, eval_edge_index, eval_batch, num_type = create_training_dataset(
        config={
            "path": data_path,
            "batch_size": batch_size,
            "batch_size_eval": batch_size_eval,
            "n_train": n_train,
            "n_val": n_eval,
            "split_random": split_random,
            "shuffle": shuffle
        },
        dtype=dtype,
        pred_force=False,
        parallel_mode=parallel_mode
    )

    # Define model
    logging.info("Initializing model...              ")
    model = build(num_type, configs)

    tot_params = 0
    for _, param in enumerate(model.get_parameters(expand=True)):
        tot_params += param.size
        logging.info(param)
    logging.info("Total parameters: %d", tot_params)

    # Instantiate loss function and optimizer
    loss_fn = nn.MSELoss()
    metric_fn = nn.MAELoss()
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)

    # 1. Define forward function
    def forward(x, pos, edge_index, batch, batch_size, energy):
        pred = model(x, pos, edge_index, batch, batch_size)
        loss = loss_fn(pred, energy)
        if batch_size != 0:
            square_atom_num = (x.shape[0] / batch_size) ** 2
        else:
            raise ValueError("batch_size should not be zero")
        if square_atom_num != 0:
            loss = loss / square_atom_num
        else:
            raise ValueError("square_atom_num should not be zero")
        return loss

    # 2. Get gradient function
    backward = ms.value_and_grad(forward, None, optimizer.parameters)
    if parallel_mode == "DATA_PARALLEL":
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

    # 3. Define function of one-step training
    @ms.jit
    def train_step(x, pos, edge_index, batch, batch_size, energy):
        loss_, grads_ = backward(x, pos, edge_index, batch, batch_size, energy)
        if parallel_mode == "DATA_PARALLEL":
            grads_ = grad_reducer(grads_)
        optimizer(grads_)
        return loss_

    def _unpack(data):
        return (data['x'], data['pos']), data['energy']

    def train_epoch(model, trainset, edge_index, batch, batch_size, loss_train: list):
        size = trainset.get_dataset_size()
        model.set_train()
        total_train_loss = 0
        loss_train_epoch = []
        ti = time.time()
        for current, data_dict in enumerate(trainset.create_dict_iterator()):
            inputs, label = _unpack(data_dict)
            loss = train_step(inputs[0], inputs[1], edge_index, batch, batch_size, label)
            # AtomWise
            loss = loss.asnumpy()
            loss_train_epoch.append(loss)
            if current % 10 == 0:
                # pylint: disable=W1203
                logging.info(f"loss: {loss:.16f}  [{current:>3d}/{size:>3d}]")
            total_train_loss += loss

        loss_train.append(loss_train_epoch)
        if size != 0:
            loss_train_avg = total_train_loss / size
        else:
            raise ValueError("size should not be zero")
        np.savetxt(os.path.join(outdir, 'loss_train.txt'), loss_train)
        t_now = time.time()
        logging.info('train loss: %.16f, time gap: %.4f', loss_train_avg, (t_now - ti))

    def test(model, dataset, edge_index, batch, batch_size, loss_fn, loss_eval: list, metric_fn, metric_list: list):
        num_batches = dataset.get_dataset_size()
        model.set_train(False)
        test_loss = 0
        metric = 0
        for _, data_dict in enumerate(dataset.create_dict_iterator()):
            inputs, label = _unpack(data_dict)
            if batch_size != 0:
                atom_num = inputs[0].shape[0] / batch_size
            else:
                raise ValueError("batch_size should not be zero")
            square_atom_num = atom_num ** 2
            pred = model(inputs[0], inputs[1], edge_index, batch, batch_size)
            if square_atom_num != 0:
                test_loss += loss_fn(pred, label).asnumpy() / square_atom_num
            else:
                raise ValueError("square_atom_num should not be zero")
            if atom_num != 0:
                metric += metric_fn(pred, label).asnumpy() / atom_num
            else:
                raise ValueError("atom_num should not be zero")

        test_loss /= num_batches
        metric /= num_batches
        # AtomWise
        loss_eval.append(test_loss)
        metric_list.append(metric)
        np.savetxt(os.path.join(outdir, 'loss_eval.txt'), loss_eval)
        np.savetxt(os.path.join(outdir, 'metric.txt'), metric_list)
        logging.info("Test: mse loss: %.16f", test_loss)
        logging.info("Test: mae metric: %.16f", metric)
        return test_loss

    # == Training ==
    if is_profiling:
        logging.info("Initializing profiler...      ")
        profiler = ms.Profiler(output_path="dump_output" + "/profiler_data", profile_memory=True)

    logging.info("Initializing train...         ")
    logging.info("seed is: %d", ms.get_seed())
    loss_eval = []
    loss_train = []
    metric_list = []
    for t in range(n_epoch):
        logging.info("Epoch %d\n-------------------------------", (t + 1))
        train_epoch(model, ds_train, edge_index, batch, batch_size, loss_train)
        test_loss = test(
            model, ds_test, eval_edge_index, eval_batch, batch_size_eval, loss_fn, loss_eval, metric_fn, metric_list
        )

        if lrdecay:
            lr_scheduler.step(test_loss)
            last_lr = optimizer.learning_rate.value()
            logging.info("lr: %.10f\n", last_lr)

        if (t + 1) % 50 == 0:
            ms.save_checkpoint(model, os.path.join(outdir, "model.ckpt"))

    if is_profiling:
        profiler.analyse()

    logging.info("Done!")

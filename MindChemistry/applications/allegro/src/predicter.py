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
"""predicter
"""

import logging
import os
import sys
import numpy as np
import mindspore as ms
from mindspore import nn

from mindchemistry.cell.geonet import Allegro
from src.allegro_embedding import AllegroEmbedding
from src.dataset import create_test_dataset
from src.potential import Potential


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
    log_file = os.path.join(outdir, 'predict.log')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def pred(configs, dtype=ms.float32):
    """Pred the model on the eval dataset."""
    batch_size_eval = configs['BATCH_SIZE_EVAL']
    n_eval = configs['N_EVAL']

    # logdir
    outdir = os.path.join(sys.path[0], 'test_allegro_output/log/6')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    log_config(outdir)

    logging.info("Loading data...                ")
    data_path = configs['DATA_PATH']
    _, _, _, ds_test, eval_edge_index, eval_batch, num_type = create_test_dataset(
        config={
            "path": data_path,
            "batch_size_eval": batch_size_eval,
            "n_val": n_eval,
        },
        dtype=dtype,
        pred_force=False
    )

    # Define model
    logging.info("Initializing model...              ")
    model = build(num_type, configs)

    # load checkpoint
    ckpt_file = sys.path[0] + '/checkpoint/model.ckpt'
    ms.load_checkpoint(ckpt_file, model)

    # Instantiate loss function and metric function
    loss_fn = nn.MSELoss()
    metric_fn = nn.MAELoss()

    # == Evaluation ==
    logging.info("Initializing Evaluation...         ")
    logging.info("seed is: %d", ms.get_seed())

    pred_list, test_loss, metric = evaluation(
        model, ds_test, eval_edge_index, eval_batch, batch_size_eval, loss_fn, metric_fn
    )

    logging.info("prediction saved")
    np.save(os.path.join(outdir, 'pred.npy'), pred_list)
    logging.info("Test: mse loss: %.16f", test_loss)
    logging.info("Test: mae metric: %.16f", metric)

    logging.info("Done!")

    return pred_list, test_loss, metric


def _unpack(data):
    return (data['x'], data['pos']), data['energy']


def evaluation(model, dataset, edge_index, batch, batch_size, loss_fn, metric_fn):
    """evaluation"""
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    test_loss = 0
    metric = 0
    pred_list = []
    for _, data_dict in enumerate(dataset.create_dict_iterator()):
        inputs, label = _unpack(data_dict)
        if batch_size != 0:
            atom_num = inputs[0].shape[0] / batch_size
        else:
            raise ValueError("batch_size should not be zero")
        square_atom_num = atom_num ** 2
        prediction = model(inputs[0], inputs[1], edge_index, batch, batch_size)
        pred_list.append(prediction.asnumpy())
        if square_atom_num != 0:
            test_loss += loss_fn(prediction, label).asnumpy() / square_atom_num
        else:
            raise ValueError("square_atom_num should not be zero")
        if atom_num != 0:
            metric += metric_fn(prediction, label).asnumpy() / atom_num
        else:
            raise ValueError("atom_num should not be zero")

    test_loss /= num_batches
    metric /= num_batches

    return pred_list, test_loss, metric

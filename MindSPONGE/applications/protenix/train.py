# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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
Training script for Protenix model.
"""
import os
import gc
import time
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.communication import init, get_rank, get_group_size

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs

from protenix.config import parse_configs, parse_sys_args
from protenix.model.protenix import Protenix
from protenix.model.loss import ProtenixLoss
from protenix.data.dataloader import get_dataloaders
from protenix.model.modules import featurization
from protenix.model.feat_batch import Batch

def forward_fn(inputs):
    cum_loss, logits = model(inputs, 42)
    return cum_loss, logits

def train_step(inputs, optim, parallel=None):
    """Define function of one-step training"""
    (loss, _), grads = grad_fn(inputs)

    if parallel == "DATA_PARALLEL":
        grads = grad_reducer(grads)
    optim(grads)
    del _, inputs, grads
    return loss


def train_loop(network, inputs, optim, parallel=None):
    network.set_train()
    loss = train_step(inputs, optim, parallel)

    loss_val = loss.asnumpy()
    print(f"loss: {loss_val:>7f}")
    del loss
    ms.hal.empty_cache()
    gc.collect()
    return loss_val


if __name__ == '__main__':
    ms.set_context(
        mode=1,
    )
    parallel_mode = (
        os.environ.get("PARALLEL_MODE", "NONE").upper()
    )
    if parallel_mode == "DATA_PARALLEL":
        init()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    print('start training')
    configs = {**configs_base, **{"data": data_configs}}
    configs = parse_configs(
        configs,
        parse_sys_args(),
    )
    loss_fn = ProtenixLoss(configs)
    deafult_channel = 256
    feature_input = 449
    pair_channel = 128
    single_channel = 384
    out_channel = 128
    feat_shape = (deafult_channel, feature_input)
    act_shape = (deafult_channel, deafult_channel, pair_channel)
    pair_shape = (deafult_channel, deafult_channel, pair_channel)
    single_shape = (deafult_channel, single_channel)
    num_templates = None
    model = Protenix(Protenix.Config(), configs, feat_shape, act_shape, pair_shape, single_shape,
                     out_channel, num_templates, loss_fn, True, dtype=ms.float32)
    if not configs.load_checkpoint_path == "":
        ckpt_file_name = configs.load_checkpoint_path
        ms.load_checkpoint(ckpt_file_name, model)

    warm_up_lr = nn.WarmUpLR(configs.lr, 2)
    lr = warm_up_lr(ms.Tensor(1))
    optimizer = nn.Adam(params=model.trainable_params(
    ), learning_rate=lr, beta1=0.9, beta2=0.95, weight_decay=1e-8)
    grad_fn = ms.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)
    if parallel_mode == "DATA_PARALLEL":
        grad_reducer = nn.DistributedGradReducer(
            optimizer.parameters, mean=True, degree=get_group_size())

    dataloader, test_dataset = get_dataloaders(
        configs,
        seed=configs.seed,
        error_dir='./',
    )
    if parallel_mode == "DATA_PARALLEL":
        rank_id = get_rank()
        rank_size = get_group_size()
    else:
        rank_id = 0
        rank_size = 1
    epochs = 200
    training_step = 2000
    count = 0
    best_loss = float('inf')

    loss_list = []
    time_list = []

    per_rank_dataset_len = len(dataloader) // rank_size
    idx = np.arange(len(dataloader))
    idx_per_rank = idx[rank_id *
                       per_rank_dataset_len: (rank_id + 1) * per_rank_dataset_len]
    for t in range(epochs):
        for i in idx_per_rank:
            data = dataloader[int(i)]
            count += 1
            print(f'=======step-{count}========')
            batch = Batch()
            batch.load_from_dict(data)
            batch.atom_perm_list = data["input_feature_dict"]["atom_perm_list"]
            batch.label_dict = data["label_dict"]
            batch.label_full_dict = data["label_full_dict"]
            max_relative_idx = 32
            max_relative_chain = 2
            batch.rel_features = featurization.create_relative_encoding(
                batch.token_features, max_relative_idx=max_relative_idx, max_relative_chain=max_relative_chain
            )

            start_step_time = time.time()
            training_loss = train_loop(model, batch, optimizer, parallel_mode)
            end_step_time = time.time()
            print(f'time: {end_step_time-start_step_time}, loss: {training_loss.item()}')
            if count % configs.checkpoint_interval == 0:
                ms.save_checkpoint(model, configs.base_dir + f'/checkpoint_{count}.ckpt')
            if training_loss < best_loss:
                best_loss = training_loss
                ms.save_checkpoint(model, configs.base_dir + '/best_model.ckpt')
            if count == training_step:
                break
        if count == training_step:
            break

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
"""training script."""
import os
import stat
import time
import rdkit
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor

from src.mol_tree import Vocab
from src.jtnn_vae import JTNNVAE
from src.datautils import MoleculeDataset
from src.utils import xavier_normal_, LossCallBack
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config


# Copy single dataset from obs to training image###
def obs_to_env(obs_data_url, data_dir):
    """obs to env"""
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except FileNotFoundError as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    # Set a cache file to determine whether the data has been copied to obs.
    # If this file exists during multi-card training, there is no need to copy the dataset multiple times.
    fd = os.open("/cache/download_input.txt", os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR)
    os.fdopen(fd, "w")
    os.close(fd)
    if os.path.exists("/cache/download_input.txt"):
        print("download_input succeed")


# Copy the output to obs###
def env_to_obs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))
    except FileNotFoundError as e:
        print('moxing upload {} to {} failed: '.format(train_dir, obs_train_url) + str(e))


def download_from_qizhi(obs_data_url, data_dir):
    """download from qizhi"""
    device_num = int(os.getenv('RANK_SIZE'))
    if device_num == 1:
        obs_to_env(obs_data_url, data_dir)
    if device_num > 1:
        # Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
        local_rank = int(os.getenv('RANK_ID'))
        if local_rank % 8 == 0:
            obs_to_env(obs_data_url, data_dir)
        # If the cache file does not exist, it means that the copy data has not been completed,
        # and Wait for 0th card to finish copying data
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)


def upload_to_qizhi(train_dir, obs_train_url):
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank = int(os.getenv('RANK_ID'))
    if device_num == 1:
        env_to_obs(train_dir, obs_train_url)
    if device_num > 1:
        if local_rank % 8 == 0:
            env_to_obs(train_dir, obs_train_url)


def main_train():
    """train"""
    vocab = [x.strip("\r\n ") for x in open(os.path.join(config.raw_data_dir, "zinc/vocab.txt"))]
    vocab = Vocab(vocab)

    net = JTNNVAE(vocab, config.hidden_size, config.latent_size, config.depth, beta=config.pretrain_beta)

    constant_init_0 = ms.common.initializer.Constant(value=0)
    for param in net.trainable_params():
        if param.dim() == 1:
            constant_init_0(param)
        else:
            xavier_normal_(param)

    dataset = MoleculeDataset(os.path.join(config.raw_data_dir, "zinc/train.txt"))
    dataset = ms.dataset.GeneratorDataset(dataset, shuffle=True, column_names=["smiles_input"],
                                          num_parallel_workers=config.num_workers)
    dataset = dataset.batch(batch_size=config.batch_size, drop_remainder=True)

    data_size = dataset.get_dataset_size()
    lr = nn.exponential_decay_lr(config.pretrain_lr, 0.9, data_size * config.pretrain_epoch, data_size, 1)

    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    net.set_train(True)
    model = Model(net, optimizer=optimizer)

    loss_cb = LossCallBack("pretrain", config.batch_size)
    time_cb = TimeMonitor(data_size=data_size)
    callbacks = [loss_cb, time_cb]

    model.train(config.pretrain_epoch,
                dataset,
                callbacks=callbacks,
                dataset_sink_mode=False)

    net.set_beta(config.vaetrain_beta)
    lr = nn.exponential_decay_lr(config.vaetrain_lr, 0.9, data_size * config.vaetrain_epoch, data_size, 1)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    model = Model(net, optimizer=optimizer)

    loss_cb = LossCallBack(run="vaetrain", bsz=config.batch_size)
    time_cb = TimeMonitor(data_size=data_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=data_size, keep_checkpoint_max=3)
    ckpt_cb = ModelCheckpoint(prefix="pre_model", directory=config.save_ckpt_dir, config=ckpt_config)
    callbacks = [loss_cb, time_cb, ckpt_cb]

    model.train(config.vaetrain_epoch,
                dataset,
                callbacks=callbacks,
                dataset_sink_mode=False)


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    set_seed(1)
    context.set_context(mode=context.PYNATIVE_MODE)
    device_id = get_device_id()
    context.set_context(device_target=config.device_target, device_id=device_id)

    if config.enable_modelarts:
        import moxing as mox

        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        if not os.path.exists(config.train_dir):
            os.makedirs(config.train_dir)
        # Initialize and copy data to training image
        download_from_qizhi(config.data_url, config.data_dir)
        config.save_ckpt_dir = config.train_dir
        config.raw_data_dir = config.data_dir

    main_train()
    if config.enable_modelarts:
        upload_to_qizhi(config.train_dir, config.train_url)

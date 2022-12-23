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
"""Evaluation script."""
import os
import stat
import random
from src.mol_tree import Vocab
from src.jtnn_vae import JTNNVAE
from src.model_utils.config import config

import rdkit
import rdkit.Chem as Chem
from tqdm import tqdm
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net


# Copy single dataset from obs to inference image ###
def obs_to_env(obs_data_url, data_dir):
    try:
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except FileNotFoundError as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))


# Copy ckpt file from obs to inference image###
# To operate on folders, use mox.file.copy_parallel. If copying a file.
# Please use mox.file.copy to operate the file, this operation is to operate the file
def obs_url_to_env(obs_ckpt_url, ckpt_url):
    try:
        mox.file.copy(obs_ckpt_url, ckpt_url)
        print("Successfully Download {} to {}".format(obs_ckpt_url, ckpt_url))
    except FileNotFoundError as e:
        print('moxing download {} to {} failed: '.format(obs_ckpt_url, ckpt_url) + str(e))


# Copy the output result to obs###
def env_to_obs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir, obs_train_url))
    except FileNotFoundError as e:
        print('moxing upload {} to {} failed: '.format(train_dir, obs_train_url) + str(e))


def eval_func():
    """evaluation"""
    vocab = [x.strip("\r\n ") for x in open(os.path.join(config.raw_data_dir, "zinc/vocab.txt"))]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, config.hidden_size, config.latent_size, config.depth, stereo=True)
    param_dict = load_checkpoint(config.ckpt_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)

    data = []
    with open(os.path.join(config.raw_data_dir, "zinc/test.txt")) as f:
        for line in f:
            s = line.strip("\r\n ").split()[0]
            data.append(s)

    acc = 0.0
    for smiles in tqdm(data, total=len(data)):
        mol = Chem.MolFromSmiles(smiles)
        smiles_3d = Chem.MolToSmiles(mol, isomericSmiles=True)

        dec_smiles = model.reconstruct(smiles_3d)
        if dec_smiles == smiles_3d:
            acc += 1

    acc /= len(data)
    fd = os.open(config.acclog_path, os.O_RDWR | os.O_CREAT, stat.S_IWUSR | stat.S_IRUSR)
    fo = os.fdopen(fd, "w+")
    fo.write(f'reconstruction accuracy: {acc}')
    os.close(fd)
    print(f'reconstruction accuracy: {acc}')


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    set_seed(1)
    random.seed(1)
    context.set_context(mode=context.PYNATIVE_MODE)
    device_id = int(os.getenv('DEVICE_ID', "0"))
    context.set_context(device_target=config.device_target, device_id=device_id)

    if config.enable_modelarts:
        import moxing as mox

        # Initialize the data and result directories in the inference image###
        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)
        if not os.path.exists(config.result_dir):
            os.makedirs(config.result_dir)

        # Copy dataset from obs to inference image
        obs_to_env(config.data_url, config.data_dir)
        # Copy ckpt file from obs to inference image
        obs_url_to_env(config.ckpt_url, config.ckpt_dir)

        config.raw_data_dir = config.data_dir
        config.ckpt_path = config.ckpt_dir
        config.acclog_path = os.path.join(config.result_dir, config.acclog_path)

    eval_func()

    # Copy result data from the local running environment back to obs,
    # and download it in the inference task corresponding to the Qizhi platform
    if config.enable_modelarts:
        env_to_obs(config.result_dir, config.result_url)

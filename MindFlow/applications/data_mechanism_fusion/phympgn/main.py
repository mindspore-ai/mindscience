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
# ==============================================================================
"""main"""
import argparse

import numpy as np
from easydict import EasyDict
from mindspore import nn
import mindspore as ms
from mindflow import get_multi_step_lr
from mindflow.utils import log_config, load_yaml_config, print_log
from src import PDECFDataset, Trainer, get_data_loader, PhyMPGN, TwoStepLoss


ms.set_device(device_target='Ascend', device_id=7)


def parse_args():
    """parse input args"""
    parser = argparse.ArgumentParser(description="cylinder flow train")
    parser.add_argument("--config_file_path", type=str,
                        default="./yamls/train.yaml")
    parser.add_argument('--train', action='store_true',
                        default=False)
    input_args = parser.parse_args()
    return input_args


def load_config():
    """load config"""
    args = parse_args()
    config = load_yaml_config(args.config_file_path)
    config['train'] = args.train
    config = EasyDict(config)
    if args.train:
        log_config('./logs', f'phympgn-{config.experiment_name}')
    else:
        log_config('./logs', f'phympgn-{config.experiment_name}-te')
    print_log(config)
    return config


def train(config):
    """train"""
    print_log('Train...')
    print_log('Loading training data...')
    tr_dataset = PDECFDataset(
        root=config.path.data_root_dir,
        raw_files=config.path.tr_raw_data,
        dataset_start=config.data.dataset_start,
        dataset_used=config.data.dataset_used,
        time_start=config.data.time_start,
        time_used=config.data.time_used,
        window_size=config.data.tr_window_size,
        training=True
    )
    tr_loader = get_data_loader(
        dataset=tr_dataset,
        batch_size=config.optim.batch_size
    )

    print_log('Loading validation data...')
    val_dataset = PDECFDataset(
        root=config.path.data_root_dir,
        raw_files=config.path.val_raw_data,
        dataset_start=config.data.dataset_start,
        dataset_used=config.data.dataset_used,
        time_start=config.data.time_start,
        time_used=config.data.time_used,
        window_size=config.data.val_window_size
    )
    val_loader = get_data_loader(
        dataset=val_dataset,
        batch_size=config.optim.batch_size
    )

    print_log('Building model...')
    model = PhyMPGN(
        encoder_config=config.network.encoder_config,
        mpnn_block_config=config.network.mpnn_block_config,
        decoder_config=config.network.decoder_config,
        laplace_block_config=config.network.laplace_block_config,
        integral=config.network.integral
    )
    print_log(f'Number of parameters: {model.num_params}')
    lr_scheduler = get_multi_step_lr(
        lr_init=config.optim.lr,
        milestones=list(np.arange(0, config.optim.start_epoch+config.optim.epochs,
                                  step=config.optim.steplr_size)[1:]),
        gamma=config.optim.steplr_gamma,
        steps_per_epoch=len(tr_loader),
        last_epoch=config.optim.start_epoch+config.optim.epochs-1
    )
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr_scheduler,
                                   eps=1.0e-8, weight_decay=1.0e-2)
    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=lr_scheduler, config=config,
        loss_func=TwoStepLoss()
    )
    trainer.train(tr_loader, val_loader)


def test(config):
    """test"""
    te_dataset = PDECFDataset(
        root=config.path.data_root_dir,
        raw_files=config.path.te_raw_data,
        dataset_start=config.data.te_dataset_start,
        dataset_used=config.data.te_dataset_used,
        time_start=config.data.time_start,
        time_used=config.data.time_used,
        window_size=config.data.te_window_size,
        training=False
    )
    te_loader = get_data_loader(
        dataset=te_dataset,
        batch_size=1,
        shuffle=False,
    )

    print_log('Building model...')
    model = PhyMPGN(
        encoder_config=config.network.encoder_config,
        mpnn_block_config=config.network.mpnn_block_config,
        decoder_config=config.network.decoder_config,
        laplace_block_config=config.network.laplace_block_config,
        integral=config.network.integral
    )
    print_log(f'Number of parameters: {model.num_params}')
    trainer = Trainer(
        model=model, optimizer=None, scheduler=None, config=config,
        loss_func=nn.MSELoss()
    )
    print_log('Test...')
    trainer.test(te_loader)


def main():
    config = load_config()
    if config.train:
        train(config)
    else:
        test(config)


if __name__ == '__main__':
    main()

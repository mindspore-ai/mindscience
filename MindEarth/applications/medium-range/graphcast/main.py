# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
graphcast train and test
"""

import argparse
import datetime
import os
import random
import time

import numpy as np

from mindspore import set_seed
from mindspore import context
from mindspore.train.serialization import load_param_into_net

from mindearth.utils import load_yaml_config
from mindearth.data import Dataset, Era5Data

from src import init_data_parallel, init_model, update_config, get_coe, get_param_dict, get_logger
from src import LossNet, GraphCastTrainer, CustomWithLossCell, InferenceModule


set_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='GraphCast')
    parser.add_argument('--config_file_path', type=str, default='./GraphCast.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")

    parser.add_argument('--distribute', type=bool, default=False)
    parser.add_argument('--amp_level', type=str, default='O2')
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')
    parser.add_argument('--load_ckpt', type=bool, default=False)
    parser.add_argument('--data_sink', type=bool, default=False)

    parser.add_argument('--processing_steps', type=int, default=16)
    parser.add_argument('--latent_dims', type=int, default=512)

    parser.add_argument('--mesh_level', type=int, default=4)
    parser.add_argument('--grid_resolution', type=float, default=1.4)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--initial_lr', type=float, default=0.00025)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=1)

    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./summary')
    parser.add_argument('--ckpt_path', type=str, default='')

    params = parser.parse_args()
    return params


def train(cfg, model, logger):
    """GraphCast train function"""
    sj_std, wj, ai = get_coe(cfg)
    data_params = cfg['data']
    rollout_ckpt_pth = cfg['summary']["ckpt_path"]
    steps_per_epoch = 0
    for t in range(data_params['start_rollout_step'], data_params['t_out_train'] + 1):
        data_params['t_out_train'] = t
        if t > 1:
            param_dict, file_dir = get_param_dict(cfg, t - 1, steps_per_epoch, rollout_ckpt_pth=rollout_ckpt_pth)
            load_param_into_net(model, param_dict)
            logger.info(f"Load pre-trained model successfully, {file_dir}")
            rollout_ckpt_pth = None

        loss_fn = LossNet(ai, wj, sj_std, data_params['feature_dims'])
        loss_cell = CustomWithLossCell(backbone=model, loss_fn=loss_fn, data_params=data_params)
        trainer = GraphCastTrainer(cfg, model, loss_cell, logger)
        trainer.train()
        steps_per_epoch = trainer.steps_per_epoch


def test(cfg, model, logger):
    """GraphCast test function"""
    data_params = cfg['data']
    inference_module = InferenceModule(model, cfg, logger)
    test_dataset_generator = Era5Data(data_params=data_params, run_mode='test')
    test_dataset = Dataset(test_dataset_generator, distribute=False,
                           num_workers=data_params['num_workers'], shuffle=False)
    test_dataset = test_dataset.create_dataset(data_params['batch_size'])
    inference_module.eval(test_dataset)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.config_file_path)
    update_config(args, config)
    use_ascend = args.device_target == 'Ascend'
    graph_path = os.path.join(args.output_dir, "graphs")
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        save_graphs=args.save_graphs,
                        save_graphs_path=graph_path)

    if config['train']['distribute']:
        init_data_parallel(use_ascend)
    else:
        context.set_context(device_id=args.device_id)
    grpahcast_model = init_model(config)
    logger_obj = get_logger(config)
    start_time = time.time()
    if args.run_mode == 'train':
        train(config, grpahcast_model, logger_obj)
    else:
        test(config, grpahcast_model, logger_obj)
    print("End-to-End total time: {} s".format(time.time() - start_time))

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
"""Nowcastnet train and test"""
import os
import datetime
import argparse
import random

import numpy as np
import mindspore as ms
from mindspore import context, nn, set_seed
from mindearth.utils.tools import load_yaml_config

from src import init_generation_model, init_data_parallel, get_logger, init_evolution_model
from src import EvolutionTrainer, GenerationTrainer, GenerateLoss, DiscriminatorLoss, EvolutionLoss
from src import EvolutionPredictor, GenerationPredictor
from src import RadarData, NowcastDataset


np.random.seed(0)
set_seed(0)
random.seed(0)


def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='NowcastNet')
    parser.add_argument('--config_file_path', type=str, default='./configs/Nowcastnet.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')
    params = parser.parse_args()
    return params


def train(cfg, logger):
    """training model"""
    module_name = cfg.get("model").get("module_name", "generation")
    if module_name == 'generation':
        # Dynamic Loss scale update cell.
        # see https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/nn/mindspore.nn.DynamicLossScaleUpdateCell.html for detail.
        loss_scale = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
        g_model, d_model = init_generation_model(cfg)
        g_loss_fn = GenerateLoss(g_model, d_model)
        d_loss_fn = DiscriminatorLoss(g_model, d_model)
        trainer = GenerationTrainer(cfg, g_model, d_model, g_loss_fn, d_loss_fn, logger, loss_scale)
        trainer.train()
    elif module_name == 'evolution':
        # Loss scale manager with a fixed loss scale value.
        # see https://www.mindspore.cn/docs/en/r2.2/api_python/amp/mindspore.amp.FixedLossScaleManager.html for detail.
        loss_scale = ms.train.loss_scale_manager.FixedLossScaleManager(loss_scale=2048)
        model = init_evolution_model(cfg)
        loss_fn = EvolutionLoss(model, cfg)
        trainer = EvolutionTrainer(cfg, model, loss_fn, logger, loss_scale)
        trainer.train()


def test(cfg, logger):
    """test"""
    train_params = cfg.get("train")
    data_params = cfg.get("data")
    module_name = cfg.get("model").get("module_name", "generation")
    test_dataset_generator = RadarData(data_params, run_mode='test', module_name=module_name)
    test_dataset = NowcastDataset(test_dataset_generator,
                                  module_name=module_name,
                                  distribute=train_params.get('distribute', False),
                                  num_workers=data_params.get('num_workers', 1),
                                  shuffle=False)
    test_dataset = test_dataset.create_dataset(data_params.get('batch_size', 1))
    if module_name == 'generation':
        g_model, _ = init_generation_model(cfg, run_mode='test')
        predictor = GenerationPredictor(cfg, g_model, logger)
        predictor.eval(test_dataset)
    elif module_name == 'evolution':
        model = init_evolution_model(cfg, run_mode='test')
        predictor = EvolutionPredictor(cfg, model, logger)
        predictor.eval(test_dataset)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.config_file_path)
    use_ascend = args.device_target == 'Ascend'
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target)
    if config.get('train').get('distribute', False):
        init_data_parallel(use_ascend)
    else:
        context.set_context(device_id=args.device_id)
    logger_obj = get_logger(config)
    if args.run_mode == 'train':
        train(config, logger_obj)
    elif args.run_mode == 'test':
        test(config, logger_obj)

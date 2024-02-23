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
fuxi train and test
"""

import argparse
import datetime
import os
import random
import time

import numpy as np

from mindspore import set_seed, context
from mindearth.utils import load_yaml_config
from mindearth.data import Dataset, Era5Data

from src import init_data_parallel, init_model, get_logger
from src import MAELossForMultiLabel, FuXiTrainer, CustomWithLossCell, InferenceModule

set_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='FuXi')
    parser.add_argument('--config_file_path', type=str, default='./configs/FuXi.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')

    params = parser.parse_args()
    return params


def train(cfg, model, logger):
    """FuXi train function"""
    data_params = cfg.get('data')
    optimizer_params = cfg.get('optimizer')
    loss_fn = MAELossForMultiLabel(data_params=data_params, optimizer_params=optimizer_params)
    loss_cell = CustomWithLossCell(backbone=model, loss_fn=loss_fn)
    trainer = FuXiTrainer(cfg, model, loss_cell, logger)
    trainer.train()


def test(cfg, model, logger):
    """FuXi test function"""
    data_params = cfg.get('data')
    test_dataset_generator = Era5Data(data_params=data_params, run_mode='test')
    test_dataset = Dataset(test_dataset_generator, distribute=False,
                           num_workers=data_params.get('num_workers'), shuffle=False)
    test_dataset = test_dataset.create_dataset(data_params.get('batch_size'))
    inference_module = InferenceModule(model, cfg, logger)
    inference_module.eval(test_dataset)


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

    fuxi_model = init_model(config, args.run_mode)

    logger_obj = get_logger(config)
    start_time = time.time()

    if args.run_mode == "train":
        train(config, fuxi_model, logger_obj)
    else:
        test(config, fuxi_model, logger_obj)
    print("End-to-End total time: {} s".format(time.time() - start_time))

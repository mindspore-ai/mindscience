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
"""
 SKNO train and test
"""

import argparse
import datetime
import os
import time
import random

import numpy as np
from mindspore import set_seed
from mindspore import context
from mindearth.utils import load_yaml_config
from mindearth.data import Dataset, Era5Data
from src import init_model, init_data_parallel, SKNOTrainer, update_config, InferenceModule, get_logger, \
    CustomWithLossCell, Lploss

set_seed(0)
np.random.seed(0)
random.seed(0)

def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='SKNO')
    parser.add_argument('--yaml', type=str, default='configs/skno.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')

    params = parser.parse_args()
    return params


def train(cfg, model, logger):
    """SKNO train function"""
    loss_fn = Lploss()
    loss_net = CustomWithLossCell(model, loss_fn)
    trainer = SKNOTrainer(cfg, model, loss_net, logger)
    trainer.train()


def test(cfg, model, logger):
    """SKNO test function"""
    cfg_data = cfg["data"]
    inference_module = InferenceModule(model, cfg, logger)
    test_dataset_generator = Era5Data(data_params=cfg_data, run_mode='valid')
    test_dataset = Dataset(test_dataset_generator, distribute=False,
                           num_workers=cfg_data['num_workers'], shuffle=False)
    test_dataset = test_dataset.create_dataset(cfg_data['batch_size'])
    inference_module.eval(test_dataset)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.yaml)
    update_config(args, config)
    use_ascend = args.device_target == 'Ascend'
    graph_path = os.path.join(config['summary']["output_dir"], "graphs")
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        save_graphs=args.save_graphs,
                        save_graphs_path=graph_path)

    if config['train']['distribute']:
        init_data_parallel(use_ascend)
    else:
        context.set_context(device_id=args.device_id)

    skno_model = init_model(config)
    logger_obj = get_logger(config)
    start_time = time.time()
    if args.run_mode == 'train':
        train(config, skno_model, logger_obj)
    else:
        test(config, skno_model, logger_obj)
    print("End-to-End total time: {} s".format(time.time() - start_time))

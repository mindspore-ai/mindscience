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
# ==============================================================================
"""
train
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

from src import init_model, init_data_parallel, ViTKNOTrainer, update_config, InferenceModule, get_logger, \
    CustomWithLossCell, Lploss

set_seed(0)
np.random.seed(0)
random.seed(0)

def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='ViTKNO')
    parser.add_argument('--yaml', type=str, default='vit_kno.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")

    parser.add_argument('--distribute', type=bool, default=False)
    parser.add_argument('--rank_size', type=int, default=1)
    parser.add_argument('--amp_level', type=str, default='O2')
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')
    parser.add_argument('--load_ckpt', type=bool, default=False)
    parser.add_argument('--data_sink', type=bool, default=False)

    parser.add_argument('--grid_resolution', type=float, default=1.4)
    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--initial_lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=1)

    parser.add_argument('--valid_frequency', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./summary')
    parser.add_argument('--ckpt_path', type=str, default='')
    input_args = parser.parse_args()
    return input_args


def train(cfg, model, logger):
    """vit-kno train function"""
    loss_fn = Lploss()
    loss_net = CustomWithLossCell(model, loss_fn)
    trainer = ViTKNOTrainer(cfg, model, loss_net, logger)
    trainer.train()


def test(cfg, model, logger):
    """vit-kno test function"""
    inference_module = InferenceModule(model, cfg, logger)
    test_dataset_generator = Era5Data(data_params=cfg["data"], run_mode='test')
    test_dataset = Dataset(test_dataset_generator, distribute=False,
                           num_workers=cfg["data"]['num_workers'], shuffle=False)
    test_dataset = test_dataset.create_dataset(cfg["data"]['batch_size'])
    inference_module.eval(test_dataset)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.yaml)
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

    vit_kno_model = init_model(config)
    logger_obj = get_logger(config)
    start_time = time.time()
    if args.run_mode == 'train':
        train(config, vit_kno_model, logger_obj)
    else:
        test(config, vit_kno_model, logger_obj)
    print("End-to-End total time: {} s".format(time.time() - start_time))

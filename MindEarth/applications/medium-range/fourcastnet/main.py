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
fourcastnet train and test
"""

import argparse
import datetime
import os
import time

from mindearth.core import RelativeRMSELoss
from mindearth.data import Era5Data
from mindearth.utils import load_yaml_config
from mindspore import context

from src import init_model, init_data_parallel, FCNTrainer, update_config, InferenceModule, get_logger


def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='FourCastNet')
    parser.add_argument('--config_file_path', type=str, default='./configs/FourCastNet.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')
    input_args = parser.parse_args()
    return input_args


def train(cfg, model, logger):
    """Fourcastnet train function"""
    loss_fn = RelativeRMSELoss()
    trainer = FCNTrainer(cfg, model, loss_fn, logger)
    trainer.train()


def test(cfg, model, logger):
    """Fourcastnet test function"""
    inference_module = InferenceModule(model, cfg, logger)
    test_dataset_generator = Era5Data(data_params=cfg["data"], run_mode='valid')
    inference_module.eval(test_dataset_generator, generator_flag=True)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.config_file_path)
    update_config(args, config)
    use_ascend = args.device_target == 'Ascend'
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target)

    if config['train']['distribute']:
        init_data_parallel(use_ascend)
    else:
        context.set_context(device_id=args.device_id)

    fno_model = init_model(config)
    logger_obj = get_logger(config)
    start_time = time.time()
    if args.run_mode == 'train':
        train(config, fno_model, logger_obj)
    else:
        test(config, fno_model, logger_obj)
    print("End-to-End total time: {} s".format(time.time() - start_time))

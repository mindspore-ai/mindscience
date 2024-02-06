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
ensoforcast train and test
"""
import os
import time
import datetime
import argparse
import random

import numpy as np
from mindspore import set_seed
from mindspore import context
import mindspore.dataset as ds
from mindspore.train.serialization import load_param_into_net
from mindearth.utils import load_yaml_config

from src import WeightedLoss, CTEFTrainer, InferenceModule, init_model
from src import get_logger, ReanalysisData, plot_correlation, get_param_dict


set_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='CTEFNet')
    parser.add_argument('--config_file_path', type=str, default='./configs/pretrain.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="GPU")
    parser.add_argument("--mode", type=str, default="PYNATIVE", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')

    params = parser.parse_args()
    return params


def train(cfg, model, logger):
    """CTEFNet train function"""
    data_params = cfg.get('data')
    opt_params = cfg.get('optimizer')
    for t in range(data_params.get('t_in'), data_params.get('t_out_train') + 1):
        if t > 1:
            param_dict, file_dir = get_param_dict(cfg, t - 1)
            load_param_into_net(model, param_dict)
            logger.info(f"Load pre-trained model successfully, {file_dir}")

        loss_fn = WeightedLoss(
            opt_params.get('loss_alpha'),
            opt_params.get('loss_beta'),
            opt_params.get('loss_gamma'),
            opt_params.get('corr_point'),
            data_params.get('obs_time')
        )
        trainer = CTEFTrainer(cfg, model, loss_fn, logger)
        trainer.train()


def test(cfg, model, logger):
    """CTEFNet test function"""
    data_params = cfg.get('data')
    summary_params = cfg.get('summary')
    test_dataset = ReanalysisData(
        data_params.get('root_dir'),
        data_params.get('test_period'),
        data_params.get('obs_time'),
        data_params.get('pred_time')
    )
    test_dataloader = ds.GeneratorDataset(
        test_dataset,
        ["data", "index"],
        shuffle=False).batch(data_params.get('valid_batch_size'), False)
    corr_list = []
    for t in range(1, summary_params.get('plot_line')+1):
        param_dict, file_dir = get_param_dict(cfg, t)
        load_param_into_net(model, param_dict)
        logger.info(f"Load pre-trained model successfully, {file_dir}")
        inference_module = InferenceModule(model, cfg, logger)
        corr_list.append(inference_module.eval(test_dataloader)[data_params.get('obs_time'):])
    plot_correlation(cfg, corr_list)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.config_file_path)
    use_ascend = args.device_target == 'Ascend'
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target)
    context.set_context(device_id=args.device_id)
    ctefnet_model = init_model(config, args.run_mode)
    logger_obj = get_logger(config)
    start_time = time.time()
    if args.run_mode == 'train':
        train(config, ctefnet_model, logger_obj)
    else:
        test(config, ctefnet_model, logger_obj)
    print("End-to-End total time: {} s".format(time.time() - start_time))

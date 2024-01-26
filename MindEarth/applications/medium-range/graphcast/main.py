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
from mindspore import set_seed, context, nn
from mindspore.train.serialization import load_param_into_net

from mindearth.utils import load_yaml_config
from mindearth.data import Dataset, Era5Data
from src import init_data_parallel, init_model, get_coe, get_param_dict, get_logger, init_tp_model
from src import LossNet, GraphCastTrainer, GraphCastTrainerTp, CustomWithLossCell, InferenceModule, InferenceModuleTp
from src import Era5DataTp


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
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')

    params = parser.parse_args()
    return params


def train(cfg, model, logger):
    """GraphCast train function"""
    sj_std, wj, ai = get_coe(cfg)
    data_params = cfg.get('data')
    summary_params = cfg.get('summary')
    rollout_ckpt_pth = summary_params.get("ckpt_path")
    steps_per_epoch = 0
    for t in range(data_params.get('rollout_steps'), data_params.get('t_out_train') + 1):
        data_params['t_out_train'] = t
        if t > 1:
            param_dict, file_dir = get_param_dict(cfg, t - 1, steps_per_epoch, rollout_ckpt_pth=rollout_ckpt_pth)
            load_param_into_net(model, param_dict)
            logger.info(f"Load pre-trained model successfully, {file_dir}")
            rollout_ckpt_pth = None
        if not data_params.get("tp", False):
            loss_fn = LossNet(ai, wj, sj_std, data_params.get('feature_dims'))
            loss_cell = CustomWithLossCell(backbone=model, loss_fn=loss_fn, data_params=data_params)
            trainer = GraphCastTrainer(cfg, model, loss_cell, logger)
        else:
            loss_scale = nn.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
            loss_fn = LossNet(ai, wj, sj_std, data_params.get('feature_dims'), data_params.get("tp"))
            loss_cell = CustomWithLossCell(backbone=model, loss_fn=loss_fn, data_params=data_params)
            trainer = GraphCastTrainerTp(cfg, model, loss_cell, logger, loss_scale)
        trainer.train()
        steps_per_epoch = trainer.steps_per_epoch


def test(cfg, model, logger):
    """GraphCast test function"""
    data_params = cfg.get('data')
    if data_params.get("tp", False):
        inference_module = InferenceModuleTp(model, cfg, logger)
        test_dataset_generator = Era5DataTp(data_params=data_params, run_mode='test')
    else:
        inference_module = InferenceModule(model, cfg, logger)
        test_dataset_generator = Era5Data(data_params=data_params, run_mode='test')
    test_dataset = Dataset(test_dataset_generator, distribute=False,
                           num_workers=data_params.get('num_workers'), shuffle=False)
    test_dataset = test_dataset.create_dataset(data_params.get('batch_size'))
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
    if config.get("data").get("tp"):
        graphcast_model = init_tp_model(config, run_mode=args.run_mode)
    else:
        graphcast_model = init_model(config, args.run_mode)
    logger_obj = get_logger(config)
    start_time = time.time()
    if args.run_mode == 'train':
        train(config, graphcast_model, logger_obj)
    else:
        test(config, graphcast_model, logger_obj)
    print("End-to-End total time: {} s".format(time.time() - start_time))

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
dgmr train and test
"""

import os
import argparse
import datetime

from mindspore import context

from mindearth.utils import load_yaml_config, create_logger
from mindearth.data import RadarData, Dataset

from src import init_model, update_config, init_data_parallel
from src import GenWithLossCell, DiscWithLossCell, DgmrTrainer, InferenceModule


def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='DgmrNet')
    parser.add_argument('--config_file_path', type=str, default='./DgmrNet.yaml')
    parser.add_argument('--device_target', '-d', type=str, choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument('--device_id', type=int, default=5)
    parser.add_argument('--distribute', type=bool, default=False)
    parser.add_argument('--rank_size', type=int, default=1)
    parser.add_argument('--amp_level', type=str, default='O2')
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')
    parser.add_argument('--load_ckpt', type=bool, default=False)

    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--keep_checkpoint_max', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./summary')
    parser.add_argument('--ckpt_path', type=str, default='')

    params = parser.parse_args()
    return params


def train(cfg, g_m, d_m, loss_g, loss_d, log):
    """dgmr train function"""
    g_m.set_train()
    d_m.set_train()
    trainer = DgmrTrainer(cfg, g_m, d_m, loss_g, loss_d, log)
    trainer.train()


def test(cfg, g_m, log):
    """dgmr test function"""
    test_dataset_generator = RadarData(data_params=cfg["data"], run_mode='test')
    test_dataset = Dataset(test_dataset_generator, distribute=False,
                           num_workers=1,
                           shuffle=False)
    test_dataset = test_dataset.create_dataset(cfg["data"]['batch_size'])
    inference_module = InferenceModule(log, config['summary']["csi_thresholds"])
    inference_module.eval(test_dataset, g_m)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.config_file_path)
    update_config(args, config)

    use_ascend = args.device_target == 'Ascend'

    logger = create_logger(path=os.path.join(config['summary']["summary_dir"], "results.log"))
    logger.info("pid: {}".format(os.getpid()))
    logger.info(config['train'])
    logger.info(config['model'])
    logger.info(config['data'])
    logger.info(config['optimizer'])
    logger.info(config['summary'])

    if config['train']['distribute']:
        init_data_parallel(use_ascend)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                            device_id=config['train']['device_id'])

    g_model, d_model = init_model(config)
    g_loss_fn = GenWithLossCell(g_model, d_model, config["model"]["generation_steps"], config["model"]["grid_lambda"])
    d_loss_fn = DiscWithLossCell(g_model, d_model)
    if args.run_mode == 'train':
        train(config, g_model, d_model, g_loss_fn, d_loss_fn, logger)
    else:
        test(config, g_model, logger)

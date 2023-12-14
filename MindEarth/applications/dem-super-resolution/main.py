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

import os
import argparse
import datetime

from mindspore import context, nn

from mindearth.utils import load_yaml_config, create_logger
from mindearth.data import DemData, Dataset

from src import InferenceModule, DemSrTrainer
from src import init_model, update_config, init_data_parallel


def get_args():
    """Get user specified parameters."""
    parser = argparse.ArgumentParser(description='DEM-SRNet')
    parser.add_argument('--config_file_path', type=str, default='DEM-SRNet.yaml')
    parser.add_argument('--device_target', '-d', type=str,
                        choices=["Ascend", "GPU"], default="Ascend")
    parser.add_argument('--device_id', type=int, default=3)
    parser.add_argument('--distribute', type=bool, default=False)
    parser.add_argument('--rank_size', type=int, default=1)
    parser.add_argument('--amp_level', type=str, default='O2')
    parser.add_argument('--run_mode', type=str, choices=["train", "test"], default='train')
    parser.add_argument('--load_ckpt', type=bool, default=False)

    parser.add_argument('--num_workers', type=int, default=1)

    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--valid_frequency', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='./summary')
    parser.add_argument('--ckpt_path', type=str, default='')

    params = parser.parse_args()
    return params


def train(cfg, model, loss_fn, log):
    """Dem train function"""
    trainer = DemSrTrainer(cfg, model, loss_fn, log)
    trainer.train()


def test(cfg, model, log):
    """Dem test function"""
    inference_module = InferenceModule(model, cfg, log)
    test_dataset_generator = DemData(data_params=cfg["data"], run_mode='test')
    test_dataset = Dataset(test_dataset_generator, distribute=False,
                           num_workers=cfg["data"]['num_workers'], shuffle=False)
    test_dataset = test_dataset.create_dataset(cfg["data"]['batch_size'])
    inference_module.eval(test_dataset)


if __name__ == '__main__':
    print("pid: {}".format(os.getpid()))
    print(datetime.datetime.now())
    args = get_args()
    config = load_yaml_config(args.config_file_path)
    update_config(args, config)

    use_ascend = args.device_target == 'Ascend'

    logger = create_logger(path=os.path.join(config['summary']["summary_dir"], "results.log"))
    logger.info(f"pid: {os.getpid()}")
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

    dem_model = init_model(config)
    dem_loss_fn = nn.MSELoss()
    if args.run_mode == 'train':
        train(config, dem_model, dem_loss_fn, logger)
    else:
        test(config, dem_model, logger)

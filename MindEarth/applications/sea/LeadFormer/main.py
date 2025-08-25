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
"main"
import argparse

import mindspore as ms
from mindspore import set_seed, context
from mindearth.utils import load_yaml_config

from src.solver import Trainer
from src.forecast import Tester

set_seed(0)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", default=5, type=int)
    parser.add_argument("--device_target", default="Ascend", type=str)
    parser.add_argument("--cfg", default="./configs/2km_ice_config.yaml", type=str)
    parser.add_argument("--mode", default="test")
    params = parser.parse_args()
    return params


def train(cfg):
    """train"""
    epoch_size = (
        cfg["train"].get("epochs", 300)
        if not cfg["train"].get("run_distribute", False)
        else cfg["train"].get("distribute_epochs", 300)
    )
    trainer = Trainer(cfg, epochs=epoch_size)
    trainer.train()


def test(cfg):
    """test"""
    evaluator = Tester(cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    args = get_parser()
    config = load_yaml_config(args.cfg)
    print("config", config)
    if args.mode == "test":
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif args.mode == "train":
        context.set_context(mode=ms.GRAPH_MODE)
    else:
        raise ValueError(f"Invalid mode: '{args.mode}'. Expected 'test' or 'train'.")
    ms.set_device(device_target=args.device_target, device_id=args.device_id)
    if args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)
    else:
        raise ValueError(f"Invalid mode: '{args.mode}'. Expected 'test' or 'train'.")

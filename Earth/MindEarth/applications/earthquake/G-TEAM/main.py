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
"main function"
import argparse

import mindspore as ms
from mindspore import context
from mindearth import load_yaml_config, make_dir

from src.utils import init_model, get_logger
from src.forcast import GTeamInference


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="./config/GTEAM.yaml", type=str)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--device_target", default="Ascend", type=str)
    parse_args = parser.parse_args()
    return parse_args


def test(cfg):
    """main test"""
    save_dir = cfg["summary"].get("summary_dir", "./summary")
    make_dir(save_dir)
    model = init_model(cfg)
    logger_obj = get_logger(cfg)
    processor = GTeamInference(model, cfg, save_dir, logger_obj)
    processor.test()


if __name__ == "__main__":
    args = get_args()
    config = load_yaml_config(args.cfg_path)
    context.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(device_target=args.device_target, device_id=args.device_id)
    test(config)

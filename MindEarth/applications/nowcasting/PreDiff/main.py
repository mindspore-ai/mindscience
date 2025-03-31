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
"diffusion main"
import argparse

import mindspore as ms
from mindspore import set_seed, context
from mindearth.utils import load_yaml_config

from src import (
    prepare_output_directory,
    configure_logging_system,
    prepare_dataset,
    init_model,
    PreDiffModule,
    DiffusionTrainer,
    DiffusionInferrence
)


set_seed(0)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--device_target", default="Ascend", type=str)
    parser.add_argument('--cfg', default="./configs/diffusion.yaml", type=str)
    parser.add_argument("--mode", default="train")
    params = parser.parse_args()
    return params


def train(cfg, arg, module):
    output_dir = prepare_output_directory(cfg, arg.device_id)
    logger = configure_logging_system(output_dir, cfg)
    dm, total_num_steps = prepare_dataset(cfg, PreDiffModule)
    trainer = DiffusionTrainer(
        main_module=module, dm=dm, logger=logger, config=cfg
    )
    trainer.train(total_steps=total_num_steps)


def test(cfg, arg, module):
    output_dir = prepare_output_directory(cfg, arg.device_id)
    logger = configure_logging_system(output_dir, cfg)
    dm, _ = prepare_dataset(cfg, PreDiffModule)
    tester = DiffusionInferrence(
        main_module=module, dm=dm, logger=logger, config=cfg
    )
    tester.test()


if __name__ == "__main__":
    args = get_parser()
    config = load_yaml_config(args.cfg)
    context.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(device_target=args.device_target, device_id=args.device_id)
    main_module = PreDiffModule(oc_file=args.cfg)
    main_module = init_model(module=main_module, config=config, mode=args.mode)
    if args.mode == "train":
        train(config, args, main_module)
    else:
        test(config, args, main_module)
        
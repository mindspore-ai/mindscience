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
# ============================================================================
"""train process"""
import os
import io
import warnings
import datetime
import argparse
import yaml

import mindspore as ms

from src import GenerationModule, RankingModule

if __name__ == "__main__":
    # set params
    # load arg parser
    parser = argparse.ArgumentParser(description="Argparser for Train")
    parser.add_argument("-s", "--stage", type=list, default=[2])
    parser.add_argument("-r", "--root", type=str, default=os.path.abspath('.'))
    parser.add_argument("-n", "--exp_name", type=str, default=str(datetime.datetime.now())[:19].replace(" ", "-"))
    parser.add_argument("-m", "--mode", default=ms.GRAPH_MODE)
    parser.add_argument("-dt", "--device_target", type=str, default='Ascend')
    parser.add_argument("-di", "--device_id", type=int, default=6)
    parser.add_argument("-c", "--config_path", type=str, default=os.path.abspath('.') + '/config.yml')
    args = parser.parse_args()

    # load config
    with io.open(args.config_path, 'r') as stream:
        params = yaml.safe_load(stream)
    train_params = params['train_params']
    wae_params = params['wae_params']
    cls_params = params['cls_params']
    ensem_params = params['ensem_params']
    train_params['root'] = args.root
    train_params['exp_name'] = args.exp_name

    # create save directory for current experiment
    exp_dir = os.path.join(args.root, 'save_dir', args.exp_name)
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    else:
        warnings.warn('Current experiment file exists.')
    train_params['folder_dir'] = exp_dir

    # set context
    ms.set_context(mode=args.mode, device_target=args.device_target, device_id=args.device_id)

    # generation model train:
    if 1 in args.stage:
        # set generation models trainer
        wae_params.update(train_params)
        cls_params.update(train_params)
        gen_trainer = GenerationModule(wae_params, cls_params)
        # generation models training
        gen_trainer.train()

    # ranking model train:
    if 2 in args.stage:
        # set ranking model trainer
        ensem_params.update(train_params)
        rank_trainer = RankingModule(ensem_params)
        # ranking model training
        rank_trainer.train()

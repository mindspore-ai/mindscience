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

import argparse
import os
import time

import numpy as np

import mindspore
from mindspore import context, nn, ops, jit, set_seed
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint

from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_test_dataset, visual, calculate_l2_error, \
    AllenCahn, MultiScaleFCSequentialOutputTransform

set_seed(123456)
np.random.seed(123456)

def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description="allen cahn train")
    parser.add_argument("--config_file_path", type=str, default="./configs/allen_cahn_cfg.yaml")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend","CPU"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=2, help="ID of the target device")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")

    input_args = parser.parse_args()
    return input_args


def test(train_args):
    '''Train and evaluate the network'''
    # load configurations
    config = load_yaml_config(train_args.config_file_path)

    # create dataset
    ac_train_dataset = create_training_dataset(config)
    train_dataset = ac_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                    shuffle=True,
                                                    prebatched_data=True,
                                                    drop_remainder=True)
    # create  test dataset
    inputs, label = create_test_dataset(config["test_dataset_path"])

    # define models and optimizers
    model = MultiScaleFCSequentialOutputTransform(in_channels=config["model"]["in_channels"],
                                                  out_channels=config["model"]["out_channels"],
                                                  layers=config["model"]["layers"],
                                                  neurons=config["model"]["neurons"],
                                                  residual=config["model"]["residual"],
                                                  act=config["model"]["activation"],
                                                  num_scales=1)

    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(model, param_dict)

    # define optimizer
    optimizer = nn.Adam(model.trainable_params(),
                        config["optimizer"]["initial_lr"])
    problem = AllenCahn(model)

    model.set_train(False)
    calculate_l2_error(model, inputs, label, config["train_batch_size"])

if __name__ == '__main__':
    print("pid:", os.getpid())
    start_time = time.time()
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    test(args)
    print("End-to-End total time: {} s".format(time.time() - start_time))
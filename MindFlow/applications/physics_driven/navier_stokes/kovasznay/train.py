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
from mindspore import context, jit, load_checkpoint, load_param_into_net
from mindspore import nn, ops, save_checkpoint, set_seed
import mindspore.common.dtype as mstype
from src import Kovasznay, calculate_l2_error, create_dataset, visual

from mindflow.cell import FCSequential
from mindflow.utils import load_yaml_config

set_seed(0)
np.random.seed(0)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="kovasznay flow train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./configs//kovasznay_cfg.yaml",
                        help="config file path")
    intput_args = parser.parse_args()

    return intput_args


def train(file_cfg):
    """Train and evaluate the network"""
    # load configurations
    config = load_yaml_config(file_cfg)

    # create dataset
    ds_train = create_dataset(config)

    # create network
    model = FCSequential(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        layers=config["model"]["layers"],
        neurons=config["model"]["neurons"],
        residual=config["model"]["residual"],
        act="tanh",
    )

    use_ascend = context.get_context(attr_key="device_target") == "Ascend"
    if use_ascend:
        model.to_float(mstype.float16)
    else:
        model.to_float(mstype.float32)

    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)

    params = model.trainable_params()
    optimizer = nn.Adam(params, learning_rate=config["optimizer"]["initial_lr"])

    # create the problem
    problem = Kovasznay(model)
    grad_fn = ops.value_and_grad(
        problem.get_loss, None, optimizer.parameters, has_aux=False
    )

    @jit
    def train_step(pde_data, bc_data):
        loss, grads = grad_fn(pde_data, bc_data)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    def train_epoch(model, dataset, i_epoch):
        model.set_train()
        n_step = dataset.get_dataset_size()
        for i_step, (pde_data, bc_data) in enumerate(dataset):
            local_time_beg = time.time()
            loss = train_step(pde_data, bc_data)

            if i_step % 50 == 0:
                print(
                    "\repoch: {}, loss: {:>f}, time elapsed: {:.1f}ms [{}/{}]".format(
                        i_epoch,
                        float(loss),
                        (time.time() - local_time_beg) * 1000,
                        i_step + 1,
                        n_step,
                    )
                )

    time_beg = time.time()
    for i_epoch in range(config["epochs"]):
        train_epoch(model, ds_train, i_epoch)
    print("End-to-End total time: {} s".format(time.time() - time_beg))
    if config["save_ckpt"]:
        save_checkpoint(model, config["save_ckpt_path"])

    visual(model, config, resolution=config["visual_resolution"])

    n_samps = 10000  # Number of test samples
    ds_test = create_dataset(config, n_samps)
    calculate_l2_error(problem, model, ds_test)


if __name__ == "__main__":
    print("pid:", os.getpid())
    args = parse_args()
    context.set_context(
        mode=context.GRAPH_MODE
        if args.mode.upper().startswith("GRAPH")
        else context.PYNATIVE_MODE,
        device_target=args.device_target,
        device_id=args.device_id,
    )
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    train(args.config_file_path)

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
import time
import datetime
import argparse
import numpy as np

import mindspore
from mindspore import nn, context, ops, Tensor, jit, set_seed, save_checkpoint
import mindspore.common.dtype as mstype

from mindflow.cell import FNO2D
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.loss import RelativeRMSELoss
from mindflow.utils import load_yaml_config
from mindflow.pde import UnsteadyFlowWithLoss
from src import calculate_l2_error, create_training_dataset

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Navier Stokes problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=3, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./navier_stokes_2d.yaml")
    input_args = parser.parse_args()
    return input_args


def train():
    '''train and evaluate the network'''
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    # prepare dataset
    train_dataset = create_training_dataset(data_params,
                                            input_resolution=model_params["input_resolution"],
                                            shuffle=True)
    test_input = np.load(os.path.join(data_params["path"], "test/inputs.npy"))
    test_label = np.load(os.path.join(data_params["path"], "test/label.npy"))

    if use_ascend:
        compute_type = mstype.float16
    else:
        compute_type = mstype.float32
    # prepare model
    model = FNO2D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
                  resolution=model_params["input_resolution"],
                  modes=model_params["modes"],
                  channels=model_params["width"],
                  depths=model_params["depth"],
                  compute_dtype=compute_type
                  )

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}-{v}")
    model_name = "_".join(model_params_list)

    # prepare optimizer
    steps_per_epoch = train_dataset.get_dataset_size()
    print("steps_per_epoch: ", steps_per_epoch)
    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["initial_lr"],
                                        last_epoch=optimizer_params["train_epochs"],
                                        steps_per_epoch=steps_per_epoch,
                                        warmup_epochs=optimizer_params["warmup_epochs"])

    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=Tensor(lr),
                                   weight_decay=optimizer_params['weight_decay'])
    problem = UnsteadyFlowWithLoss(model, loss_fn=RelativeRMSELoss(), data_format="NHWTC")

    def forward_fn(train_inputs, train_label):
        loss = problem.get_loss(train_inputs, train_label)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(train_inputs, train_label):
        loss, grads = grad_fn(train_inputs, train_label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)
    summary_dir = os.path.join(config["summary_dir"], model_name)
    ckpt_dir = os.path.join(summary_dir, "ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(1, 1+optimizer_params["train_epochs"]):
        local_time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epoch):
            cur_loss = sink_process()
        print(f"epoch: {epoch} train loss: {cur_loss} epoch time: {time.time() - local_time_beg:.2f}s")

        model.set_train(False)
        if epoch % config["save_ckpt_interval"] == 0:
            save_checkpoint(model, os.path.join(ckpt_dir, model_params["name"]))

        if epoch % config['eval_interval'] == 0:
            calculate_l2_error(model, test_input, test_label, config["test_batch_size"])


if __name__ == '__main__':
    print(f"pid: {os.getpid()}")
    print(datetime.datetime.now())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target, device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print(f"use_ascend: {use_ascend}")
    print(f"device_id: {context.get_context(attr_key='device_id')}")
    train()

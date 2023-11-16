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
from mindspore import context, nn, ops, jit, set_seed, load_checkpoint, load_param_into_net, data_sink
from mindspore.amp import all_finite
from mindflow.cell import FCSequential
from mindflow.utils import load_yaml_config

from src import create_train_dataset, create_test_dataset, calculate_l2_error, NavierStokesRANS
from eval import predict


set_seed(0)
np.random.seed(0)


def parse_args():
    "parse command line arguments"
    parser = argparse.ArgumentParser(description="cylinder flow train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./configs/rans.yaml")
    args_ret = parser.parse_args()
    return args_ret



def train(input_args):
    '''Train and evaluate the network'''
    # load configurations
    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optim_params = config["optimizer"]
    summary_params = config["summary"]

    # create training dataset
    dataset = create_train_dataset(data_params["data_path"], data_params["batch_size"],)

    # create test dataset
    inputs, label = create_test_dataset(data_params["data_path"])

    model = FCSequential(in_channels=model_params["in_channels"],
                         out_channels=model_params["out_channels"],
                         layers=model_params["layers"],
                         neurons=model_params["neurons"],
                         residual=model_params["residual"],
                         act='tanh')

    if summary_params["load_ckpt"]:
        param_dict = load_checkpoint(summary_params["load_ckpt_path"])
        load_param_into_net(model, param_dict)
    if not os.path.exists(os.path.abspath(summary_params['ckpt_path'])):
        os.makedirs(os.path.abspath(summary_params['ckpt_path']))

    params = model.trainable_params()
    optimizer = nn.Adam(params, optim_params["initial_lr"], weight_decay=optim_params["weight_decay"])
    problem = NavierStokesRANS(model)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    def forward_fn(pde_data, data, label):
        loss = problem.get_loss(pde_data, data, label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, data, label):
        loss, grads = grad_fn(pde_data, data, label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            is_finite = all_finite(grads)
            if is_finite:
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, optimizer(grads))
            loss_scaler.adjust(is_finite)
        else:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    epochs = optim_params["train_epochs"]
    sink_process = data_sink(train_step, dataset, sink_size=1)
    train_data_size = dataset.get_dataset_size()

    for epoch in range(1, 1 + epochs):
        # train
        time_beg = time.time()
        model.set_train(True)
        for _ in range(train_data_size + 1):
            step_train_loss = sink_process()
        print(f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg)*1000 :.3f}ms")
        model.set_train(False)
        if epoch % summary_params["eval_interval_epochs"] == 0:
            # eval
            calculate_l2_error(model, inputs, label, config)
            predict(model=model, epochs=epoch, input_data=inputs, label=label, path=summary_params["visual_dir"])
        if epoch % summary_params["save_checkpoint_epochs"] == 0:
            ckpt_name = "rans_{}.ckpt".format(epoch + 1)
            mindspore.save_checkpoint(model, os.path.join(summary_params['ckpt_path'], ckpt_name))


if __name__ == '__main__':
    print("pid:", os.getpid())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    start_time = time.time()
    train(args)
    print(f"End-to-End total time: {time.time() - start_time} s")

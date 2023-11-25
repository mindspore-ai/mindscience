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
"""Lstm train"""
import os
import time
import argparse

import numpy as np

from mindspore import nn, ops, context, save_checkpoint, set_seed, jit, data_sink
from mindflow.utils import load_yaml_config
from src import create_lstm_dataset, Lstm, plot_train_loss
from cae_eval import cae_eval

np.random.seed(0)
set_seed(0)


def lstm_train():
    """Lstm train process"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    if args.case == 'sod' or args.case == 'shu_osher':
        data_params = config["1D_lstm_data"]
        model_params = config["1D_lstm_model"]
        optimizer_params = config["1D_lstm_optimizer"]
    else:
        data_params = config["2D_lstm_data"]
        model_params = config["2D_lstm_model"]
        optimizer_params = config["2D_lstm_optimizer"]

    # prepare summary file
    summary_dir = optimizer_params["summary_dir"]
    ckpt_dir = os.path.join(summary_dir, 'ckpt')

    # prepare model
    lstm = Lstm(model_params["latent_size"], model_params["hidden_size"], model_params["num_layers"])
    loss_fn = nn.MSELoss()
    lstm_opt = nn.Adam(lstm.trainable_params(), optimizer_params["lr"], weight_decay=optimizer_params["weight_decay"])

    # Define forward function
    def forward_fn(data, label):
        logits = lstm(data)
        loss = loss_fn(logits, label)
        return loss

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, lstm_opt.parameters, has_aux=False)

    @jit
    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        loss = ops.depend(loss, lstm_opt(grads))
        return loss

    # prepare dataset
    latent_true = cae_eval(args.config_file_path, args.case)
    lstm_dataset, _ = create_lstm_dataset(latent_true, data_params["batch_size"], data_params["time_size"],
                                          data_params["latent_size"], data_params["time_window"],
                                          data_params["gaussian_filter_sigma"])

    # data sink
    sink_process = data_sink(train_step, lstm_dataset, sink_size=1)
    train_data_size = lstm_dataset.get_dataset_size()

    print(f"====================Start Lstm train=======================")
    train_loss = []
    for epoch in range(1, optimizer_params["epochs"] + 1):
        local_time_beg = time.time()
        lstm.set_train()
        epoch_train_loss = 0
        for _ in range(train_data_size):
            epoch_train_loss = ops.squeeze(sink_process(), axis=())
        train_loss.append(epoch_train_loss)
        print(f"epoch: {epoch} train loss: {epoch_train_loss} epoch time: {time.time() - local_time_beg:.2f}s")

        if epoch % optimizer_params["save_ckpt_interval"] == 0:
            save_checkpoint(lstm, f"{ckpt_dir}/lstm_{epoch}.ckpt")
    print(f"=====================End Lstm train========================")
    plot_train_loss(train_loss, summary_dir, optimizer_params["epochs"], "lstm")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LstmNet')
    parser.add_argument("--case", type=str, default="sod",
                        choices=["sod", "shu_osher", "riemann", "kh", "cylinder"],
                        help="Which case to run, support 'sod', 'shu_osher', 'riemann', 'kh', 'cylinder")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print(f"pid: {os.getpid()}")
    lstm_train()

# ============================================================================
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
"""train"""
import os
import time
import argparse

import yaml
import numpy as np

from mindspore import nn, ops, context, save_checkpoint, set_seed, jit, data_sink
from src import create_caetransformer_dataset, CaeInformer
from eval import cae_transformer_prediction, cae_transformer_eval


np.random.seed(0)
set_seed(0)


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="CAE-Transformer for 2D cylinder flow")
    parser.add_argument(
        "--mode",
        type=str,
        default="PYNATIVE",
        choices=["GRAPH", "PYNATIVE"],
        help="Context mode, support 'GRAPH', 'PYNATIVE'",
    )
    parser.add_argument(
        "--save_graphs",
        type=bool,
        default=False,
        choices=[True, False],
        help="Whether to save intermediate compilation graphs",
    )
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        choices=["GPU", "Ascend"],
        help="The target device to run, support 'Ascend', 'GPU'",
    )
    parser.add_argument(
        "--device_id", type=int, default=0, help="ID of the target device"
    )
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    input_args = parser.parse_args()
    return input_args


def train():
    """train process"""
    # prepare params
    with open(args.config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    data_params = config["data"]
    model_params = config["cae_transformer"]
    optimizer_params = config["optimizer"]

    # prepare summary file
    summary_dir = optimizer_params["summary_dir"]
    ckpt_dir = os.path.join(summary_dir, "ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # prepare model
    model = CaeInformer(**model_params)
    loss_fn = nn.MSELoss()
    optimizer = nn.AdamWeightDecay(
        model.trainable_params(),
        optimizer_params["lr"],
        weight_decay=optimizer_params["weight_decay"],
    )

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    # prepare dataset
    train_dataset, eval_dataset = create_caetransformer_dataset(
        data_params['data_path'],
        data_params["batch_size"],
        data_params["seq_len"],
        data_params["pred_len"],
    )

    # data sink
    sink_process = data_sink(train_step, train_dataset, sink_size=1)
    train_data_size = train_dataset.get_dataset_size()
    print(f"====================Start cae transformer train=====================")
    train_loss = []
    model.set_train()
    for epoch in range(1, optimizer_params["epochs"] + 1):
        local_time_beg = time.time()
        epoch_train_loss = 0
        model.set_train(True)
        for _ in range(train_data_size):
            epoch_train_loss = ops.squeeze(sink_process(), axis=())
        train_loss.append(epoch_train_loss)
        print(f"epoch: {epoch} train loss: {epoch_train_loss} epoch time: {time.time() - local_time_beg:.2f}s")

        if epoch % optimizer_params["save_ckpt_interval"] == 0:
            save_checkpoint(model, f"{ckpt_dir}/model_{epoch}.ckpt")
        if epoch % optimizer_params["eval_interval"] == 0:
            model.set_train(False)
            cae_transformer_eval(model, eval_dataset, data_params)

    print(f"====================End cae transformer train=======================")
    cae_transformer_prediction(args)


if __name__ == "__main__":
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    train()

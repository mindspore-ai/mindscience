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
"""
train
"""
import os
import time
import argparse
import datetime
import numpy as np

import mindspore
from mindspore import nn, context, ops, Tensor, set_seed
from mindspore.nn.loss import MSELoss

from mindflow.cell import KNO1D
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config

from src.dataset import create_training_dataset
from src.trainer import BurgersWithLoss
from src.utils import visual

parser = argparse.ArgumentParser(description='Burgers 1D problem')
parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                    help="Context mode, support 'GRAPH', 'PYNATIVE'")
parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                    help="Whether to save intermediate compilation graphs")
parser.add_argument("--save_graphs_path", type=str, default="./graphs")
parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                    help="The target device to run, support 'Ascend', 'GPU'")
parser.add_argument("--device_id", type=int, default=1, help="ID of the target device")
parser.add_argument("--config_file_path", type=str, default="./burgers1d.yaml")
args = parser.parse_args()

set_seed(0)
np.random.seed(0)

context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                    save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                    device_target=args.device_target, device_id=args.device_id)
use_ascend = context.get_context(attr_key='device_target') == "Ascend"


def main():
    '''Train and evaluate the network'''
    config = load_yaml_config('burgers1d.yaml')
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    # create training dataset
    train_dataset = create_training_dataset(data_params, shuffle=True)

    # create test dataset
    eval_dataset = create_training_dataset(
        data_params, shuffle=False, is_train=False)

    model = KNO1D(in_channels=data_params['in_channels'],
                  channels=model_params['channels'],
                  modes=model_params['modes'],
                  depths=model_params['depths'],
                  resolution=model_params['resolution']
                  )

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}:{v}")
    model_name = "_".join(model_params_list)
    print(model_name)

    train_size = train_dataset.get_dataset_size()
    eval_size = eval_dataset.get_dataset_size()

    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["lr"],
                                        last_epoch=optimizer_params["epochs"],
                                        steps_per_epoch=train_size,
                                        warmup_epochs=1)
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=Tensor(lr),
                                   weight_decay=optimizer_params["weight_decay"])
    model.set_train()
    loss_fn = MSELoss()
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None
    problem = BurgersWithLoss(model, data_params["out_channels"], loss_fn)

    def forward_fn(inputs, labels):
        loss, l_recons, l_pred = problem.get_loss(inputs, labels)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss, l_recons, l_pred

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(inputs, labels):
        (loss, l_recons, l_pred), grads = grad_fn(inputs, labels)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss, l_recons, l_pred

    def eval_step(inputs, labels):
        return problem.get_rel_loss(inputs, labels)

    train_sink = mindspore.data_sink(train_step, train_dataset, sink_size=1)
    eval_sink = mindspore.data_sink(eval_step, eval_dataset, sink_size=1)

    summary_dir = os.path.join(config["summary_dir"], model_name)
    os.makedirs(summary_dir, exist_ok=True)
    print(summary_dir)

    for epoch in range(1, optimizer_params["epochs"] + 1):
        time_beg = time.time()
        l_recons_train = 0.0
        l_pred_train = 0.0
        for _ in range(train_size):
            loss, l_recons, l_pred = train_sink()
            l_recons_train += l_recons.asnumpy()
            l_pred_train += l_pred.asnumpy()
        l_recons_train = l_recons_train / train_size
        l_pred_train = l_pred_train / train_size
        print(f"epoch: {epoch} epoch time: {(time.time() - time_beg):>8f}s,"
              f" recons loss: {l_recons_train:>8f}, pred loss: {l_pred_train:>8f}, Total loss: {loss:>8f}")

        if epoch % config['eval_interval'] == 0:
            l_recons_eval = 0.0
            l_pred_eval = 0.0
            print("---------------------------start evaluation-------------------------")
            for _ in range(eval_size):
                l_recons, l_pred = eval_sink()
                l_recons_eval += l_recons.asnumpy()
                l_pred_eval += l_pred.asnumpy()
            l_recons_eval = l_recons_eval / eval_size
            l_pred_eval = l_pred_eval / eval_size
            print(f'Eval epoch: {epoch}, recons loss: {l_recons_eval},'
                  f' relative pred loss: {l_pred_eval}')
            print("---------------------------end evaluation---------------------------")
            mindspore.save_checkpoint(model, ckpt_file_name=summary_dir + '/save_model.ckpt')

    # Infer and plot some data.
    inputs = np.load(os.path.join(data_params["path"], "test/inputs.npy"))  # (200,1024,1)
    problem = BurgersWithLoss(model, 10, loss_fn)
    visual(problem, inputs)


if __name__ == '__main__':
    print("pid:", os.getpid())
    print(datetime.datetime.now())

    main()

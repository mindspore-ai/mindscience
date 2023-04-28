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
import datetime
import argparse
import numpy as np

import mindspore
from mindspore import nn, context, ops, Tensor, set_seed, dtype
from mindspore.nn.loss import MSELoss

from mindflow.cell import KNO2D
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config

from src.dataset import create_training_dataset
from src.trainer import NavierStokesWithLoss
from src.utils import visual

parser = argparse.ArgumentParser(description='Navier Stokes problem')
parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                    help="Context mode, support 'GRAPH', 'PYNATIVE'")
parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                    help="Whether to save intermediate compilation graphs")
parser.add_argument("--save_graphs_path", type=str, default="./graphs")
parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                    help="The target device to run, support 'Ascend', 'GPU'")
parser.add_argument("--device_id", type=int, default=1, help="ID of the target device")
parser.add_argument("--config_file_path", type=str, default="./navier_stokes_2d.yaml")
args = parser.parse_args()

set_seed(0)
np.random.seed(0)

context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                    save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                    device_target=args.device_target, device_id=args.device_id)
use_ascend = context.get_context(attr_key='device_target') == "Ascend"


def main():
    '''Train and evaluate the network'''
    config = load_yaml_config('navier_stokes_2d.yaml')
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    # create training dataset
    train_dataset = create_training_dataset(data_params, shuffle=True)
    test_input = np.load(os.path.join(data_params["path"], "test/inputs.npy"))
    test_label = np.load(os.path.join(data_params["path"], "test/label.npy"))
    print('test_input: ', test_input.shape)
    print('test_label: ', test_label.shape)

    model = KNO2D(in_channels=data_params['in_channels'],
                  channels=model_params['channels'],
                  modes=model_params['modes'],
                  depths=model_params['depths'],
                  resolution=model_params['resolution'],
                  compute_dtype=dtype.float16 if use_ascend else dtype.float32
                  )

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}:{v}")
    model_name = "_".join(model_params_list)
    print(model_name)

    train_size = train_dataset.get_dataset_size()

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
    problem = NavierStokesWithLoss(model, data_params["out_channels"], loss_fn, data_format="NHWTC")

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

    train_sink = mindspore.data_sink(train_step, train_dataset, sink_size=1)

    summary_dir = os.path.join(config["summary_dir"], model_name)
    os.makedirs(summary_dir, exist_ok=True)
    print(summary_dir)

    for epoch in range(1, optimizer_params["epochs"] + 1):
        time_beg = time.time()
        train_recons_full = 0.0
        train_pred_full = 0.0
        for _ in range(train_size):
            loss_full, l_recons, l_pred = train_sink()
            train_recons_full += l_recons.asnumpy()
            train_pred_full += l_pred.asnumpy()
        train_recons_full = train_recons_full / train_size
        train_pred_full = train_pred_full / train_size
        print(f"epoch: {epoch}, time cost: {(time.time() - time_beg):>8f}s,"
              f" recons loss: {train_recons_full:>8f}, pred loss: {train_pred_full:>8f}, Total loss: {loss_full:>8f}")

        if epoch % config['eval_interval'] == 0:
            l_recons_all, l_pred_all = problem.test(test_input, test_label)
            print(f'Eval epoch: {epoch}, recons loss: {l_recons_all},'
                  f' relative pred loss: {l_pred_all}')
            mindspore.save_checkpoint(model, ckpt_file_name=summary_dir + '/save_model.ckpt')

    # Infer and plot some data.
    visual(problem, test_input, test_label, t_out=10)


if __name__ == '__main__':
    print("pid:", os.getpid())
    print(datetime.datetime.now())

    main()

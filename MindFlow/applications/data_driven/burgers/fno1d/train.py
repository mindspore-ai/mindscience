# Copyright 2021 Huawei Technologies Co., Ltd
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
"""train"""
import os
import time
import argparse
import datetime
import numpy as np

from mindspore import context, nn, Tensor, set_seed, ops, data_sink, jit, save_checkpoint
from mindspore import dtype as mstype
from mindflow import FNO1D, RelativeRMSELoss, load_yaml_config, get_warmup_cosine_annealing_lr
from mindflow.pde import UnsteadyFlowWithLoss
from mindflow.utils import log_config, print_log

from src import create_training_dataset

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Burgers 1D problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/fno1d.yaml")
    input_args = parser.parse_args()
    return input_args


def train(input_args):
    '''Train and evaluate the network'''
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    # create training dataset
    train_dataset = create_training_dataset(data_params, model_params, shuffle=True)

    # create test dataset
    test_input, test_label = np.load(os.path.join(data_params["root_dir"], "test/inputs.npy")), \
        np.load(os.path.join(data_params["root_dir"], "test/label.npy"))
    test_input = Tensor(np.expand_dims(test_input, -2), mstype.float32)
    test_label = Tensor(np.expand_dims(test_label, -2), mstype.float32)

    model = FNO1D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
                  n_modes=model_params["modes"],
                  resolutions=model_params["resolutions"],
                  hidden_channels=model_params["hidden_channels"],
                  n_layers=model_params["depths"],
                  projection_channels=4*model_params["hidden_channels"],
                  )

    steps_per_epoch = train_dataset.get_dataset_size()
    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["learning_rate"],
                                        last_epoch=optimizer_params["epochs"],
                                        steps_per_epoch=steps_per_epoch,
                                        warmup_epochs=1)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O1')
    else:
        loss_scaler = None

    problem = UnsteadyFlowWithLoss(
        model, loss_fn=RelativeRMSELoss(), data_format="NHWTC")

    summary_dir = config["summary"]["summary_dir"]
    print_log(summary_dir)

    def forward_fn(data, label):
        loss = problem.get_loss(data, label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    sink_process = data_sink(train_step, train_dataset, 1)
    ckpt_dir = "./checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(1, optimizer_params["epochs"] + 1):
        model.set_train()
        local_time_beg = time.time()
        for _ in range(steps_per_epoch):
            cur_loss = sink_process()
        print_log(
            f"epoch: {epoch} train loss: {cur_loss.asnumpy():.8f}"\
            f" epoch time: {time.time() - local_time_beg:.2f}s"\
            f" step time: {(time.time() - local_time_beg)/steps_per_epoch:.4f}s")

        if epoch % config['summary']['test_interval'] == 0:
            eval_time_start = time.time()
            model.set_train(False)
            print_log(
                "================================Start Evaluation================================")
            rms_error = problem.get_loss(
                test_input, test_label)/test_input.shape[0]
            print_log(f"mean rms_error: {rms_error}")
            print_log(
                "=================================End Evaluation=================================")
            print_log(f'evaluation time: {time.time() - eval_time_start}s')
            save_checkpoint(model, os.path.join(
                ckpt_dir, f"{model_params['name']}_epoch{epoch}"))


if __name__ == '__main__':
    log_config('./logs', 'fno1d')
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    train(args)

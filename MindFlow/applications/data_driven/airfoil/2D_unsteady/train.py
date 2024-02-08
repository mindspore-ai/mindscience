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
from mindspore import nn, Tensor, context, ops, jit, set_seed, data_sink, save_checkpoint
from mindspore import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.loss import RelativeRMSELoss
from mindflow.utils import load_yaml_config, log_config, print_log

from src import Trainer, init_dataset, init_model, plt_log, check_file_path, count_params

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Airfoil 2D_unsteady Simulation')
    parser.add_argument("--exp_name", type=str, default="Exp01", help="name of current experiment")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Support mode: 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--backbone", type=str, default="fno2d", choices=["fno2d", "unet2d"],
                        help="Support model backbone: 'fno2d', 'unet2d'")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument('--device_id', type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config/2D_unsteady.yaml")
    input_args = parser.parse_args()
    return input_args


def train(input_args):
    '''train and test the network'''

    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]

    train_dataset, test_dataset, means, stds = init_dataset(data_params)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite, auto_mixed_precision
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        compute_dtype = mstype.float16
        model = init_model(input_args.backbone, data_params, model_params, compute_dtype=compute_dtype)
        auto_mixed_precision(model, optimizer_params["amp_level"][input_args.backbone])
    else:
        loss_scaler = None
        compute_dtype = mstype.float32
        model = init_model(input_args.backbone, data_params, model_params, compute_dtype=compute_dtype)

    if model_params['load_ckpt']:
        param_dict = load_checkpoint(summary_params['pretrained_ckpt_dir'])
        load_param_into_net(model, param_dict)
        print_log("Load pre-trained model successfully")

    loss_fn = RelativeRMSELoss()
    summary_dir = os.path.join(summary_params["summary_dir"], input_args.exp_name, input_args.backbone)
    ckpt_dir = os.path.join(summary_dir, "ckpt_dir")
    check_file_path(ckpt_dir)
    check_file_path(os.path.join(ckpt_dir, 'img'))
    print_log('model parameter count:', count_params(model.trainable_params()))
    print_log(
        f'learing rate: {optimizer_params["lr"][input_args.backbone]}, \
        T_in: {data_params["T_in"]}, T_out: {data_params["T_out"]}')
    steps_per_epoch = train_dataset.get_dataset_size()

    lr = get_warmup_cosine_annealing_lr(optimizer_params["lr"][input_args.backbone], steps_per_epoch,
                                        optimizer_params["epochs"], optimizer_params["warm_up_epochs"])
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=Tensor(lr),
                                   weight_decay=optimizer_params["weight_decay"])

    trainer = Trainer(model, data_params, loss_fn, means, stds)

    def forward_fn(inputs, labels):
        loss, _, _, _ = trainer.get_loss(inputs, labels)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(inputs, labels):
        loss, grads = grad_fn(inputs, labels)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
        loss_new = ops.depend(loss, optimizer(grads))
        return loss_new

    def test_step(inputs, labels):
        return trainer.get_loss(inputs, labels)

    train_size = train_dataset.get_dataset_size()
    test_size = test_dataset.get_dataset_size()
    train_sink = data_sink(train_step, train_dataset, sink_size=1)
    test_sink = data_sink(test_step, test_dataset, sink_size=1)
    test_interval = summary_params["test_interval"]
    save_ckpt_interval = summary_params["save_ckpt_interval"]

    for epoch in range(1, optimizer_params["epochs"] + 1):
        time_beg = time.time()
        train_l2_step = 0.0
        model.set_train()
        for _ in range(1, train_size + 1):
            loss = train_sink()
            train_l2_step += loss.asnumpy()
        train_l2_step = train_l2_step / train_size / data_params["T_out"]
        print_log(
            f"epoch: {epoch}, step time: {(time.time() - time_beg) / steps_per_epoch:>7f},\
            train loss: {train_l2_step:>7f}")

        if epoch % test_interval == 0:
            model.set_train(False)
            test_l2_by_step = [0.0 for _ in range(data_params["T_out"])]
            print_log("---------------------------start test-------------------------")
            for _ in range(test_size):
                _, pred, truth, step_losses = test_sink()
                for i in range(data_params["T_out"]):
                    test_l2_by_step[i] += step_losses[i].asnumpy()
            test_l2_by_step = [error / test_size for error in test_l2_by_step]
            test_l2_step = np.mean(test_l2_by_step)
            print_log(f' test epoch: {epoch}, test loss: {test_l2_step}')
            print_log("---------------------------end test---------------------------")

            plt_log(predicts=pred.asnumpy(),
                    labels=truth.asnumpy(),
                    img_dir=os.path.join(ckpt_dir, 'img'),
                    epoch=epoch
                    )

        if epoch % save_ckpt_interval == 0:
            save_checkpoint(model, ckpt_file_name=os.path.join(ckpt_dir, 'airfoil2D_unsteady.ckpt'))


if __name__ == '__main__':
    args = parse_args()
    log_config('./logs', args.backbone)
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    train(args)

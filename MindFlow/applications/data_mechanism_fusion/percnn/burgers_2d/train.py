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
import argparse
import os
import time

import numpy as np
from mindspore import context, jit, nn, ops, save_checkpoint, set_seed
import mindspore.common.dtype as mstype
from mindflow.utils import load_yaml_config, print_log, log_config, log_timer
from src import RecurrentCNNCell, RecurrentCNNCellBurgers, Trainer, UpScaler, post_process

set_seed(123456)
np.random.seed(123456)


def train_stage(trainer, stage, pattern, config, ckpt_dir, use_ascend):
    """train stage"""
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite
        loss_scaler = DynamicLossScaler(2**10, 2, 100)

    if 'milestone_num' in config.keys():
        milestone = list([(config['epochs']//config['milestone_num'])*(i + 1)
                          for i in range(config['milestone_num'])])
        learning_rate = config['learning_rate']
        lr = float(config['learning_rate'])*np.array(list([config['gamma']
                                                           ** i for i in range(config['milestone_num'])]))
        learning_rate = nn.piecewise_constant_lr(milestone, list(lr))
    else:
        learning_rate = config['learning_rate']

    if stage == 'pretrain':
        params = trainer.upconv.trainable_params()
    else:
        params = trainer.upconv.trainable_params() + trainer.recurrent_cnn.trainable_params()

    optimizer = nn.Adam(params, learning_rate=learning_rate)

    def forward_fn():
        if stage == 'pretrain':
            loss = trainer.get_ic_loss()
        else:
            loss = trainer.get_loss()
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    if stage == 'pretrain':
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=False)
    else:
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=True)

    @jit
    def train_step():
        loss, grads = grad_fn()
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

    best_loss = 100000
    for epoch in range(1, 1 + config['epochs']):
        time_beg = time.time()
        trainer.upconv.set_train(True)
        trainer.recurrent_cnn.set_train(True)
        if stage == 'pretrain':
            step_train_loss = train_step()
            print_log(
                f"epoch: {epoch} train loss: {step_train_loss} \
                    epoch time: {(time.time() - time_beg)*1000 :5.3f}ms \
                    step time: {(time.time() - time_beg)*1000 :5.3f}ms")
        else:
            step_train_loss, loss_data, loss_ic, loss_phy, loss_valid = train_step()
            print_log(f"epoch: {epoch} train loss: {step_train_loss} ic_loss: {loss_ic} data_loss: {loss_data} "
                      f"val_loss: {loss_valid} phy_loss: {loss_phy} "
                      f"epoch time: {(time.time() - time_beg)*1000 :5.3f}ms "
                      f"step time: {(time.time() - time_beg)*1000 :5.3f}ms ")
            if step_train_loss < best_loss:
                best_loss = step_train_loss
                print_log('best loss', best_loss, 'save model')
                save_checkpoint(trainer.upconv, os.path.join(ckpt_dir, f"{pattern}_{config['name']}_upconv.ckpt"))
                save_checkpoint(trainer.recurrent_cnn,
                                os.path.join(ckpt_dir, f"{pattern}_{config['name']}_recurrent_cnn.ckpt"))
    if pattern == 'physics_driven':
        trainer.recurrent_cnn.show_coef()


def parse_args():
    """parse input args"""
    parser = argparse.ArgumentParser(description="burgers train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/data_driven_percnn_burgers.yaml")
    input_args = parser.parse_args()
    return input_args


@log_timer
def train(input_args):
    """train"""
    burgers_config = load_yaml_config(input_args.config_file_path)

    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    if use_ascend:
        compute_dtype = mstype.float16
    else:
        compute_dtype = mstype.float32

    data_config = burgers_config['data']
    optimizer_config = burgers_config['optimizer']
    model_config = burgers_config['model']
    summary_config = burgers_config['summary']

    upconv = UpScaler(in_channels=model_config['in_channels'],
                      out_channels=model_config['out_channels'],
                      hidden_channels=model_config['upscaler_hidden_channels'],
                      kernel_size=model_config['kernel_size'],
                      stride=model_config['stride'],
                      has_bais=True)

    if use_ascend:
        from mindspore.amp import auto_mixed_precision
        auto_mixed_precision(upconv, 'O1')

    pattern = data_config['pattern']
    if pattern == 'data_driven':
        recurrent_cnn = RecurrentCNNCell(input_channels=model_config['in_channels'],
                                         hidden_channels=model_config['rcnn_hidden_channels'],
                                         kernel_size=model_config['kernel_size'],
                                         compute_dtype=compute_dtype)
    else:
        recurrent_cnn = RecurrentCNNCellBurgers(kernel_size=model_config['kernel_size'],
                                                init_coef=model_config['init_coef'],
                                                compute_dtype=compute_dtype)

    percnn_trainer = Trainer(upconv=upconv,
                             recurrent_cnn=recurrent_cnn,
                             timesteps_for_train=data_config['rollout_steps'],
                             dx=data_config['dx'],
                             dt=data_config['dy'],
                             nu=data_config['nu'],
                             data_path=os.path.join(data_config['root_dir'], data_config['file_name']),
                             compute_dtype=compute_dtype)

    ckpt_dir = os.path.join(summary_config["root_dir"], summary_config['ckpt_dir'])
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    train_stage(percnn_trainer, 'pretrain', pattern, optimizer_config['pretrain'], ckpt_dir, use_ascend)
    train_stage(percnn_trainer, 'finetune', pattern, optimizer_config['finetune'], ckpt_dir, use_ascend)
    post_process(percnn_trainer, pattern)


if __name__ == '__main__':
    log_config('./logs', 'percnn')
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id,
                        max_call_depth=99999999)
    print_log("pid:", os.getpid())
    print_log(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    train(args)

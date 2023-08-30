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
import argparse
import time
import numpy as np

from mindspore import ops, context, nn, set_seed, save_checkpoint, jit
import mindspore.common.dtype as mstype
from mindflow.utils import load_yaml_config, print_log, log_config

from src import RecurrentCNNCell, UpScaler, RecurrentCNNCellBurgers
from src import Trainer
from src import post_process

set_seed(123456)
np.random.seed(123456)


def train_stage(trainer, stage, pattern, config):
    """train stage"""
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
        params = trainer.upconv.trainable_params(
        ) + trainer.recurrent_cnn.trainable_params()

    optimizer = nn.Adam(params, learning_rate=learning_rate)

    def forward_fn():
        if stage == 'pretrain':
            return trainer.get_ic_loss()
        return trainer.get_loss()

    if stage == 'pretrain':
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=False)
    else:
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=True)

    @ jit
    def train_step():
        res, grads = grad_fn()
        res = ops.depend(res, optimizer(grads))
        return res

    best_loss = 100000
    for epoch in range(1, 1 + config['epochs']):
        time_beg = time.time()
        trainer.upconv.set_train(True)
        trainer.recurrent_cnn.set_train(True)
        if stage == 'pretrain':
            step_train_loss = train_step()
            print_log(
                f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) :.3f} s")
        else:
            step_train_loss, loss_data, loss_ic, loss_phy, loss_valid = train_step()
            print_log(f"epoch: {epoch} train loss: {step_train_loss} ic_loss: {loss_ic} data_loss: {loss_data}\
                   val_loss: {loss_valid} phy_loss: {loss_phy} epoch time: {(time.time() - time_beg): .3f} s")
            if step_train_loss < best_loss:
                best_loss = step_train_loss
                print_log('best loss', best_loss, 'save model')
                save_checkpoint(trainer.upconv, os.path.join("./model", pattern,
                                                             f"{config['name_conf']}_upconv.ckpt"))
                save_checkpoint(trainer.recurrent_cnn, os.path.join("./model", pattern,
                                                                    f"{config['name_conf']}_recurrent_cnn.ckpt"))
    if pattern == 'physics_driven':
        trainer.recurrent_cnn.show_coef()


def parse_args():
    """parse input args"""
    parser = argparse.ArgumentParser(description="burgers train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=2,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./percnn_burgers.yaml")
    parser.add_argument("--pattern", type=str, default="data_driven")
    input_args = parser.parse_args()
    return input_args


def train(input_args):
    """train"""
    pattern = input_args.pattern
    burgers_config = load_yaml_config(input_args.config_file_path)[pattern]

    upconv = UpScaler(in_channels=burgers_config['input_channel'],
                      out_channels=burgers_config['out_channels'],
                      hidden_channels=burgers_config['upscaler_hidden_channel'],
                      kernel_size=burgers_config['kernel_size'],
                      stride=burgers_config['stride'],
                      has_bais=True)

    if pattern == 'data_driven':
        recurrent_cnn = RecurrentCNNCell(input_channels=burgers_config['input_channel'],
                                         hidden_channels=burgers_config['rcnn_hidden_channel'],
                                         kernel_size=burgers_config['kernel_size'],
                                         compute_dtype=mstype.float32)
    else:
        recurrent_cnn = RecurrentCNNCellBurgers(kernel_size=burgers_config['kernel_size'],
                                                init_coef=burgers_config['init_coef'],
                                                compute_dtype=mstype.float32)

    percnn_trainer = Trainer(upconv=upconv,
                             recurrent_cnn=recurrent_cnn,
                             timesteps_for_train=burgers_config['use_timestep'],
                             dx=burgers_config['dx'],
                             dt=burgers_config['dy'],
                             nu=burgers_config['nu'],
                             dataset_path=burgers_config['dataset_path'],
                             compute_dtype=mstype.float32)

    train_stage(percnn_trainer, 'pretrain', pattern, burgers_config['pretrain'])
    train_stage(percnn_trainer, 'finetune', pattern, burgers_config['finetune'])
    post_process(percnn_trainer)


if __name__ == '__main__':
    log_config('./logs', 'percnn')
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id,
                        max_call_depth=99999999)
    print_log("pid:", os.getpid())
    print_log(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    train(args)

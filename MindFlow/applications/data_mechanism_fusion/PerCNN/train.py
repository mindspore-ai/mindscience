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
import time
import numpy as np

from mindspore import ops, context, nn, set_seed, save_checkpoint, jit, load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindflow.utils import load_yaml_config

from src import RCNN
from src import Trainer
from src import post_process_v2

set_seed(123456)
np.random.seed(123456)

parser = argparse.ArgumentParser(description="burgers train")
parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                    help="Running in GRAPH_MODE OR PYNATIVE_MODE")
parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                    help="Whether to save intermediate compilation graphs")
parser.add_argument("--save_graphs_path", type=str, default="./graphs")
parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                    help="The target device to run, support 'Ascend', 'GPU'")
parser.add_argument("--device_id", type=int, default=1,
                    help="ID of the target device")
parser.add_argument("--config_file_path", type=str, default="./percnn_burgers.yaml")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                    save_graphs=args.save_graphs,
                    save_graphs_path=args.save_graphs_path,
                    device_target=args.device_target,
                    device_id=args.device_id,
                    max_call_depth=99999999)
print(
    f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
use_ascend = context.get_context(attr_key='device_target') == "Ascend"


def pretrain(trainer):
    """pretrain"""
    pretrain_config = load_yaml_config(args.config_file_path)["pretrain"]
    learning_rate = pretrain_config['learning_rate']
    optimizer = nn.Adam(trainer.model.trainable_params(),
                        learning_rate=learning_rate)

    def forward_fn():
        return trainer.get_ic_loss()

    grad_fn = ops.value_and_grad(
        forward_fn, None, trainer.model.trainable_params(), has_aux=False)

    @ jit
    def train_step():
        loss, grads = grad_fn()
        loss = ops.depend(loss, optimizer(grads))
        return loss

    for epoch in range(1, 1 + pretrain_config['epochs']):
        time_beg = time.time()
        model.set_train(True)
        step_train_loss = train_step()
        print(
            f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) :.3f} s")


def train(trainer):
    """train"""
    train_config = load_yaml_config(args.config_file_path)["train"]

    milestone = list([100*(i + 1) for i in range(150)])
    learning_rates = list(
        [train_config['learning_rate']*train_config['gama']**i for i in range(150)])
    step_lr = nn.piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.Adam(trainer.model.trainable_params(),
                        learning_rate=step_lr)
    best_loss = 100000

    def forward_fn():
        return trainer.get_loss()

    grad_fn = ops.value_and_grad(
        forward_fn, None, trainer.model.trainable_params(), has_aux=True)

    @ jit
    def train_step():
        res, grads = grad_fn()
        res = ops.depend(res, optimizer(grads))
        return res

    for epoch in range(1, 1 + train_config['epochs']):
        time_beg = time.time()
        model.set_train(True)
        step_train_loss, loss_data, loss_ic, loss_phy, loss_valid = train_step()
        print(f"epoch: {epoch} train loss: {step_train_loss} ic_loss: {loss_ic} data_loss: {loss_data} \
              val_loss: {loss_valid} phy_loss: {loss_phy} epoch time: {(time.time() - time_beg): .3f} s")
        if step_train_loss < best_loss:
            best_loss = step_train_loss
            print('best loss', best_loss, 'save model')
            save_checkpoint(model, "./model/checkpoint" +
                            train_config['name_conf'] + ".ckpt")


if __name__ == '__main__':
    INPUT_CHANNELS = 2
    HIDDEN_CHANNELS = 8
    INPUT_KERNEL_SIZE = 5
    INFER_STEP = 1800
    effective_step = list(range(0, INFER_STEP + 1))

    model = RCNN(input_channels=INPUT_CHANNELS,
                 hidden_channels=HIDDEN_CHANNELS,
                 input_kernel_size=INPUT_KERNEL_SIZE,
                 infer_step=INFER_STEP,
                 effective_step=effective_step)

    percnn_trainer = Trainer(model,
                             time_steps=INFER_STEP,
                             dx=1.0/100,
                             dt=0.00025,
                             nu=1/200,
                             compute_dtype=mstype.float32)

    pretrain(percnn_trainer)
    train(percnn_trainer)

    config = load_yaml_config(args.config_file_path)
    ckpt_file_name = config["ckpt_file_name"]
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(model, param_dict)

    output, _ = percnn_trainer.model(percnn_trainer.init_state_low)
    output = ops.concat(output, axis=0)
    output = ops.concat((output, output[:, :, :, 0:1]), axis=3)
    output = ops.concat((output, output[:, :, 0:1, :]), axis=2)
    truth_clean = np.concatenate(
        (percnn_trainer.truth_clean, percnn_trainer.truth_clean[:, :, :, 0:1]), axis=3)
    truth_clean = np.concatenate(
        (truth_clean, truth_clean[:, :, 0:1, :]), axis=2)
    low_res = percnn_trainer.truth[:, :, ::2, ::2]

    output, low_res = output.asnumpy(), low_res.asnumpy()
    fig_save_path = config["fig_save_path"]

    print(output.shape, truth_clean.shape, low_res.shape)

    for i in range(0, INFER_STEP + 1, 10):
        err = post_process_v2(output, truth_clean, low_res, xmin=0, xmax=1, ymin=0, ymax=1,
                              num=i, fig_save_path=fig_save_path)
        print(i, err)

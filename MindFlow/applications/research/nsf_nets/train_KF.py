# Copyright 2024 Huawei Technologies Co., Ltd
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
"""NSFNet train"""
# pylint: disable=C0103
import os
import time
import argparse

import numpy as np
import mindspore as ms
from mindspore import set_seed, context, nn
from mindflow.utils import load_yaml_config
from src.network_kf import VPNSFNets
from src.datasets_kf import read_training_data

np.random.seed(123456)
set_seed(123456)

def train(model, Niter, lr):
    """
    Training
    """
    # Get the gradients function
    params_train = model.dnn.trainable_params()
    optimizer_Adam = nn.Adam(params_train, learning_rate=lr)
    grad_fn = ms.value_and_grad(model.loss_fn, None, optimizer_Adam.parameters, has_aux=True)
    model.dnn.set_train()

    start_time = time.time()

    for epoch in range(1, 1+Niter):

        (loss, loss_b, loss_f), grads = grad_fn(model.xb, model.yb, model.x_f, model.y_f, model.ub, model.vb)
        optimizer_Adam(grads)

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Total_loss: %.3e, Loss_b: %.3e, Loss_f: %.3e, Time: %.2f' %\
                    (epoch, loss.item(), loss_b.item(), loss_f.item(), elapsed))

            loss_history_adam_pretrain = np.empty([0])
            loss_b_history_adam_pretrain = np.empty([0])
            loss_f_history_adam_pretrain = np.empty([0])

            loss_history_adam_pretrain = np.append(loss_history_adam_pretrain, loss.numpy())
            loss_b_history_adam_pretrain = np.append(loss_b_history_adam_pretrain, loss_b.numpy())
            loss_f_history_adam_pretrain = np.append(loss_f_history_adam_pretrain, loss_f.numpy())

            start_time = time.time()

    np.save(f'Loss-Coe/train/loss_history_adam_pretrain', loss_history_adam_pretrain)
    np.save(f'Loss-Coe/train/loss_b_history_adam_pretrain', loss_b_history_adam_pretrain)
    np.save(f'Loss-Coe/train/loss_f_history_adam_pretrain', loss_f_history_adam_pretrain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="./config/NSFNet_KF.yaml")
    # args = parser.parse_args()
    args = parser.parse_known_args()[0]

    config = load_yaml_config(args.config_file_path)
    params = config["params"]

    model_name = params['model_name']
    case = params['case']
    device = params['device']
    device_id = params['device_id']
    network_size = params['network_size']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    load_params = params['load_params']
    second_path = params['second_path1']

    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=device)

    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    if use_ascend:
        msfloat_type = ms.float16
        npfloat_type = np.float16
    else:
        msfloat_type = ms.float32
        npfloat_type = np.float32

    xb_train, yb_train, ub_train, vb_train, x_train, y_train = read_training_data()

    model_train = VPNSFNets(xb_train, yb_train, ub_train, vb_train, x_train,
                            y_train, network_size, use_ascend, msfloat_type,
                            npfloat_type, load_params, second_path)
    for epoch_train, lr_train in zip(epochs, learning_rate):
        train(model_train, int(np.float(epoch_train)), lr_train)
    ms.save_checkpoint(model_train.dnn, f'model/{second_path}/model.ckpt')

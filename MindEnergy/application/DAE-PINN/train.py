# Copyright 2025 Huawei Technologies Co., Ltd
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
"""training DAE-PINN"""

import time
import os
import argparse

from mindspore import ops, jit, Tensor
from mindspore import context
from mindspore.experimental import optim
import numpy as np

from mindscience.utils import load_yaml_config

from src.utils import make_config
from src.model import ThreeBusPN
from src.data import get_dataset
from src.trainer import DaeTrainer


def main():
    config = load_yaml_config(args.config_file)
    model_params, data_params, optim_params, ode_params, summary_params = config[
        'model'], config['data'], config['optimizer'], config['ode'], config['summary']
    summary_dir = summary_params['summary_dir']
    os.makedirs(summary_dir, exist_ok=True)

    dynamic, algebraic = make_config(model_params)
    net = ThreeBusPN(
        dynamic,
        algebraic,
        use_input_layer=model_params['use_input_layer'],
        stacked=not model_params['unstacked']
    )

    train_dataset, test_dataset, val_dataset = get_dataset(data_params)

    num_irk_stages = model_params['num_irk_stages']
    # collecting RK data
    data_dir = data_params['data_dir']
    irk_data = np.loadtxt(os.path.join(data_dir, 'IRK_weights', f'Butcher_IRK{num_irk_stages}.txt'),
                          ndmin=2, dtype=np.float32)
    irk_weights = np.reshape(
        irk_data[0:num_irk_stages**2+num_irk_stages], (num_irk_stages+1, num_irk_stages))
    irk_weights = Tensor(irk_weights)
    irk_times = irk_data[num_irk_stages**2 + num_irk_stages:]
    trainer = DaeTrainer(net, irk_weights=irk_weights, irk_times=irk_times, h=ode_params['h'],
                         dyn_weight=model_params['dyn_weight'], alg_weight=model_params['alg_weight'])

    optimizer = optim.Adam(net.trainable_params(), lr=float(optim_params['lr']))
    scheduler_type = optim_params['scheduler_type']
    use_scheduler = optim_params['use_scheduler']
    if use_scheduler:
        if scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=optim_params['patience'],
                factor=optim_params['factor'],
            )
        elif scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=optim_params['patience'], gamma=optim_params['factor']
            )
        else:
            scheduler = None
    else:
        scheduler = None

    def forward_fn(x):
        loss = trainer.get_loss(x)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(x):
        loss, grads = grad_fn(x)
        optimizer(grads)
        return loss
    test_interval = summary_params['test_interval']

    for epoch in range(1, 1 + optim_params['epochs']):
        time_beg = time.time()
        net.set_train(True)
        loss_val = []
        for data, in train_dataset:
            step_train_loss = train_step(data)
            loss_val.append(step_train_loss.numpy())
        time_end = time.time()

        print(
            f"epoch: {epoch} train loss: {np.mean(loss_val)} epoch time: {time_end-time_beg:.3f}s")
        if use_scheduler:
            if scheduler_type == "plateau":
                net.set_train(False)
                loss_val = trainer.get_loss(val_dataset)
                scheduler.step(loss_val)
            else:
                scheduler.step()
        if epoch % test_interval == 0:
            net.set_train(False)
            loss_test = trainer.get_loss(test_dataset)
            print(f'test loss: {loss_test}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dae-pinns-example")
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument('--method', type=str, default='BDF',
                        help="integration method")
    parser.add_argument('--config_file', type=str, default='./configs/config.yaml',
                        help="config yaml path")
    parser.add_argument('--device_target', type=str, default='Ascend',
                        help="device target", choices=['CPU', 'Ascend'])
    parser.add_argument('--mode', type=str, default='PYNATIVE',
                        help="mindspore context mode")
    args = parser.parse_args()
    mode = context.PYNATIVE_MODE if args.mode.lower().startswith('graph') else context.GRAPH_MODE
    context.set_context(mode=mode, device_target=args.device_target,
                        device_id=args.device_id)
    main()

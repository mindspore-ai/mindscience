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
"""Training."""
import os
import time
import argparse
import numpy as np

from src.dataset import create_train_dataset, create_test_dataset
from src.poisson import Poisson
from src.utils import calculate_l2_error, visual

from mindspore import context, save_checkpoint, nn, ops, jit, set_seed
from mindflow import load_yaml_config
from mindflow.cell import MultiScaleFCSequential

set_seed(123456)
np.random.seed(123456)


def train(file_cfg, ckpt_dir, n_epochs):
    """Train a model."""
    # Load config
    config = load_yaml_config(file_cfg)

    # Create the dataset
    ds_train = create_train_dataset(config)
    ds_test = create_test_dataset(config)

    # Create the model
    model = MultiScaleFCSequential(config['model']['in_channels'],
                                   config['model']['out_channels'],
                                   config['model']['layers'],
                                   config['model']['neurons'],
                                   residual=True,
                                   act=config['model']['activation'],
                                   num_scales=config['model']['num_scales'],
                                   amp_factor=1.0,
                                   scale_factor=2.0,
                                   input_scale=[10., 10.],
                                   )
    print(model)

    # Create the problem and optimizer
    problem = Poisson(model)

    params = model.trainable_params() + problem.loss_fn.trainable_params()
    steps_per_epoch = ds_train.get_dataset_size()
    milestone = [int(steps_per_epoch * n_epochs * x) for x in [0.4, 0.6, 0.8]]
    lr_init = config["optimizer"]["initial_lr"]
    learning_rates = [lr_init * (0.1**x) for x in [0, 1, 2]]
    lr_ = nn.piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.Adam(params, learning_rate=lr_)

    # Prepare loss scaler
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None

    def forward_fn(pde_data, bc_data, src_data):
        loss = problem.get_loss(pde_data, bc_data, src_data)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data, src_data):
        loss, grads = grad_fn(pde_data, bc_data, src_data)
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

    def train_epoch(model, dataset, i_epoch):
        local_time_beg = time.time()

        model.set_train()
        for _, (pde_data, bc_data, src_data) in enumerate(dataset):
            loss = train_step(pde_data, bc_data, src_data)

        print(
            f"epoch: {i_epoch} train loss: {float(loss):.8f}" +
            f" epoch time: {time.time() - local_time_beg:.2f}s")

    keep_ckpt_max = config['keep_checkpoint_max']

    for i_epoch in range(1, 1 + n_epochs):
        train_epoch(model, ds_train, i_epoch)

        # Save last checkpoints
        save_name = os.path.join(ckpt_dir, f"epoch-{i_epoch % keep_ckpt_max}.ckpt")
        save_checkpoint(model, save_name)

        if i_epoch % 5 == 1 or i_epoch == n_epochs:
            # Evaluate the model
            calculate_l2_error(model, ds_test)

            # Visual comparison of label and prediction
            visual(model, ds_test, file_name=f"epoch-{i_epoch}_result.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="poisson")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument('--ckpt_dir', default='./')
    parser.add_argument('--n_epochs', default=250, type=int)
    parser.add_argument("--config_file_path", type=str, default="./poisson_cfg.yaml")
    args = parser.parse_args()

    context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
        device_target=args.device_target, device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print(f'pid: {os.getpid()}')
    time_beg = time.time()
    train(args.config_file_path, args.ckpt_dir, args.n_epochs)
    print(f"End-to-End total time: {time.time() - time_beg:.1f} s")

"""Training."""
import os
import time
import argparse
import numpy as np
import sympy

import mindspore as ms
from mindspore import context, save_checkpoint, nn, ops, jit, set_seed

import mindflow as mf
from mindflow import load_yaml_config, print_log, log_timer

from src.model import create_model
from src.lr_scheduler import OneCycleLR
from src.dataset import create_dataset
from src.poisson import Poisson
from src.utils import calculate_l2_error
from src.boundary import get_bc

set_seed(123456)
np.random.seed(123456)


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description="poisson")
    parser.add_argument(
        "--geom_name",
        type=str,
        default="disk",
        choices=[
            "rectangle",
            "disk",
            "triangle",
            "pentagon",
            "polygon",
            "tetrahedron",
            "cylinder",
            "cone",
            "interval",
        ],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="GRAPH",
        choices=["GRAPH", "PYNATIVE"],
        help="Running in GRAPH_MODE OR PYNATIVE_MODE",
    )
    parser.add_argument(
        "--device_target",
        type=str,
        default="GPU",
        choices=["GPU", "Ascend"],
        help="The target device to run, support 'Ascend', 'GPU'",
    )
    parser.add_argument(
        "--device_id", type=int, default=0, help="ID of the target device"
    )
    parser.add_argument("--ckpt_dir", default="./ckpt")
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument(
        "--config_file_path", type=str, default="./configs/poisson_cfg.yaml"
    )
    input_args = parser.parse_args()
    return input_args

def pde(out_vars, in_vars):
    """ define the PDE"""
    poisson = 0
    src_term = 1
    sym_u = out_vars[0]
    for var in in_vars:
        poisson += sympy.diff(sym_u, (var, 2))
        src_term *= sympy.sin(4 * sympy.pi * var)
    poisson += src_term
    equations = {"poisson": poisson}
    return equations

@log_timer
def train(geom_name, file_cfg, ckpt_dir, n_epochs):
    """Train a model."""
    # Load config
    config = load_yaml_config(file_cfg)

    # Create the dataset
    ds_train, n_dim = create_dataset(geom_name, config)

    # Create the model
    model = create_model(**config["model"][f"{n_dim}d"])

    # Create the problem and optimizer
    problem = Poisson(model, n_dim, pde, get_bc(
        config['data']['BC']['BC_type']))
    params = model.trainable_params() + problem.loss_fn.trainable_params()
    steps_per_epoch = config["data"]["domain"]["size"] // config["data"]["train"]["batch_size"]
    learning_rate = OneCycleLR(
        total_steps=steps_per_epoch * n_epochs, **config["optimizer"]
    )
    optimizer = nn.Adam(params, learning_rate=learning_rate)

    # prepare loss scaler
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite, auto_mixed_precision

        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, "O3")
    else:
        loss_scaler = None

    def forward_fn(pde_data, bc_data):
        loss = problem.get_loss(pde_data, bc_data)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    # Create
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data):
        loss, grads = grad_fn(pde_data, bc_data)
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
        steps_per_epochs = dataset.get_dataset_size()
        print_log(f"number of steps_per_epochs: {steps_per_epochs}")
        model.set_train()
        for i_step, (pde_data, bc_data) in enumerate(dataset):
            local_time_beg = time.time()
            loss = train_step(pde_data, bc_data)

            if i_step % 50 == 0 or i_step + 1 == steps_per_epochs:
                local_time_end = time.time()
                epoch_seconds = (local_time_end - local_time_beg) * 1000
                step_seconds = epoch_seconds/steps_per_epochs
                print(f"epoch: {i_epoch} train loss: {float(loss)} "
                      f"epoch time: {epoch_seconds:5.3f}ms step time: {step_seconds:5.3f}ms")

    keep_ckpt_max = config["keep_checkpoint_max"]

    for i_epoch in range(1, 1 + n_epochs):
        train_epoch(model, ds_train, i_epoch)

        # Save last checkpoints
        save_name = os.path.join(
            ckpt_dir, "{}_{}d_{}.ckpt".format(geom_name, n_dim, i_epoch % keep_ckpt_max)
        )
        save_checkpoint(model, save_name)


def test(geom_name, checkpoint, file_cfg, n_samps):
    """Evaluate a model."""
    # Create the dataset
    config = mf.load_yaml_config(file_cfg)
    ds_test_, n_dim_ = create_dataset(geom_name, config, n_samps)

    # Load the model
    model_ = create_model(**config["model"][f"{n_dim_}d"])
    checkpoint = ms.load_checkpoint(checkpoint)
    ms.load_param_into_net(model_, checkpoint)

    # Evaluate the model
    calculate_l2_error(model_, ds_test_, n_dim_)


if __name__ == "__main__":
    print(f"pid: {os.getpid()}")
    args = parse_args()
    context.set_context(
        mode=context.GRAPH_MODE
        if args.mode.upper().startswith("GRAPH")
        else context.PYNATIVE_MODE,
        device_target=args.device_target,
        device_id=args.device_id,
    )
    use_ascend = context.get_context(attr_key="device_target") == "Ascend"
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    train(args.geom_name, args.config_file_path, args.ckpt_dir, args.n_epochs)
    ckpt = os.path.join(args.ckpt_dir, "{}_{}d_{}.ckpt".format(args.geom_name, 2, 1))
    test(args.geom_name, ckpt, args.config_file_path, 5000)

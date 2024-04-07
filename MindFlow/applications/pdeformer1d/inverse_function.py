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
r"""
Recover a function (the source term, or the wave velocity field) of a PDE using
gradient descent based on the pre-trained PDEformer model.
"""
import argparse
import math
from typing import Tuple, Dict, Any
from omegaconf import DictConfig
import numpy as np

import mindspore as ms
from mindspore import dtype as mstype
from mindspore import ops, nn, Tensor, context
from mindspore.common.initializer import initializer, Uniform
from mindspore.amp import DynamicLossScaler, auto_mixed_precision

from src.data.load_inverse_data import get_inverse_data
from src.data.load_inverse_data import inverse_observation
from src.utils.load_yaml import load_config
from src.utils.record import init_record
from src.utils.visual import plot_1d, plot_2dxn
from src.utils.tools import set_seed
from src.core.losses import LossFunction
from src.core.lr_scheduler import get_lr_list
from src.cell import get_model


def parse_args():
    r"""Parse input args."""
    parser = argparse.ArgumentParser(description="pde foundation model")
    parser.add_argument("--mode", type=str, default="GRAPH",
                        choices=["GRAPH", "PYNATIVE"], help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False,
                        choices=[True, False], help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend", "CPU"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument('--distributed', action='store_true',
                        help='enable distributed training (data parallel)')
    parser.add_argument("--config_file_path", type=str,
                        default="configs/config_yzh_grammar.yaml")
    return parser.parse_args()


class InverseProblem:
    r"""
    Using gradient descent to solve the inverse problem of recovering a
    function-valued term in a PDE.

    Args:
        pde_idx (int): The index of the PDE to solve.
        data_tuple (tuple): The model's input and output data packaged in batch
            form have the same PDE structure, but different initial conditions
            (and different source terms for the wave equation).
        inverse_config (dict): The configuration of the inverse problem.
        data_info (dict): A dictionary of data information.
    """

    def __init__(self,
                 pde_idx: int,
                 data_tuple: Tuple[Tensor],
                 inverse_config: DictConfig,
                 data_info: Dict[str, Any]) -> None:
        self.pde_idx = pde_idx
        self.data_tuple = data_tuple
        self.input_tuple = data_tuple[:-1]
        u_label = data_tuple[-1]  # tensor
        node_function = data_tuple[2]
        self.tx_grid_shape = (data_info["n_t_grid"], data_info["n_x_grid"])
        self.pde_latex = data_info["pde_latex"]

        # shape of node_function is [n_graph, num_function, num_points_function, 2]
        self.n_graph, self.num_function, self.num_points_function, _ = node_function.shape
        # [n_graph, n_t_grid * n_x_grid, 2] -> [n_graph, n_t_grid, n_x_grid, 2]
        coordinate_gt = self.input_tuple[-1].asnumpy().reshape(
            (self.n_graph, data_info["n_t_grid"], data_info["n_x_grid"], 2))
        # [n_graph, n_t_grid * n_x_grid, 1] -> [n_graph, n_t_grid, n_x_grid, 1]
        self.u_label = u_label.asnumpy().reshape(
            (self.n_graph, data_info["n_t_grid"], data_info["n_x_grid"], 1))

        recovered = initializer(Uniform(scale=0.01),
                                [1, 1, self.num_points_function, 1], mstype.float32)
        self.recovered = ms.Parameter(recovered, name='function_valued_term')

        # Shape is [n_graph, num_function, num_points_function, 2].
        mask = np.zeros(node_function.shape).astype(np.bool_)
        function_node_id = config.inverse.function_node_id
        mask[:, function_node_id, :, -1] = True
        self.mask = Tensor(mask, dtype=mstype.bool_)  # [n_graph, num_function, num_points_function, 2]

        self.gt_np = node_function.asnumpy()[0, function_node_id, :, -1]  # [num_points_function]

        # add noise and spatial-temporal subsampling
        u_noisy, u_obs_plot, u_obs, coordinate_obs = inverse_observation(
            config.inverse.observation, self.u_label, coordinate_gt)
        self.u_noisy = u_noisy  # [n_graph, n_t_grid, n_x_grid, 1]
        self.u_obs_plot = u_obs_plot  # [n_graph, n_t_grid, n_x_grid, 1]
        # these are the only tensor-valued properties of this class object
        # Shape is [n_graph, n_tx_obs_pts, 1].
        self.u_obs = Tensor(u_obs, dtype=mstype.float32)
        # Shape is [n_graph, n_tx_obs_pts, 2].
        self.coordinate_obs = Tensor(coordinate_obs, dtype=mstype.float32)

        # regularization_loss
        self.regularize_type = inverse_config.function_regularize.type
        self.regularize_weight = inverse_config.function_regularize.weight
        x_coord = node_function[0, function_node_id, :, 0]  # [num_points_function]
        self.delta_x = x_coord[1:] - x_coord[:-1]  # [num_points_function - 1]
        if self.delta_x.min() <= 0:
            raise ValueError("'x_coord' should be strictly increasing.")

    def get_data_tuple(self, is_train: bool = True) -> Tuple[Tensor]:
        r"""
        Get the data tuple for training or testing.

        Args:
            is_train (bool): Whether to get the training data or testing data.

        Returns:
            A data tuple for training or testing.

        """
        (node_type, node_scalar, node_function, in_degree, out_degree,
         attn_bias, spatial_pos, coordinate_gt, _) = self.data_tuple

        recovered_repeat = self.recovered  # [1, 1, num_points_function, 1]
        node_function_ = ops.select(self.mask, recovered_repeat, node_function)

        if is_train:
            coordinate = self.coordinate_obs  # [n_graph, num_obs_point, 2]
        else:
            coordinate = coordinate_gt  # [n_graph, num_point, 2]

        data_tuple = (node_type, node_scalar, node_function_, in_degree, out_degree,
                      attn_bias, spatial_pos, coordinate, self.u_obs)
        return data_tuple

    def regularization_loss(self) -> Tensor:
        r"""Calculate the regularization loss."""
        recovered_s = self.recovered[0, 0, :, 0]  # [num_points_function]
        dx_s = (recovered_s[1:] - recovered_s[:-1]) / self.delta_x
        if self.regularize_type == "L1":
            penalty = ops.mean(ops.abs(dx_s))
        elif self.regularize_type == "squareL2":
            penalty = ops.mean(ops.square(dx_s))
        elif self.regularize_type == "L2":
            penalty = ops.sqrt(ops.mean(ops.square(dx_s)) + 1e-6)
        else:
            raise ValueError(f"Unknown regularize_type '{self.regularize_type}'!")
        return self.regularize_weight * penalty

    def compare(self, enable_plot: bool = False) -> float:
        r"""
        Compare the ground-truth and the recovered function values.

        Args:
            enable_plot (bool): Whether to plot the comparison results.

        Returns:
            The nRMSE between the ground-truth and the recovered function values.
        """
        recovered_np = self.recovered.asnumpy().reshape((self.num_points_function,))

        gt_norm = np.linalg.norm(self.gt_np, ord=2, axis=0, keepdims=False)
        nrmse = np.linalg.norm(recovered_np - self.gt_np, ord=2, axis=0, keepdims=False) / (gt_norm + 1.0e-6)

        if enable_plot:
            record.visual(plot_1d, self.gt_np, recovered_np, f"inverse-{self.pde_idx}.png",
                          title=self.pde_latex, save_dir=record.inverse_dir)

        return nrmse.item()

    def visual(self, model: nn.Cell) -> None:
        r"""
        Visualization of the real equation solution (label), noisy equation solution (noisy),
        observed equation solution with coordinate subsampling (obs),
        model's predicted solution given ground-truth equation function value (raw_pred), and
        model's predicted solution given the recovered equation function value (pred).

        Args:
            model (nn.Cell): The model to predict the solution.

        returns:
            None.
        """
        label = self.u_label  # [n_graph, n_t_grid, n_x_grid, 1]
        noisy = self.u_noisy  # [n_graph, n_t_grid, n_x_grid, 1]
        obs = self.u_obs_plot  # [n_graph, n_t_grid, n_x_grid, 1]

        raw_input_tuple = self.data_tuple[:-1]
        raw_pred = model(*raw_input_tuple)  # [n_graph, num_point, 1]

        input_tuple = self.get_data_tuple(is_train=False)[:-1]
        pred = model(*input_tuple)  # [n_graph, num_point, 1]

        idx_list = list(range(config.inverse.plot_num_per_cls))
        tx_grid_shape = self.tx_grid_shape
        for plot_idx in idx_list:
            label_plot = label[plot_idx, :, :, 0]
            noisy_plot = noisy[plot_idx, :, :, 0]
            obs_plot = obs[plot_idx, :, :, 0]

            raw_pred_plot = raw_pred[plot_idx, :, 0].view(tx_grid_shape).asnumpy().astype(np.float32)
            pred_plot = pred[plot_idx, :, 0].view(tx_grid_shape).asnumpy().astype(np.float32)

            file_name = f"compare-{self.pde_idx}-{plot_idx}.png"
            plot_list = [label_plot, noisy_plot, obs_plot, raw_pred_plot, pred_plot]
            record.visual(plot_2dxn, plot_list, file_name,
                          title=self.pde_latex, save_dir=record.image2d_dir)


def inverse(model: nn.Cell) -> None:
    r"""
    Solve the inverse problem that recovers the function-valued term in a PDE
    from the observed data using gradient descent based on the pre-trained model.
    """
    # loss function
    loss_fn = LossFunction(config.inverse.loss.type.upper(),
                           normalize=True,
                           reduce_mean=True,
                           normalize_eps=config.inverse.loss.normalize_eps)

    # learning rate
    lr_var = get_lr_list(1,
                         config.inverse.epochs,
                         config.inverse.learning_rate,
                         lr_scheduler_type="mstep",
                         lr_milestones=[1.])

    # auto mixed precision
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    nrmse_all = []
    pde_samples = config.inverse.pde_samples
    if isinstance(pde_samples, int):
        pde_samples = range(pde_samples)
    for pde_idx in pde_samples:
        data_tuple, data_info = get_inverse_data(config, pde_idx)
        if 'pde_latex' not in data_info:
            raise KeyError(f"'pde_latex' not found in data_info for PDE {pde_idx}.")
        if 'coef_list' not in data_info:
            raise KeyError(f"'coef_list' not found in data_info for PDE {pde_idx}.")
        record.print(f"PDE {pde_idx}: {data_info['pde_latex']}\n  coefs: {data_info['coef_list']}")
        invp = InverseProblem(pde_idx, data_tuple, config.inverse, data_info)

        # optimizer
        params = [{'params': [invp.recovered], 'lr': lr_var, 'weight_decay': config.inverse.weight_decay}]
        optimizer = nn.Adam(params)

        # distributed training (data parallel)
        if use_ascend and args.distributed:
            grad_reducer = nn.DistributedGradReducer(optimizer.parameters)
        else:
            def grad_reducer(x_in):
                return x_in

        # define forward function
        get_data_tuple = invp.get_data_tuple
        regularization_loss = invp.regularization_loss

        def forward_fn():
            data_tuple = get_data_tuple(is_train=True)  # pylint: disable=W0640
            input_tuple = data_tuple[:-1]  # tuple
            label = data_tuple[-1]  # tensor
            pred = model(*input_tuple)
            loss = loss_fn(pred, label) + regularization_loss()  # pylint: disable=W0640

            # auto mixed precision
            if use_ascend:
                loss = loss_scaler.scale(loss)

            return loss, pred

        # define gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # function for one step of training
        @ms.jit
        def train_step():
            (loss, pred), grads = grad_fn()  # pylint: disable=W0640

            grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
            # distributed training (data parallel)
            grads = grad_reducer(grads)  # pylint: disable=W0640

            # auto mixed precision
            if use_ascend:
                loss = loss_scaler.unscale(loss)
                grads = loss_scaler.unscale(grads)

            loss = ops.depend(loss, optimizer(grads))  # pylint: disable=W0640

            return loss, pred

        # training loop
        print_interval = math.ceil(config.inverse.epochs / 100)
        for epoch in range(1, 1 + config.inverse.epochs):
            loss, _ = train_step()

            if (epoch - 1) % print_interval == 0 or epoch == config.inverse.epochs:
                # record
                loss = loss.asnumpy().item()
                if epoch == config.inverse.epochs:
                    nrmse = invp.compare(enable_plot=True)
                    nrmse_all.append(nrmse)
                else:
                    nrmse = invp.compare(enable_plot=False)
                record.print(f"PDE {pde_idx}, epoch {epoch}: loss {loss:>10f} nrmse {nrmse:>7f}")
                record.add_scalar(f"train_pde-{pde_idx}/loss", loss, epoch)
                record.add_scalar(f"train_pde-{pde_idx}/nrmse", nrmse, epoch)

        invp.visual(model)

    nrmse_mean = np.array(nrmse_all).mean()
    record.print(f"nrmse_mean: {nrmse_mean:>7f}")

    record.print("inversion done!")


if __name__ == "__main__":
    # seed
    set_seed(123456)

    # args
    args = parse_args()

    # mindspore context
    context.set_context(
        mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
        device_target=args.device_target, device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    # compute_type
    compute_type = mstype.float16 if use_ascend else mstype.float32

    # load config file
    config, config_str = load_config(args.config_file_path)

    # record
    record = init_record(use_ascend, 0, args, config, config_str, inverse_problem=True)

    # model
    model_ = get_model(config, record, compute_type)

    # inverse
    try:
        inverse(model_)
    except Exception as err:
        record.close()
        raise err

    record.close()

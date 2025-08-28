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
# ==============================================================================
"""
Evaluation metrics for DeepONet-Grid-UQ
"""

from typing import Any, Dict, List, Optional, Tuple

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
import numpy as np


class MetricsCalculator:
    """Metrics calculator for DeepONet evaluation"""

    def __init__(self):
        self.norm = ops.LpNorm(axis=-1, keep_dims=False)
        self.norm_l1 = ops.LpNorm(axis=-1, keep_dims=False, p=1)
        self.reduce_mean = ops.ReduceMean()
        self.abs = ops.Abs()
        self.exp = ops.Exp()

    def l2_relative_error(self, y_true: ms.Tensor, y_pred: ms.Tensor) -> float:
        diff = (y_true - y_pred).reshape(-1)
        true = y_true.reshape(-1)
        numerator = ops.norm(diff, ord=2)
        denominator = ops.norm(true, ord=2)
        value = numerator / denominator
        if hasattr(value, "asnumpy"):
            value = value.asnumpy()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def l1_relative_error(self, y_true: ms.Tensor, y_pred: ms.Tensor) -> float:
        diff = (y_true - y_pred).reshape(-1)
        true = y_true.reshape(-1)
        numerator = ops.norm(diff, ord=1)
        denominator = ops.norm(true, ord=1)
        value = numerator / denominator
        if hasattr(value, "asnumpy"):
            value = value.asnumpy()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    def fraction_in_CI(
        self,
        s: ms.Tensor,
        s_mean: ms.Tensor,
        s_std: ms.Tensor,
        xi: float = 2.0,
        verbose: bool = False,
    ) -> float:
        """Compute fraction of true trajectory in predicted confidence interval"""
        # Reshape to 1D if needed
        s = s.reshape(-1)
        s_mean = s_mean.reshape(-1)
        s_std = s_std.reshape(-1)

        # Check if points are within confidence interval
        within_CI = self.abs(s - s_mean) <= xi * s_std
        ratio = float(self.reduce_mean(within_CI.astype(ms.float32)))

        if verbose:
            print(f"% of the true traj. within the error bars is {100 * ratio:.3f}")

        return ratio

    def trajectory_rel_error(
        self, s_true: ms.Tensor, s_pred: ms.Tensor, verbose: bool = False
    ) -> Tuple[float, float]:
        """Compute trajectory relative errors"""
        s_true_flat = s_true.reshape(-1)
        s_pred_flat = s_pred.reshape(-1)

        l1_error = self.l1_relative_error(s_true_flat, s_pred_flat)
        l2_error = self.l2_relative_error(s_true_flat, s_pred_flat)

        if verbose:
            print(f"The L1 relative error is {l1_error:.5f}")
            print(f"The L2 relative error is {l2_error:.5f}")

        return l1_error, l2_error


def compute_metrics(
    s_true: List[ms.Tensor],
    s_pred: List[ms.Tensor],
    metrics: List[str],
    verbose: bool = False,
) -> List[List[float]]:
    """Compute metrics for multiple trajectories"""
    calculator = MetricsCalculator()
    out = []

    for metric_name in metrics:
        temp = []
        for k in range(len(s_true)):
            if metric_name.lower() == "l1":
                error = calculator.l1_relative_error(s_true[k], s_pred[k])
            elif metric_name.lower() == "l2":
                error = calculator.l2_relative_error(s_true[k], s_pred[k])
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
            temp.append(error)

        temp_np = np.array(temp)
        out.append(
            [
                np.round(100 * np.max(temp_np), decimals=5),
                np.round(100 * np.min(temp_np), decimals=5),
                np.round(100 * np.mean(temp_np), decimals=5),
            ]
        )

    if verbose:
        try:
            print(
                f"l1-relative errors: max={out[0][0]:.3f}, min={out[0][1]:.3f}, mean={out[0][2]:.3f}"
            )
            print(
                f"l2-relative errors: max={out[1][0]:.3f}, min={out[1][1]:.3f}, mean={out[1][2]:.3f}"
            )
        except BaseException:
            print("not the correct metrics")

    return out


def update_metrics_history(
    history: Dict[str, List[float]], state: List[float]
) -> Dict[str, List[float]]:
    """Update metrics history"""
    if "max" not in history:
        history["max"] = []
    if "min" not in history:
        history["min"] = []
    if "mean" not in history:
        history["mean"] = []

    history["max"].append(state[0])
    history["min"].append(state[1])
    history["mean"].append(state[2])

    return history


def compute_r2_score(y_true: ms.Tensor, y_pred: ms.Tensor) -> float:
    """Compute R² score"""
    ss_res = ops.ReduceSum()((y_true - y_pred) ** 2)
    ss_tot = ops.ReduceSum()((y_true - ops.ReduceMean()(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot
    return float(r2)


def compute_mae(y_true: ms.Tensor, y_pred: ms.Tensor) -> float:
    """Compute Mean Absolute Error"""
    mae = ops.ReduceMean()(ops.Abs()(y_true - y_pred))
    return float(mae)


def compute_mse(y_true: ms.Tensor, y_pred: ms.Tensor) -> float:
    """Compute Mean Squared Error"""
    mse = ops.ReduceMean()((y_true - y_pred) ** 2)
    return float(mse)

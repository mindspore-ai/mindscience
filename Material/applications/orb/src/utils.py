# ============================================================================
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
"""Experiment utilities."""

import math
import random
import re
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Tuple, TypeVar, Any

import yaml
import numpy as np
import mindspore as ms
from mindspore import Tensor, mint

from src import base

T = TypeVar("T")


def load_cfg(filename):
    """load_cfg

    Load configurations from yaml file and return a namespace object
    """
    from argparse import Namespace
    with open(filename, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Namespace(**cfg)


def ensure_detached(x: base.Metric) -> base.Metric:
    """Ensure that the tensor is detached and on the CPU."""
    return x


def to_item(x: base.Metric) -> base.Metric:
    """Convert a tensor to a python scalar."""
    if isinstance(x, Tensor):
        return x.item()
    return x


def prefix_keys(
        dict_to_prefix: Dict[str, T], prefix: str, sep: str = "/"
) -> Dict[str, T]:
    """Add a prefix to dictionary keys with a separator."""
    return {f"{prefix}{sep}{k}": v for k, v in dict_to_prefix.items()}


def seed_everything(seed: int, rank: int = 0) -> None:
    """Set the seed for all pseudo random number generators."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    ms.manual_seed(seed + rank)


class ScalarMetricTracker:
    """Keep track of average scalar metric values."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the AverageMetrics."""
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, metrics: Mapping[str, base.Metric]) -> None:
        """Update the metric counts with new values."""
        for k, v in metrics.items():
            if isinstance(v, Tensor) and v.nelement() > 1:
                continue  # only track scalar metrics
            if isinstance(v, Tensor) and v.isnan().any():
                continue
            self.sums[k] += ensure_detached(v)
            self.counts[k] += 1

    def get_metrics(self):
        """Get the metric values, possibly reducing across gpu processes."""
        return {k: to_item(v) / self.counts[k] for k, v in self.sums.items()}


def gradient_clipping(
        model: ms.nn.Cell, clip_value: float
) -> List[Any]:
    """Add gradient clipping hooks to a model.

    This is the correct way to implement gradient clipping, because
    gradients are clipped as gradients are computed, rather than after
    all gradients are computed - this means expoding gradients are less likely,
    because they are "caught" earlier.

    Args:
        model: The model to add hooks to.
        clip_value: The upper and lower threshold to clip the gradients to.

    Returns:
        A list of handles to remove the hooks from the parameters.
    """
    handles = []

    def _clip(grad):
        if grad is None:
            return grad
        return grad.clamp(min=-clip_value, max=clip_value)

    for parameter in model.trainable_params():
        if parameter.requires_grad:
            h = parameter.register_hook(_clip)
            handles.append(h)

    return handles


def get_optim(
        lr: float, total_steps: int, model: ms.nn.Cell
) -> Tuple[ms.experimental.optim.Optimizer, Optional[ms.experimental.optim.lr_scheduler.LRScheduler]]:
    """Configure optimizers, LR schedulers and EMA."""

    # Initialize parameter groups
    params = []

    # Split parameters based on the regex
    for param in model.trainable_params():
        name = param.name
        if re.search(r"(.*bias|.*layer_norm.*|.*batch_norm.*)", name):
            params.append({"params": param, "weight_decay": 0.0})
        else:
            params.append({"params": param})

    # Create the optimizer with the parameter groups
    optimizer = ms.experimental.optim.Adam(params, lr=lr)

    # Create the learning rate scheduler
    scheduler = ms.experimental.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=1.0e-9, max_lr=lr, step_size_up=int(total_steps*0.04), step_size_down=total_steps
    )

    return optimizer, scheduler


def rand_angles(*shape, dtype=None):
    r"""random rotation angles

    Parameters
    ----------
    *shape : int

    Returns
    -------
    alpha : `Tensor`
        tensor of shape :math:`(\mathrm{shape})`

    beta : `Tensor`
        tensor of shape :math:`(\mathrm{shape})`

    gamma : `Tensor`
        tensor of shape :math:`(\mathrm{shape})`
    """
    alpha, gamma = 2 * math.pi * mint.rand(2, *shape, dtype=dtype)
    beta = mint.rand(shape, dtype=dtype).mul(2).sub(1).acos()
    return alpha, beta, gamma


def matrix_x(angle: Tensor) -> Tensor:
    r"""matrix of rotation around X axis

    Parameters
    ----------
    angle : `Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = mint.ones_like(angle)
    z = mint.zeros_like(angle)
    return mint.stack(
        [
            mint.stack([o, z, z], dim=-1),
            mint.stack([z, c, -s], dim=-1),
            mint.stack([z, s, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_y(angle: Tensor) -> Tensor:
    r"""matrix of rotation around Y axis

    Parameters
    ----------
    angle : `Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = mint.ones_like(angle)
    z = mint.zeros_like(angle)
    return mint.stack(
        [
            mint.stack([c, z, s], dim=-1),
            mint.stack([z, o, z], dim=-1),
            mint.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )


def matrix_z(angle: Tensor) -> Tensor:
    r"""matrix of rotation around Z axis

    Parameters
    ----------
    angle : `Tensor`
        tensor of any shape :math:`(...)`

    Returns
    -------
    `Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = mint.ones_like(angle)
    z = mint.zeros_like(angle)
    return mint.stack(
        [
            mint.stack([c, -s, z], dim=-1),
            mint.stack([s, c, z], dim=-1),
            mint.stack([z, z, o], dim=-1),
        ],
        dim=-2,
    )


def angles_to_matrix(alpha, beta, gamma):
    r"""conversion from angles to matrix

    Parameters
    ----------
    alpha : `Tensor`
        tensor of shape :math:`(...)`

    beta : `Tensor`
        tensor of shape :math:`(...)`

    gamma : `Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = ms.numpy.broadcast_arrays(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)


def rand_matrix(*shape, dtype=None):
    r"""random rotation matrix

    Parameters
    ----------
    *shape : int

    Returns
    -------
    `Tensor`
        tensor of shape :math:`(\mathrm{shape}, 3, 3)`
    """
    rotation_matrix = angles_to_matrix(*rand_angles(*shape, dtype=dtype))
    return rotation_matrix

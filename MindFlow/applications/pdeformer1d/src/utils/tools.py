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
r"""Some tool functions."""
import random
from typing import Tuple
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor


def set_seed(seed: int) -> None:
    r"""Set random seed"""
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


def calculate_num_params(model: nn.Cell) -> str:
    r"""Calculate the number of parameters."""
    num_params = 0
    for param in model.trainable_params():
        num_params += np.prod(param.shape)

    if num_params < 1000:
        num_str = str(num_params)
    elif num_params < 1000 * 1000:
        num_str = f"{(num_params / 1000):.2f}" + "K"
    elif num_params < 1000 * 1000 * 1000:
        num_str = f"{(num_params / (1000*1000)):.2f}" + "M"
    else:
        num_str = f"{(num_params / (1000*1000*1000)):.2f}" + "G"

    return num_str


class AllGather(nn.Cell):
    r"""Use nn.Cell to encapsulate ops.AllGather()."""

    def __init__(self) -> None:
        super().__init__()
        self.allgather = ops.AllGather()

    def construct(self, x: Tensor) -> Tensor:
        return self.allgather(x)


def postprocess_batch_data(label: Tensor,
                           pred: Tensor,
                           data_info: dict,
                           idx: int,
                           model_type: str) -> Tuple[np.ndarray, np.ndarray, str]:
    r"""
    Postprocess the batch data to get pde latex and the label and predicted values (with only first component).

    Args:
        label (Tensor): The label tensor. Shape: math:`[bsz, n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[bsz, dim\_out, n\_t\_grid, n\_x\_grid]` or `[bsz, n\_t\_grid, n\_x\_grid, dim\_out]`.
        pred (Tensor): The predicted tensor. Shape: math:`[bsz, n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[bsz, dim\_out, n\_t\_grid, n\_x\_grid]` or `[bsz, n\_t\_grid, n\_x\_grid, dim\_out]`.
        data_info (dict): The data information.
        idx (int): The index of the sample in the batch.
        model_type (str): The model type.

    Returns:
        Tuple[np.ndarray, np.ndarray, str]: The label and predicted values (with only first component) and pde latex.
    """
    if len(label.shape) == 3:
        tx_grid_shape = (data_info["n_t_grid"], data_info["n_x_grid"])
        label_ = label[idx, :, 0].view(tx_grid_shape)
        pred_ = pred[idx, :, 0].view(tx_grid_shape)
    elif len(label.shape) == 4:
        if model_type == "u-net":
            label_ = label[idx, 0, :, :]
            pred_ = pred[idx, 0, :, :]
        elif model_type == "fno":
            label_ = label[idx, :, :, 0]
            pred_ = pred[idx, :, :, 0]
        else:
            raise ValueError(f"The model_type {model_type} is not supported!")
    else:
        raise RuntimeError("Tensor shape error!")
    label_ = label_.asnumpy().astype(np.float32)
    pred_ = pred_.asnumpy().astype(np.float32)
    return label_, pred_, data_info["pde_latex"]


def postprocess_data(label: Tensor,
                     pred: Tensor,
                     data_info: dict,
                     model_type: str) -> Tuple[np.ndarray, np.ndarray, str]:
    r"""
    Postprocess the data (only one sample) to get pde latex and the label and predicted
    values (with only first component).

    Args:
        label (Tensor): The label tensor. Shape: math:`[n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[dim\_out, n\_t\_grid, n\_x\_grid]` or `[n\_t\_grid, n\_x\_grid, dim\_out]`.
        pred (Tensor): The predicted tensor. Shape: math:`[n\_t\_grid * n\_x\_grid, dim\_out]`
            or `[dim\_out, n\_t\_grid, n\_x\_grid]` or `[n\_t\_grid, n\_x\_grid, dim\_out]`.

    Returns:
        Tuple[np.ndarray, np.ndarray, str]: The label and predicted values (with only first component) and pde latex.
    """
    if len(label.shape) == 2:
        tx_grid_shape = (data_info["n_t_grid"], data_info["n_x_grid"])
        label_ = label[:, 0].view(tx_grid_shape)
        pred_ = pred[:, 0].view(tx_grid_shape)
    elif len(label.shape) == 3:
        if model_type == "u-net":
            label_ = label[0, :, :]
            pred_ = pred[0, :, :]
        elif model_type == "fno":
            label_ = label[:, :, 0]
            pred_ = pred[:, :, 0]
        else:
            raise ValueError(f"The model_type {model_type} is not supported!")
    else:
        raise RuntimeError("Tensor shape error!")

    label_ = label_.asnumpy().astype(np.float32)
    pred_ = pred_.asnumpy().astype(np.float32)
    return label_, pred_, data_info["pde_latex"]

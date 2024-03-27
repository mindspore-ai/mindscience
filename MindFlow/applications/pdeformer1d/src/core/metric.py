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
r"""This module provides functions to compute and record metrics."""
import numpy as np
from mindspore import Tensor


def calculate_l2_error(label: Tensor, pred: Tensor) -> np.array:
    r"""
    Computes the relative L2 loss.

    Args:
        label (Tensor): The shape of tensor is math:`(bsz, \ldot)`.
        pred (Tensor): The shape of tensor is math:`(bsz, \ldots)`.

    Returns:
        Tensor: The relative L2 loss. The shape of tensor is math:`(bsz)`.
    """
    label = label.view((label.shape[0], -1)).asnumpy()  # [bsz, *]
    pred = pred.view((label.shape[0], -1)).asnumpy()  # [bsz, *]

    error_norm = np.linalg.norm(pred - label, ord=2, axis=1, keepdims=False)  # [bsz]
    label_norm = np.linalg.norm(label, ord=2, axis=1, keepdims=False)  # [bsz]
    relative_l2_error = error_norm / (label_norm + 1.0e-6)  # [bsz]
    relative_l2_error = relative_l2_error.clip(0, 5)  # [bsz]

    return relative_l2_error


class L2ErrorRecord:
    r"""Records the L2 errors on different datasets."""

    def __init__(self) -> None:
        self.dict = dict()

    @staticmethod
    def dict2str(l2_error_dict: dict) -> str:
        r"""
        Converts the L2 error dictionary to a string.

        Args:
            l2_error_dict (dict): The L2 error dictionary.

        Returns:
            str: The string of the L2 error dictionary.
        """
        tmp = ""
        for key, value in l2_error_dict.items():
            tmp += f"{key}: {value:>7f} "
        return tmp

    def append(self, pde: str, param: str, l2_error: np.array) -> dict:
        r"""
        Appends the L2 error of a specific dataset to the record.

        Args:
            pde (str): The name of the PDE.
            param (str): The name of the PDE parameter.
            l2_error (np.array): The L2 error of the dataset. The shape of tensor is math:`(bsz)`.

        Returns:
            dict: The L2 error of the specific dataset.
        """
        centered_min = np.percentile(l2_error, 1)
        centered_max = np.percentile(l2_error, 99)
        centered_mean = l2_error.clip(centered_min, centered_max).mean()

        l2_error_dict = {
            "l2_error_mean": l2_error.mean(),
            "l2_error_min": l2_error.min(),
            "l2_error_max": l2_error.max(),
            "l2_error_centered_mean": centered_mean,
            "l2_error_centered_min": centered_min,
            "l2_error_centered_max": centered_max,
        }

        if pde in self.dict:
            self.dict[pde][param] = l2_error_dict
        else:
            self.dict[pde] = {param: l2_error_dict}

        return l2_error_dict

    def reduce(self, pde: str) -> dict:
        r"""
        Reduces the L2 error of each PDE which contains the different parameters to a single value.

        Args:
            pde (str): The name of the PDE.

        Returns:
            dict: The L2 error of the specific PDE.
        """
        mean_tmp = []
        min_tmp = []
        max_tmp = []
        centered_mean_tmp = []
        centered_min_tmp = []
        centered_max_tmp = []

        if pde == "all":
            for _, value in self.dict.items():
                for _, sub_value in value.items():
                    mean_tmp.append(sub_value["l2_error_mean"])
                    min_tmp.append(sub_value["l2_error_min"])
                    max_tmp.append(sub_value["l2_error_max"])
                    centered_mean_tmp.append(sub_value["l2_error_centered_mean"])
                    centered_min_tmp.append(sub_value["l2_error_centered_min"])
                    centered_max_tmp.append(sub_value["l2_error_centered_max"])
        else:
            for _, value in self.dict[pde].items():
                mean_tmp.append(value["l2_error_mean"])
                min_tmp.append(value["l2_error_min"])
                max_tmp.append(value["l2_error_max"])
                centered_mean_tmp.append(value["l2_error_centered_mean"])
                centered_min_tmp.append(value["l2_error_centered_min"])
                centered_max_tmp.append(value["l2_error_centered_max"])

        l2_error_dict = {
            "l2_error_mean": np.mean(mean_tmp),
            "l2_error_min": np.min(min_tmp),
            "l2_error_max": np.max(max_tmp),
            "l2_error_centered_mean": np.mean(centered_mean_tmp),
            "l2_error_centered_min": np.min(centered_min_tmp),
            "l2_error_centered_max": np.max(centered_max_tmp),
        }

        return l2_error_dict

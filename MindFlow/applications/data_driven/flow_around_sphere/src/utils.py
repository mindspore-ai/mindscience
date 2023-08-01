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
"""
utils
"""
import os
import time

import numpy as np


def check_file_path(path):
    """check file dir."""
    if not os.path.exists(path):
        os.makedirs(path)


def calculate_metric(problem, dataloader):
    """
    Evaluate the model respect to dataloader.

    Args:
        problem(UnsteadyFlow3D): the class of unsteady self-defined data-driven problems.
        dataloader(BatchDataset): The data set used to evaluate the model

    Return:
        Float, model metric value.
    """
    print("================================Start Evaluation================================", flush=True)
    time_beg = time.time()
    metric = 0.

    cal_data_size = dataloader.get_dataset_size()
    for _, (inputs, labels) in enumerate(dataloader.create_tuple_iterator()):
        metric += problem.get_metric(inputs, labels)
    mean_metric = metric / cal_data_size
    print(f"mean metric: {mean_metric:.8f}  eval total time:{time.time() - time_beg:.2f}", flush=True)
    print("=================================End Evaluation=================================", flush=True)


def max_min_normalize(train_original_data, data_numpy):
    """
    Standardize the dataset with Max-Min in train_original_data,.
    Args:
        train_original_data(ndarray): Train flow snapshots to estimate overall performance.
        data_numpy(ndarray): Original flow snapshot array. -> (T, C, D, H, W)
    """
    min_value = np.min(train_original_data, axis=(0, 2, 3, 4))
    range_value = np.max(train_original_data, axis=(0, 2, 3, 4)) - np.min(train_original_data, axis=(0, 2, 3, 4))
    temp_data_numpy = np.empty_like(data_numpy)
    for i in range_value(data_numpy.shape[1]):
        temp_data_numpy[:, i, ...] = (data_numpy[:, i, ...] - min_value[i]) / range_value[i]
    return temp_data_numpy

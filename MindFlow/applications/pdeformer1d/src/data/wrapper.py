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
r"""Load the dataset according to the configuration."""

from typing import Tuple
from omegaconf import DictConfig

from .load_multi_pde import multi_pde_dataset
from .load_single_pde import single_pde_dataset


def load_dataset(config: DictConfig) -> Tuple:
    r"""
    Load the dataset according to the configuration.

    Args:
        config (DictConfig): Training configurations.

    Returns:
        dataloader_train (BatchDataset): Data loader for the training dataset.
        train_loader_dict (Dict[str, Dict[str, Tuple]]): A nested
            dictionary containing the data iterator instances of the training
            dataset for the evaluation, in which the random operations (data
            augmentation, spatio-temporal subsampling, etc) are disabled. Here,
            to be more specific, 'Tuple' actually refers to
            'Concatenate[TupleIterator, Dataset]'.
        test_loader_dict (Dict[str, Dict[str, Tuple]]): Similar to
            `train_loader_dict`, but for the testing dataset.
        data_info (Dict[str, Any]): A dictionary containing basic information
            about the current dataset.
    """
    if config.data.type == "multi_pde":
        if config.model_type != "pdeformer":
            raise ValueError("multi_pde dataset only supports model_type==pdeformer")
        return multi_pde_dataset(config)
    if config.data.type == "single_pde":
        return single_pde_dataset(config)
    raise NotImplementedError

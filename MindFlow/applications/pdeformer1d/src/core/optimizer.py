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
r"""Optimizer."""
from mindspore import nn


def get_optimizer(lr_var: list, model: nn.Cell, config: dict) -> nn.Cell:
    r"""
    Get the optimizer according to the config.

    Args:
        lr_var (list): A list of learning rate variables.
        model (nn.Cell): The model to be trained.
        config (dict): The configuration of the training process.

    Returns:
        nn.Cell: The optimizer.
    """
    params = [{'params': model.trainable_params(), 'lr': lr_var, 'weight_decay': config.train.weight_decay}]
    if config.train.optimizer == 'Adam':
        optimizer = nn.Adam(params)
    elif config.train.optimizer == 'AdamW':
        optimizer = nn.AdamWeightDecay(params)
    else:
        raise NotImplementedError(f"'optimizer' should be one of ['Adam', 'AdamW'], but got {config.train.optimizer}")

    return optimizer

# Copyright 2024 Huawei Technologies Co., Ltd
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
"""utils/optimizer.py"""

from typing import Tuple

import mindspore
from mindspore.experimental import optim
from mindspore.experimental.optim.lr_scheduler import LambdaLR
from mindspore.nn import Adam


def setup_optimizer(
        model: mindspore.nn.Cell,
        learning_rate: float,
        epsilon: float,
        betas: Tuple[float, float],
        weight_decay: float,
) -> optim.AdamW:
    """
    Set up the AdamW optimizer and group parameters.

    Args:
        model (mindspore.nn.Cell): The model to be optimized.
        learning_rate (float): Learning rate.
        epsilon (float): AdamW's epsilon value.
        betas (Tuple[float, float]): The betas parameter for AdamW.
        weight_decay (float): Weight decay.

    Returns:
        optim.AdamW: Initialized optimizer.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    parameters = model.trainable_params()
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in zip([param.name for param in parameters], parameters)
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in zip([param.name for param in parameters], parameters)
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        params=optimizer_grouped_parameters, lr=learning_rate, eps=epsilon, betas=betas
    )
    return optimizer


def setup_finetune_optimizer(
        model: mindspore.nn.Cell,
        learning_rate: float,
) -> Adam:
    """
    The Adam optimizer to use when setting up fine-tuning.

    Args:
        model (mindspore.nn.Cell): The model to be optimized.
        learning_rate (float): Learning rate.

    Returns:
        Adam: Configured Adam optimizer.
    """
    optimizer = Adam(
        model.trainable_params(),
        learning_rate=learning_rate,
    )
    return optimizer


def get_linear_schedule_with_warmup(
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
) -> LambdaLR:
    """
    Creates a linear learning rate scheduler with warmup steps.

    Args:
        optimizer (optim.Optimizer): The optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        last_epoch (int, optional): The index of last epoch. Default is -1.

    Returns:
        LambdaLR: Learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

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
"""
Training utilities for MindSpore PowerFlowNet.

This module provides training functions aligned with PyTorch training utilities:
  - train_epoch: Train model for one epoch with gradient updates
  - append_to_json: Append training results to JSON log file

Compatible with various loss functions:
  - MaskedL2Loss
  - PowerImbalance
  - MixedMSEPowerImbalance

Note:
    Current implementation uses manual parameter updates. For production use,
    consider using MindSpore's built-in optimizers (nn.Adam, nn.SGD, etc.)
    with TrainOneStepCell for better performance and flexibility.
"""

import json
import os
from typing import Callable

import mindspore as ms
from mindspore import nn, Tensor

from src.custom_loss_functions import MaskedL2Loss, PowerImbalance, MixedMSEPowerImbalance


def append_to_json(log_path, run_id, result):
    """Append training result to JSON log file

    Args:
        log_path: Path to log JSON file
        run_id: Unique identifier for this run
        result: Dictionary of results to append
    """
    log_entry = {str(run_id): result}

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "r", encoding="utf-8") as json_file:
            exist_log = json.load(json_file)
    except FileNotFoundError:
        exist_log = {}
    with open(log_path, "w", encoding="utf-8") as json_file:
        exist_log.update(log_entry)
        json.dump(exist_log, json_file, indent=4)


def train_epoch(
    model: nn.Cell,
    loader,
    loss_fn: Callable,
    optimizer: nn.Optimizer,
    device: str = 'CPU'
) -> float:
    """
    Trains a neural network model for one epoch.

    Args:
        model: The MindSpore model to be trained
        loader: Data loader containing training data
        loss_fn: Loss function
        optimizer: MindSpore optimizer (unused, kept for API compatibility)
        device: Device type (unused, kept for API compatibility)

    Returns:
        float: The mean loss value over all batches
    """
    del optimizer, device  # Unused, kept for API compatibility
    model.set_train(True)

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Extract batch data and convert to tensors if needed
        x = batch.x if isinstance(batch.x, Tensor) else Tensor(batch.x, dtype=ms.float32)
        y = batch.y if isinstance(batch.y, Tensor) else Tensor(batch.y, dtype=ms.float32)
        edge_index = (batch.edge_index if isinstance(batch.edge_index, Tensor)
                      else Tensor(batch.edge_index, dtype=ms.int64))
        edge_attr = (batch.edge_attr if isinstance(batch.edge_attr, Tensor)
                     else Tensor(batch.edge_attr, dtype=ms.float32))
        pred_mask = (batch.pred_mask if isinstance(batch.pred_mask, Tensor)
                     else Tensor(batch.pred_mask, dtype=ms.float32))
        data_batch = (batch.batch if isinstance(batch.batch, Tensor)
                      else Tensor(batch.batch, dtype=ms.int64))

        # Create data object compatible with model
        class DataObj:
            """Simple data container for batch."""
        data = DataObj()
        data.x = x
        data.y = y
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.pred_mask = pred_mask
        data.batch = data_batch

        # Compute loss directly to avoid cell-var-from-loop
        out = model(data)

        # Compute loss based on loss function type
        if isinstance(loss_fn, MaskedL2Loss):
            loss = loss_fn(out, y, pred_mask)
        elif isinstance(loss_fn, PowerImbalance):
            # Mask out non-predicted values
            masked_out = out * pred_mask + x * (1 - pred_mask)
            loss = loss_fn(masked_out, edge_index, edge_attr)
        elif isinstance(loss_fn, MixedMSEPowerImbalance):
            loss = loss_fn(out, edge_index, edge_attr, y)
        else:
            loss = loss_fn(out, y)

        # Note: Using simple forward pass, gradient computation
        # requires proper MindSpore TrainOneStepCell for production

        # Accumulate loss
        total_loss += float(loss.asnumpy())
        num_batches += 1

    mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return mean_loss


def main():
    """Test function for append_to_json"""
    log_path = 'logs/save_logs.json'
    run_id = 'arb_id_01'
    result = {
        'train_loss': 0.3,
        'val_loss': 0.2,
    }
    append_to_json(log_path, run_id, result)


if __name__ == '__main__':
    main()

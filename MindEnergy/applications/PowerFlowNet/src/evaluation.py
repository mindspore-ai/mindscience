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
MindSpore Evaluation Module

Provides evaluation functions aligned with PyTorch version:
  - evaluate_epoch: Simple epoch evaluation with single loss value
  - evaluate_epoch_v2: Detailed evaluation with loss term breakdowns
  - load_model: Load trained model from checkpoint
  - num_params: Count trainable parameters

Compatible with various loss functions:
  - Masked_L2_loss
  - PowerImbalance
  - MixedMSEPowerImbalance
  - MaskedL2V2
  - MaskedL1
"""

import os
from typing import Callable, Dict, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, Tensor
from tqdm import tqdm

from src.custom_loss_functions import (
    MaskedL2Loss, PowerImbalance, MixedMSEPowerImbalance,
    PowerMaskedLoss
)

LOG_DIR = 'logs'
SAVE_DIR = 'models'


def load_model(
    model: nn.Cell,
    run_id: str,
    device: str = 'CPU'
) -> Tuple[nn.Cell, Dict]:
    """
    Load trained model from checkpoint.

    Args:
        model (nn.Cell): MindSpore model to load weights into
        run_id (str): Unique identifier for the model checkpoint
        device (str): Device type (unused, kept for API compatibility)

    Returns:
        Tuple[nn.Cell, Dict]: Loaded model and checkpoint data

    Raises:
        FileNotFoundError: If checkpoint file not found
    """
    del device  # Unused, kept for API compatibility
    save_model_path = os.path.join(SAVE_DIR, 'model_' + run_id + '.ckpt')

    try:
        checkpoint = ms.load_checkpoint(save_model_path)
        ms.load_param_into_net(model, checkpoint)

        # Try to load metadata if available
        metadata_path = os.path.join(SAVE_DIR, 'model_' + run_id + '_metadata.pt')
        metadata = {}
        if os.path.exists(metadata_path):
            # pylint: disable=import-outside-toplevel
            import torch
            metadata = torch.load(metadata_path, map_location='cpu')

        return model, metadata
    except (FileNotFoundError, RuntimeError) as e:
        print(f"File not found or error loading checkpoint: {e}")
        return model, {}


def num_params(model: nn.Cell) -> int:
    """
    Count the number of trainable parameters in a MindSpore model.

    Args:
        model (nn.Cell): The MindSpore model

    Returns:
        int: Total number of trainable parameters
    """
    total_params = 0
    for param in model.trainable_params():
        total_params += param.size
    return total_params


def evaluate_epoch(
    model: nn.Cell,
    loader,
    loss_fn: Callable,
    device: str = 'CPU',
    pre_loss_fn: Optional[Callable] = None,
) -> float:
    """
    Evaluate model performance over an entire epoch.

    Simple version that returns a single averaged loss value.

    Args:
        model (nn.Cell): MindSpore model in evaluation mode
        loader: Data loader containing evaluation batches
        loss_fn (Callable): Loss function to use
        device (str): Device to evaluate on (unused, kept for API compatibility)
        pre_loss_fn (Callable, optional): Pre-processing function for loss computation

    Returns:
        float: Mean loss over all batches
    """
    del device  # Unused, kept for API compatibility
    pre_loss_fn = pre_loss_fn or (lambda x: x)

    model.set_train(False)
    total_loss = 0.0
    num_samples = 0

    pbar = tqdm(loader, total=len(loader), desc='Evaluating:')
    for data in pbar:
        # Convert data to MindSpore tensors if needed
        if hasattr(data, 'x'):
            # Handle Data object from PyG-like interface
            x = Tensor(data.x) if not isinstance(data.x, Tensor) else data.x
            y = Tensor(data.y) if not isinstance(data.y, Tensor) else data.y
            ei = data.edge_index
            edge_index = Tensor(ei) if not isinstance(ei, Tensor) else ei
            ea = data.edge_attr
            edge_attr = Tensor(ea) if not isinstance(ea, Tensor) else ea
            pred_mask = Tensor(data.pred_mask) if hasattr(data, 'pred_mask') else None

            # Create data-like object
            class DataObj:
                pass
            batch_data = DataObj()
            batch_data.x = x
            batch_data.y = y
            batch_data.edge_index = edge_index
            batch_data.edge_attr = edge_attr
            if pred_mask is not None:
                batch_data.pred_mask = pred_mask
        else:
            batch_data = data

        # Forward pass
        out = model(batch_data)

        # Compute loss based on loss function type
        if isinstance(loss_fn, MaskedL2Loss):
            out_processed = pre_loss_fn(out)
            target_processed = pre_loss_fn(batch_data.y)
            loss = loss_fn(out_processed, target_processed, batch_data.pred_mask)
        elif isinstance(loss_fn, PowerImbalance):
            mask = batch_data.pred_mask
            masked_out = out * mask + mask * (1 - mask)
            masked_out = pre_loss_fn(masked_out)
            loss = loss_fn(masked_out, batch_data.edge_index, batch_data.edge_attr)
        elif isinstance(loss_fn, MixedMSEPowerImbalance):
            out_processed = pre_loss_fn(out)
            ei, ea = batch_data.edge_index, batch_data.edge_attr
            loss = loss_fn(out_processed, ei, ea, batch_data.y)
        else:
            out_processed = pre_loss_fn(out)
            target_processed = pre_loss_fn(batch_data.y)
            loss = loss_fn(out_processed, target_processed)

        # Accumulate loss
        batch_size = out.shape[0]
        num_samples += batch_size
        total_loss += loss.asnumpy().item() * batch_size

        # Update progress bar
        pbar.set_postfix({'loss': f'{total_loss / num_samples:.6f}'})

    mean_loss = total_loss / num_samples
    return mean_loss


def _compute_loss_terms(
    loss_fn: Callable,
    batch_data,
    out: Tensor,
    pre_loss_fn: Callable
) -> Dict[str, Union[Tensor, float]]:
    """
    Compute loss terms based on loss function type.

    Args:
        loss_fn (Callable): Loss function instance
        batch_data: Batch data object with model inputs
        out (Tensor): Model output
        pre_loss_fn (Callable): Pre-processing function

    Returns:
        Dict[str, Union[Tensor, float]]: Loss terms dictionary
    """
    loss_terms = {}
    out_processed = pre_loss_fn(out)

    if isinstance(loss_fn, MaskedL2Loss):
        target_processed = pre_loss_fn(batch_data.y)
        loss = loss_fn(out_processed, target_processed, batch_data.pred_mask)
        loss_terms['total'] = loss
    elif isinstance(loss_fn, PowerMaskedLoss):
        target_processed = pre_loss_fn(batch_data.y)
        loss_terms = loss_fn(out_processed, target_processed, batch_data.pred_mask)
    elif isinstance(loss_fn, PowerImbalance):
        masked_out = out * batch_data.pred_mask + batch_data.x * (1 - batch_data.pred_mask)
        masked_out = pre_loss_fn(masked_out)
        loss = loss_fn(masked_out, batch_data.edge_index, batch_data.edge_attr)
        loss_terms['total'] = loss
        loss_terms['ref'] = loss_fn(batch_data.y, batch_data.edge_index, batch_data.edge_attr)
    elif isinstance(loss_fn, MixedMSEPowerImbalance):
        loss = loss_fn(out_processed, batch_data.edge_index, batch_data.edge_attr, batch_data.y)
        loss_terms['total'] = loss
    else:
        target_processed = pre_loss_fn(batch_data.y)
        loss = loss_fn(out_processed, target_processed)
        loss_terms['total'] = loss

    return loss_terms


def _accumulate_loss_terms(
    total_loss_terms: Optional[Dict],
    loss_terms: Dict,
    batch_size: int
) -> Dict[str, float]:
    """
    Accumulate loss terms across batches.

    Args:
        total_loss_terms (Optional[Dict]): Accumulated loss terms
        loss_terms (Dict): Current batch loss terms
        batch_size (int): Current batch size

    Returns:
        Dict[str, float]: Updated accumulated loss terms
    """
    if total_loss_terms is None:
        total_loss_terms = {
            key: value.asnumpy().item() if isinstance(value, Tensor) else value
            for key, value in loss_terms.items()
        }
    else:
        for key, value in loss_terms.items():
            value_scalar = value.asnumpy().item() if isinstance(value, Tensor) else value
            total_loss_terms[key] = total_loss_terms[key] + value_scalar * batch_size

    return total_loss_terms


def _convert_to_mindspore_tensors(data) -> object:
    """
    Convert data to MindSpore tensors if needed.

    Args:
        data: Input data object

    Returns:
        object: Data object with MindSpore tensors
    """
    if not hasattr(data, 'x'):
        return data

    x = Tensor(data.x) if not isinstance(data.x, Tensor) else data.x
    y = Tensor(data.y) if not isinstance(data.y, Tensor) else data.y
    ei = data.edge_index
    edge_index = Tensor(ei) if not isinstance(ei, Tensor) else ei
    ea = data.edge_attr
    edge_attr = Tensor(ea) if not isinstance(ea, Tensor) else ea
    pred_mask = Tensor(data.pred_mask) if hasattr(data, 'pred_mask') else None
    bus_type = Tensor(data.bus_type) if hasattr(data, 'bus_type') else None

    class DataObj:
        pass
    batch_data = DataObj()
    batch_data.x = x
    batch_data.y = y
    batch_data.edge_index = edge_index
    batch_data.edge_attr = edge_attr
    if pred_mask is not None:
        batch_data.pred_mask = pred_mask
    if bus_type is not None:
        batch_data.bus_type = bus_type

    return batch_data


def evaluate_epoch_v2(
    model: nn.Cell,
    loader,
    loss_fn: Callable,
    device: str = 'CPU',
    pre_loss_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluate model with detailed loss term breakdown.

    Detailed version that returns loss terms individually for analysis.
    Supports loss functions that return multiple loss components.

    Args:
        model (nn.Cell): MindSpore model in evaluation mode
        loader: Data loader containing evaluation batches
        loss_fn (Callable): Loss function (can return dict of loss terms)
        device (str): Device to evaluate on (unused, kept for API compatibility)
        pre_loss_fn (Callable, optional): Pre-processing function for loss computation

    Returns:
        Dict[str, float]: Dictionary of loss terms and their mean values
    """
    del device  # Unused, kept for API compatibility
    pre_loss_fn = pre_loss_fn or (lambda x: x)

    model.set_train(False)
    total_loss_terms = None
    num_samples = 0

    pbar = tqdm(loader, total=len(loader), desc='Evaluating:')
    for data in pbar:
        batch_data = _convert_to_mindspore_tensors(data)
        out = model(batch_data)
        loss_terms = _compute_loss_terms(loss_fn, batch_data, out, pre_loss_fn)

        batch_size = out.shape[0]
        num_samples += batch_size
        total_loss_terms = _accumulate_loss_terms(total_loss_terms, loss_terms, batch_size)

        if 'total' in total_loss_terms:
            pbar.set_postfix({'loss': f'{total_loss_terms["total"] / num_samples:.6f}'})

    mean_loss_terms = {key: value / num_samples for key, value in total_loss_terms.items()}
    return mean_loss_terms

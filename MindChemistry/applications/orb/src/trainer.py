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
"""Trainer."""

from typing import Dict, Optional, Tuple

import mindspore as ms
from mindspore import ops, Tensor, mint

from src import base, segment_ops

class OrbLoss:
    """Loss function for ORB models.

    This class is used to compute the loss for the ORB model.
    It can be used to compute the loss for both node and graph predictions.
    """

    def __init__(self, model):
        """Initializes the OrbLoss.

        Args:
            target: either the name of a PropertyDefinition or a PropertyDefinition itself.
        """
        self.model = model

    def loss_node(self, batch, out_batch=None):
        """Apply mlp to compute loss and metrics."""
        batch_n_node = batch.n_node
        assert batch.node_targets is not None
        target = batch.node_targets['forces'].squeeze(-1)
        pred = out_batch["node_pred"].squeeze(-1)
        # make sure we remove fixed atoms before normalization
        pred, target, batch_n_node = _remove_fixed_atoms(
            pred, target, batch_n_node, batch.fix_atoms, self.model.training
        )
        mae = mint.abs(pred - self.model.node_head.normalizer(target))
        raw_pred = self.model.node_head.normalizer.inverse(pred)
        raw_mae = mint.abs(raw_pred - target)

        mae = mae.mean(dim=-1)
        mae = segment_ops.aggregate_nodes(
            mae, batch_n_node, reduction="mean"
        ).mean()
        raw_mae = raw_mae.mean(dim=-1)
        raw_mae = segment_ops.aggregate_nodes(
            raw_mae, batch_n_node, reduction="mean"
        ).mean()

        metrics = {
            "node_mae": mae.item(),
            "node_mae_raw": raw_mae.item(),
            "node_cosine_sim": ops.cosine_similarity(raw_pred, target, dim=-1).mean().item(),
            "fwt_0.03": forces_within_threshold(raw_pred, target, batch_n_node),
        }
        return mae, base.ModelOutput(loss=mae, log=metrics)

    def loss_graph(self, batch, out_batch=None):
        """Apply mlp to compute loss and metrics.

        Depending on whether the target is real/binary/categorical, we
        use an MSE/cross-entropy loss. In the case of cross-entropy, the
        preds are logits (not normalised) to take advantage of numerically
        stable log-softmax.
        """
        assert batch.system_targets is not None
        target = batch.system_targets['stress'].squeeze(-1)
        if self.model.stress_head.compute_stress:
            pred = out_batch["stress_pred"].squeeze(-1)
        else:
            pred = out_batch["graph_pred"].squeeze(-1)

        normalized_target = self.model.stress_head.normalizer(target)
        errors = normalized_target - pred
        mae = mint.abs(errors).mean()

        raw_pred = self.model.stress_head.normalizer.inverse(pred)
        raw_mae = mint.abs(raw_pred - target).mean()
        metrics = {"stress_mae": mae.item(), "stress_mae_raw": raw_mae.item()}
        return mae, base.ModelOutput(loss=mae, log=metrics)


    def loss_energy(self, batch, out_batch=None):
        """Apply mlp to compute loss and metrics."""
        assert batch.system_targets is not None
        target = batch.system_targets['energy'].squeeze(-1)
        pred = out_batch["graph_pred"].squeeze(-1)

        reference = self.model.graph_head.reference(batch.atomic_numbers, batch.n_node).squeeze(-1)
        reference_target = target - reference
        if self.model.graph_head.atom_avg:
            reference_target = reference_target / batch.n_node

        normalized_reference = self.model.graph_head.normalizer(reference_target)
        model_loss = normalized_reference - pred

        raw_pred = self.model.graph_head.normalizer.inverse(pred)
        if self.model.graph_head.atom_avg:
            raw_pred = raw_pred * batch.n_node
        raw_mae = mint.abs((raw_pred + reference) - target).mean()

        reference_mae = mint.abs(reference_target).mean()
        model_mae = mint.abs(model_loss).mean()
        metrics = {
            "energy_reference_mae": reference_mae.item(),
            "energy_mae": model_mae.item(),
            "energy_mae_raw": raw_mae.item(),
        }
        return model_mae, base.ModelOutput(loss=model_mae, log=metrics)

    def loss(self, batch, label=None):
        """Loss function of Orb GraphRegressor."""
        assert label is None, "Orb GraphRegressor does not support labels."

        out = self.model(
            batch.edge_features,
            batch.node_features,
            batch.senders,
            batch.receivers,
            batch.n_node,
        )
        loss = Tensor(0.0, ms.float32)
        metrics: Dict = {}

        loss1, graph_out = self.loss_energy(batch, out)
        metrics.update(graph_out.log)
        loss = loss.type_as(loss1) + loss1

        loss2, stress_out = self.loss_graph(batch, out)
        metrics.update(stress_out.log)
        loss = loss.type_as(loss2) + loss2

        loss3, node_out = self.loss_node(batch, out)
        metrics.update(node_out.log)
        loss = loss.type_as(loss3) + loss3

        metrics["loss"] = loss.item()
        return loss, metrics


def binary_accuracy(
        pred: Tensor, target: Tensor, threshold: float = 0.5
) -> float:
    """Calculate binary accuracy between 2 tensors.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        threshold: Binary classification threshold. Default 0.5.

    Returns:
        mean accuracy.
    """
    return ((pred > threshold) == target).to(ms.float32).mean().item()


def categorical_accuracy(pred: Tensor, target: Tensor) -> float:
    """Calculate accuracy for K class classification.

    Args:
        pred: the tensor of logits for K classes of shape (..., K)
        target: tensor of integer target values of shape (...)

    Returns:
        mean accuracy.
    """
    pred_labels = mint.argmax(pred, dim=-1)
    return (pred_labels == target).to(ms.float32).mean().item()


def error_within_threshold(
        pred: Tensor, target: Tensor, threshold: float = 0.02
) -> float:
    """Calculate MAE between 2 tensors within a threshold.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        threshold: margin threshold. Default 0.02 (derived from OCP metrics).

    Returns:
        Mean predictions within threshold.
    """
    error = mint.abs(pred - target)
    within_threshold = error < threshold
    return within_threshold.to(ms.float32).mean().item()


def forces_within_threshold(
        pred: Tensor,
        target: Tensor,
        batch_num_nodes: Tensor,
        threshold: float = 0.03,
) -> float:
    """Calculate MAE between batched graph tensors within a threshold.

    The predictions for a graph are counted as being within the threshold
    only if all nodes in the graph have predictions within the threshold.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        batch_num_nodes: A tensor containing the number of nodes per
            graph.
        threshold: margin threshold. Default 0.03 (derived from OCP metrics).

    Returns:
        Mean predictions within threshold.
    """
    error = mint.abs(pred - target)
    largest_dim_fwt = error.max(-1)[0] < threshold

    count_within_threshold = segment_ops.aggregate_nodes(
        largest_dim_fwt.float(), batch_num_nodes, reduction="sum"
    )
    # count equals batch_num_nodes if all nodes within threshold
    return (count_within_threshold == batch_num_nodes).to(ms.float32).mean().item()


def energy_and_forces_within_threshold(
        pred_energy: Tensor,
        pred_forces: Tensor,
        target_energy: Tensor,
        target_forces: Tensor,
        batch_num_nodes: Tensor,
        fixed_atoms: Optional[Tensor] = None,
        threshold: Tuple[float, float] = (0.02, 0.03),
) -> float:
    """Calculate MAE between batched graph energies and forces within a threshold.

    The predictions for a graph are counted as being within the threshold
    only if all nodes in the graph have predictions within the threshold AND
    the energies are also within a threshold. A combo of the two above functions.

    Args:
        pred_*: the prediction tensors.
        target_*: the tensor of target values.
        batch_num_nodes: A tensor containing the number of nodes per
            graph.
        fixed_atoms: A tensor of bools indicating which atoms are fixed.
        threshold: margin threshold. Default (0.02, 0.03) (derived from OCP metrics).
    Returns:
        Mean predictions within threshold.
    """
    energy_err = mint.abs(pred_energy - target_energy)
    ewt = energy_err < threshold[0]

    forces_err = mint.abs(pred_forces - target_forces)
    largest_dim_fwt = forces_err.max(-1).values < threshold[1]

    working_largest_dim_fwt = largest_dim_fwt

    if fixed_atoms is not None:
        fixed_per_graph = segment_ops.aggregate_nodes(
            fixed_atoms.int(), batch_num_nodes, reduction="sum"
        )
        # remove the fixed atoms from the counts
        batch_num_nodes = batch_num_nodes - fixed_per_graph
        # remove the fixed atoms from the forces
        working_largest_dim_fwt = largest_dim_fwt[not fixed_atoms]

    force_count_within_threshold = segment_ops.aggregate_nodes(
        working_largest_dim_fwt.int(), batch_num_nodes, reduction="sum"
    )
    fwt = force_count_within_threshold == batch_num_nodes

    # count equals batch_num_nodes if all nodes within threshold
    return (fwt & ewt).to(ms.float32).mean().item()


def _remove_fixed_atoms(
        pred_node: Tensor,
        node_target: Tensor,
        batch_n_node: Tensor,
        fix_atoms: Optional[Tensor],
        training: bool,
):
    """We use inf targets on purpose to designate nodes for removal."""
    assert len(pred_node) == len(node_target)
    if fix_atoms is not None and not training:
        pred_node = pred_node[~fix_atoms]
        node_target = node_target[~fix_atoms]
        batch_n_node = segment_ops.aggregate_nodes(
            (~fix_atoms).int(), batch_n_node, reduction="sum"
        )
    return pred_node, node_target, batch_n_node


def bce_loss(
        pred: Tensor, target: Tensor, metric_prefix: str = ""
) -> Tuple:
    """Binary cross-entropy loss with accuracy metric."""
    loss = mint.nn.BCEWithLogitsLoss()(pred, target.float())
    accuracy = binary_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )


def cross_entropy_loss(
        pred: Tensor, target: Tensor, metric_prefix: str = ""
) -> Tuple:
    """Cross-entropy loss with accuracy metric."""
    loss = mint.nn.CrossEntropyLoss()(pred, target.long())
    accuracy = categorical_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )

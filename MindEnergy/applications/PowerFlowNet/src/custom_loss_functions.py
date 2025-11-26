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
Custom Loss Functions for MindSpore PowerFlowNet

Implements aligned loss functions with PyTorch version:
  - MaskedL2Loss: L2 loss with masking support
  - PowerMaskedLoss: Unified L1/L2 loss with per-feature breakdown
  - PowerImbalance: Physics-informed loss for power flow equations
  - MixedMSEPowerImbalance: Combined MSE and power imbalance loss
"""

from typing import Dict, Tuple

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype


class MaskedL2Loss(nn.Cell):
    """
    Masked L2 Loss Function

    Computes L2 loss only on masked (prediction) elements with optional regularization.

    Args:
        regularize (bool): If True, add L2 loss on non-masked elements as regularization
        regcoeff (float): Coefficient for regularization term
    """

    def __init__(self, regularize=True, regcoeff=1.0):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.regularize = regularize
        self.regcoeff = regcoeff

    def construct(self, output: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            output: Model output, shape (N, F)
            target: Target values, shape (N, F)
            mask: Binary mask indicating which elements to predict, shape (N, F)

        Returns:
            Scalar loss value
        """
        # Convert mask to boolean
        masked = ops.cast(mask, mstype.bool_)

        # Select masked elements
        output_masked = ops.masked_select(output, masked)
        target_masked = ops.masked_select(target, masked)

        # Compute loss on masked elements
        loss = self.criterion(output_masked, target_masked)

        # Add regularization on non-masked elements if enabled
        if self.regularize:
            masked_inv = ops.cast(1 - mask, mstype.bool_)
            output_reg = ops.masked_select(output, masked_inv)
            target_reg = ops.masked_select(target, masked_inv)

            if output_reg.shape[0] > 0:  # Only compute if there are non-masked elements
                reg_loss = self.criterion(output_reg, target_reg)
                loss = loss + self.regcoeff * reg_loss

        return loss


class PowerMaskedLoss(nn.Cell):
    """
    Unified Masked Loss with Feature-wise Breakdown

    Supports both L1 and L2 loss types with per-feature breakdown.
    Features:
      - F_0: vm (voltage magnitude)
      - F_1: va (voltage angle)
      - F_2: Pd (active power)
      - F_3: Qd (reactive power)

    Args:
        loss_type (str): 'l2' for MSE loss, 'l1' for L1 loss
    """

    def __init__(self, loss_type: str = 'l2'):
        super().__init__()
        if loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        else:
            raise TypeError("PowerMaskedLoss only supports L1 and MSE loss.")

    def construct(self, output: Tensor, target: Tensor, mask: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            output: Model output, shape (N, F)
            target: Target values, shape (N, F)
            mask: Binary mask, shape (N, F)

        Returns:
            Dict with keys: 'total', 'balanced_total', 'vm', 'va', 'p', 'q'
        """
        # Compute per-element error using criterion
        error = self.criterion(output, target)  # (N, F)

        # Compute per-feature masked loss
        mask_float = ops.cast(mask, ms.float32)
        mask_sum = ops.clamp(ops.sum(mask_float, 0), min=1e-6)  # (F,)
        error_masked = ops.sum(error * mask_float, 0) / mask_sum  # (F,)

        # Build loss terms dictionary
        loss_terms = {}

        # Total loss: weighted by number of predictions per feature
        total_mask_sum = ops.clamp(ops.sum(mask_float), min=1e-6)
        loss_terms['total'] = ops.sum(error_masked * mask_sum) / total_mask_sum

        # Balanced total: mean of per-feature losses
        loss_terms['balanced_total'] = ops.mean(error_masked)

        # Per-feature losses
        loss_terms['vm'] = error_masked[0]  # Voltage magnitude
        loss_terms['va'] = error_masked[1]  # Voltage angle
        loss_terms['p'] = error_masked[2]   # Active power
        loss_terms['q'] = error_masked[3]   # Reactive power

        return loss_terms


# Backward compatibility aliases
def MaskedL2V2():  # pylint: disable=invalid-name
    """Backward compatibility alias for PowerMaskedLoss with L2 loss."""
    return PowerMaskedLoss(loss_type='l2')


def MaskedL1():  # pylint: disable=invalid-name
    """Backward compatibility alias for PowerMaskedLoss with L1 loss."""
    return PowerMaskedLoss(loss_type='l1')


class PowerImbalance(nn.Cell):
    """
    Power Imbalance Loss (Physics-informed)

    Enforces power flow equations on predicted values using Kirchhoff's law.
    Computes power balance at each node based on admittance matrix.

    Power Flow Equations:
        P_ji = g_ij*(e_i*e_j - e_i^2 + f_i*f_j - f_i^2) + b_ij*(f_i*e_j - e_i*f_j)
        Q_ji = g_ij*(f_i*e_j - e_i*f_j) + b_ij*(-e_i*e_j + e_i^2 - f_i*f_j + f_i^2)

    Where:
        - e_i = V_m^i * cos(V_a^i), f_i = V_m^i * sin(V_a^i)
        - g_ij = r / (r^2 + x^2), b_ij = -x / (r^2 + x^2)

    Args:
        xymean: Mean of node features for denormalization
        xystd: Std of node features for denormalization
        edgemean: Mean of edge features for denormalization
        edgestd: Std of edge features for denormalization
        reduction: 'mean' or 'sum'
    """

    base_sn = 100  # kVA
    base_voltage = 345  # kV
    base_ohm = 1190.25  # V^2/SN
    PI = 3.141592653589793

    def __init__(self, xymean, xystd, edgemean, edgestd, reduction='mean'):
        super().__init__()

        # Store normalization parameters - ensure shape is (1, F)
        if isinstance(xymean, Tensor):
            self.xymean = xymean[:1] if xymean.shape[0] > 1 else xymean
            self.xystd = xystd[:1] if xystd.shape[0] > 1 else xystd
            self.edgemean = edgemean
            self.edgestd = edgestd
        else:
            import numpy as np  # pylint: disable=import-outside-toplevel
            xymean = np.array(xymean)
            xystd = np.array(xystd)
            if len(xymean.shape) > 1 and xymean.shape[0] > 1:
                xymean = xymean[:1]
                xystd = xystd[:1]
            self.xymean = Tensor(xymean, dtype=ms.float32)
            self.xystd = Tensor(xystd, dtype=ms.float32)
            self.edgemean = Tensor(edgemean, dtype=ms.float32)
            self.edgestd = Tensor(edgestd, dtype=ms.float32)

        self.reduction = reduction

    def denormalize(self, x: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        """Denormalize node and edge features"""
        x_denorm = x * self.xystd + self.xymean
        edge_attr_denorm = edge_attr * self.edgestd + self.edgemean
        return x_denorm, edge_attr_denorm

    def is_directed(self, edge_index: Tensor) -> bool:
        """Determine if graph is directed by checking one edge"""
        src_0 = int(edge_index[0, 0].asnumpy())
        dst_0 = int(edge_index[1, 0].asnumpy())
        # Check if reverse edge exists
        mask = (edge_index[0] == dst_0) & (edge_index[1] == src_0)
        return not ops.any(mask)

    def undirect_graph(self, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform directed graph to undirected by duplicating edges

        Args:
            edge_index: shape (2, E)
            edge_attr: shape (E, fe)

        Returns:
            Undirected edge_index (2, 2*E) and edge_attr (2*E, fe)
        """
        # Reverse edges
        edge_index_rev = ops.stack([edge_index[1], edge_index[0]], axis=0)
        edge_index_new = ops.concat([edge_index, edge_index_rev], axis=1)
        edge_attr_new = ops.concat([edge_attr, edge_attr], axis=0)
        return edge_index_new, edge_attr_new

    def compute_message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """Calculate injected power P_ji and Q_ji for each edge

        Power flow formula:
            P_ji = g_ij*(e_i*e_j - e_i^2 + f_i*f_j - f_i^2) + b_ij*(f_i*e_j - e_i*f_j)
            Q_ji = g_ij*(f_i*e_j - e_i*f_j) + b_ij*(-e_i*e_j + e_i^2 - f_i*f_j + f_i^2)

        Args:
            x_i: Source node features (num_edges, F)
            x_j: Target node features (num_edges, F)
            edge_attr: Edge features [r, x] (num_edges, 2)

        Returns:
            Power flow [P_ji, Q_ji] (num_edges, 2)
        """
        # Extract resistance and reactance
        r = edge_attr[:, 0:1]  # (num_edges, 1)
        x = edge_attr[:, 1:2]  # (num_edges, 1)

        # Compute conductance g and susceptance b
        z_sq = r ** 2 + x ** 2 + 1e-8  # Add small epsilon for numerical stability
        g_ij = r / z_sq  # (num_edges, 1)
        b_ij = -x / z_sq  # (num_edges, 1)

        # Extract voltage magnitude and angle
        vm_i = x_i[:, 0:1]  # (num_edges, 1)
        va_i = self.PI / 180.0 * x_i[:, 1:2]  # Convert to radians
        vm_j = x_j[:, 0:1]
        va_j = self.PI / 180.0 * x_j[:, 1:2]

        # Convert to rectangular form: e = V*cos(θ), f = V*sin(θ)
        e_i = vm_i * ops.cos(va_i)
        f_i = vm_i * ops.sin(va_i)
        e_j = vm_j * ops.cos(va_j)
        f_j = vm_j * ops.sin(va_j)

        # Power flow equations
        p_ji = g_ij * (e_i * e_j - e_i ** 2 + f_i * f_j - f_i ** 2) + \
               b_ij * (f_i * e_j - e_i * f_j)
        q_ji = g_ij * (f_i * e_j - e_i * f_j) + \
               b_ij * (-e_i * e_j + e_i ** 2 - f_i * f_j + f_i ** 2)

        return ops.concat([p_ji, q_ji], axis=1)  # (num_edges, 2)

    def aggregate(self, messages: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        """Aggregate messages to source nodes (sum aggregation)

        PyTorch MessagePassing with flow='target_to_source':
        - The edge direction is REVERSED internally
        - x_i comes from edge_index[0] (original source, reversed target)
        - x_j comes from edge_index[1] (original target, reversed source)
        - Aggregation happens at the reversed target = edge_index[0] (original source)

        Args:
            messages: Power flow messages (num_edges, 2)
            edge_index: Edge connectivity (2, num_edges)
            num_nodes: Number of nodes

        Returns:
            Aggregated power at each node (num_nodes, 2)
        """
        # With flow='target_to_source', aggregate to edge_index[0] (source nodes)
        target_idx = edge_index[0]  # (num_edges,) - aggregate to SOURCE nodes

        # Scatter add: aggregate messages to target nodes
        # Use unsorted_segment_sum for aggregation
        output = ops.unsorted_segment_sum(messages, target_idx, num_nodes)

        return output  # (num_nodes, 2)

    def construct(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Compute power imbalance loss.

        Formula:
            ΔP_i = Σ_{j∈N_i} P_ji - P_i (injected power)
            ΔQ_i = Σ_{j∈N_i} Q_ji - Q_i (reactive power)
            loss = mean(ΔP^2 + ΔQ^2)

        Args:
            x: Node features [Vm, Va, Pd, Qd, ...], shape (N, F)
            edge_index: Edge connectivity, shape (2, num_edges)
            edge_attr: Edge features [r, x], shape (num_edges, 2)

        Returns:
            Scalar power imbalance loss
        """
        # Convert to undirected if needed
        if self.is_directed(edge_index):
            edge_index, edge_attr = self.undirect_graph(edge_index, edge_attr)

        # Denormalize features
        x, edge_attr = self.denormalize(x, edge_attr)

        num_nodes = x.shape[0]

        # With flow='target_to_source', PyTorch reverses the edge direction internally:
        # - x_i comes from edge_index[0] (original source)
        # - x_j comes from edge_index[1] (original target)
        src_idx = edge_index[0]  # i (original source, reversed target)
        dst_idx = edge_index[1]  # j (original target, reversed source)

        x_i = ops.gather(x, src_idx, axis=0)  # features from edge_index[0] (num_edges, F)
        x_j = ops.gather(x, dst_idx, axis=0)  # features from edge_index[1] (num_edges, F)

        # Compute power flow messages (P_ji, Q_ji)
        messages = self.compute_message(x_i, x_j, edge_attr)  # (num_edges, 2)

        # Aggregate to edge_index[0] (source nodes, because of target_to_source flow)
        aggregated = self.aggregate(messages, edge_index, num_nodes)  # (num_nodes, 2)

        # Compute power imbalance: ΔP = -Σ P_ji + P_i, ΔQ = -Σ Q_ji + Q_i
        delta_p = -aggregated[:, 0:1] + x[:, 2:3]  # (num_nodes, 1)
        delta_q = -aggregated[:, 1:2] + x[:, 3:4]  # (num_nodes, 1)

        # Compute loss: sum of squared imbalances
        dpq = ops.concat([delta_p, delta_q], axis=1)  # (num_nodes, 2)
        dpq_sq = (dpq ** 2).sum(axis=1)  # (num_nodes,)

        # Reduction
        if self.reduction == 'mean':
            loss = dpq_sq.mean()
        else:
            loss = dpq_sq.sum()

        return loss


class MixedMSEPowerImbalance(nn.Cell):
    """
    Combined MSE and Power Imbalance Loss

    Blends standard MSE loss with physics-informed power imbalance term.

    Args:
        xymean, xystd: Normalization for node features
        edgemean, edgestd: Normalization for edge features
        alpha (float): Weight for MSE vs power imbalance (0-1)
    """

    def __init__(self, xymean, xystd, edgemean, edgestd, alpha=0.9):
        super().__init__()

        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.power_imbalance = PowerImbalance(xymean, xystd, edgemean, edgestd)

    def construct(
        self,
        output: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        Args:
            output: Model prediction
            edge_index: Graph edges
            edge_attr: Edge features
            target: Ground truth target

        Returns:
            Mixed loss value
        """
        # MSE component
        mse = self.mse_loss(output, target)

        # Power imbalance component
        pi = self.power_imbalance(output, edge_index, edge_attr)

        # Combined loss
        loss = self.alpha * mse + (1 - self.alpha) * pi

        return loss

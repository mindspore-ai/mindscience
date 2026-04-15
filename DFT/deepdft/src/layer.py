# Copyright 2021 Huawei Technologies Co., Ltd
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""inner layer"""
from typing import Tuple, List
import itertools
import numpy as np
import mindspore as ms
from mindspore import nn, ops


# mindpore tensor version
def pad_and_stack(tensors: List[ms.Tensor]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape and tensors[0].shape[0]:
        max_len = max(tensor.shape[0] for tensor in tensors)
        padded_tensors = [ops.pad(tensor, [0, max_len - tensor.shape[0]], value=0) for tensor in tensors]
        return ops.Stack(axis=0)(padded_tensors)
    return ops.Stack(axis=0)(tensors)

# numpy array version
def pad_and_stack_np(tensors: List[np.ndarray]):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        max_len = max(tensor.shape[0] for tensor in tensors)
        padded_tensors = [np.pad(tensor, [(0, max_len - tensor.shape[0])] + [(0, 0)] * (len(tensor.shape) - 1), mode='constant', constant_values=0) for tensor in tensors]
        return np.stack(padded_tensors, axis=0)
    return np.stack(tensors, axis=0)

class ShiftedSoftplus(nn.Cell):
    """Compute shifted soft-plus activation function"""
    def __init__(self):
        super().__init__()
        self.softplus = ops.Softplus()
        self.log2 = 0.6931471805599453
    def construct(self, x):
        return self.softplus(x) - self.log2


def unpad_and_cat(stacked_seq: ms.Tensor, seq_len: ms.Tensor):
    """Unpad and concatenate by removing batch dimension"""
    unstacked = stacked_seq.unbind(0)
    seq_len = seq_len.asnumpy().tolist()
    unpadded = [ops.narrow(t, 0, 0, l) for (t, l) in zip(unstacked, seq_len)]
    return ops.cat(unpadded, axis=0)


def batch_dim_reduction(stacked_seq: ms.Tensor):
    """Unpad and concatenate by removing batch dimension, Used in GRAPH mode"""
    if not stacked_seq.shape[1]:
        return ops.zeros(stacked_seq.shape[1:])
    else:
        return ops.reshape(stacked_seq, stacked_seq.shape[1:])


def sum_splits(values: ms.Tensor, splits: ms.Tensor):
    """Sum across dimension 0 of the tensor `values` in chunks"""
    # prepare an index vector for summation
    ind = ops.zeros(splits.sum(), dtype=splits.dtype)
    ind[ops.cumsum(splits, axis=0)[:-1]] = 1
    ind = ops.cumsum(ind, axis=0)
    # prepare the output
    sum_y = ops.zeros(splits.shape + values.shape[1:], dtype=values.dtype)
    # do the actual summation
    sum_y.index_add(0, ind, values)
    return sum_y


def calc_distance(
    positions: ms.Tensor, 
    cells: ms.Tensor,
    edges: ms.Tensor,
    edges_displacement: ms.Tensor,
    splits: ms.Tensor,
    return_diff=False,
):
    """Calculate distance of edges"""
    unitcell_repeat = ops.repeat_interleave(cells, splits, axis=0)              # num_edges, 3, 3
    displacement = ops.unsqueeze(edges_displacement, 1).matmul(unitcell_repeat) # num_edges, 1, 3
    displacement = displacement.squeeze(axis=1)                                 # num_edges, 3
    neigh_pos = positions[edges[:, 0]]                                          # num_edges, 3
    neigh_abs_pos = neigh_pos + displacement                                    # num_edges, 3
    this_pos = positions[edges[:, 1]]                                           # num_edges, 3
    diff = this_pos - neigh_abs_pos                                             # num_edges, 3
    dist = ops.sqrt(ops.sum(ops.square(diff), dim=1, keepdim=True))             # num_edges, 1
    if return_diff:
        return dist, diff
    else:
        return dist


def calc_distance_to_probe(
    positions: ms.Tensor,           # num_atoms, 3
    positions_probe: ms.Tensor,     # num_probe_atoms, 3
    cells: ms.Tensor,               # batch_size, 3, 3
    edges: ms.Tensor,               # num_edges, 2
    edges_displacement: ms.Tensor,  # num_edges, 3
    splits: ms.Tensor,              # batch_size
    return_diff=False,
):
    """Calculate distance of edges"""
    if splits[0]:
        unitcell_repeat = ops.repeat_interleave(cells, splits, axis=0)              # num_edges, 3, 3
        displacement = ops.unsqueeze(edges_displacement, 1).matmul(unitcell_repeat) # num_edges, 1, 3
        displacement = displacement.squeeze(axis=1)                                 # num_edges, 3
        neigh_pos = positions[edges[:, 0]]                                          # num_edges, 3
        neigh_abs_pos = neigh_pos + displacement                                    # num_edges, 3
        this_pos = positions_probe[edges[:, 1]]                                     # num_edges, 3
        diff = this_pos - neigh_abs_pos                                             # num_edges, 3
        dist = ops.sqrt(ops.sum(ops.square(diff), dim=1, keepdim=True))             # num_edges, 1
    else:
        diff = ops.zeros((0, 3))
        dist = ops.zeros((0, 1))
    if return_diff:
        return dist, diff
    else:
        return dist


def gaussian_expansion(input_x: ms.Tensor, expand_params: List[Tuple]):
    """
    Expand each feature in a number of Gaussian basis function.
    Expand_params is a list of length input_x.shape[1]

    Args:
        input_x: (num_edges, num_features) tensor
        expand_params: list of None or (start, step, stop) tuples

    Returns:
        (num_edges, ``ceil((stop - start)/step)``) tensor

    """
    if input_x.shape[0]:
        feat_list = ops.unbind(input_x, dim=1)
        expanded_list = []
        for step_tuple, feat in itertools.zip_longest(expand_params, feat_list):
            assert feat is not None, "Too many expansion parameters given"
            if step_tuple:
                start, step, stop = step_tuple
                feat_expanded = ops.unsqueeze(feat, dim=1)
                sigma = step
                basis_mu = ops.arange(start, stop, step, dtype=input_x.dtype)
                expanded_list.append(
                    ops.exp(-((feat_expanded - basis_mu) ** 2) / (2.0 * sigma ** 2))
                )
            else:
                expanded_list.append(ops.unsqueeze(feat, 1))
        return ops.cat(expanded_list, axis=1)
    else:
        step_tuple = expand_params[0]
        start, step, stop = step_tuple
        return ops.zeros((0, int((stop - start) / step)))
    

def sinc_expansion(input_x: ms.Tensor, expand_params: List[Tuple]):
    """Expand each feature in a sinc-like basis function expansion."""
    if input_x.shape[0]:
        feat_list = ops.unbind(input_x, dim=1)
        expanded_list = []
        min_length = min(len(expand_params), len(feat_list))
        for i in range(min_length):
            step_tuple = expand_params[i]
            feat = feat_list[i]
            assert feat is not None, "Too many expansion parameters given"
            n, cutoff = step_tuple
            feat_expanded = ops.unsqueeze(feat, dim=1)
            n_range = ops.arange(n, dtype=input_x.dtype) + 1
            # multiplication by pi n_range / cutoff is done in original painn for some reason
            out = ops.sinc(n_range/cutoff*feat_expanded)*np.pi*n_range/cutoff
            expanded_list.append(out)

        for feat in feat_list[min_length:]:
            expanded_list.append(ops.unsqueeze(feat, 1))
        return ops.cat(expanded_list, axis=1)
    else:
        step_tuple = expand_params[0]
        n, _ = step_tuple
        return ops.zeros((0, n))


def cosine_cutoff(distance: ms.Tensor, cutoff: float):
    """Calculate cutoff value based on distance."""
    return ops.where(
        distance < cutoff,
        0.5 * (ops.cos(np.pi * distance / cutoff) + 1),
        ms.Tensor(0.0, dtype=ms.float32),
    )


class SchnetMessageFunction(nn.Cell):
    def __init__(self, input_size, edge_size, output_size, hard_cutoff):
        super().__init__()
        self.msg_function_edge = nn.SequentialCell(
            nn.Dense(edge_size, output_size),
            ShiftedSoftplus(),
            nn.Dense(output_size, output_size),
        )
        self.msg_function_node = nn.SequentialCell(
            nn.Dense(input_size, input_size),
            ShiftedSoftplus(),
            nn.Dense(input_size, output_size),
        )

        self.soft_cutoff_func = lambda x: 1.0 - ops.sigmoid(
            5 * (x - (hard_cutoff - 1.5))
        )

    def construct(self, node_state, edge_state, edge_distance):
        gates = self.msg_function_edge(edge_state) * self.soft_cutoff_func(
            edge_distance
        )
        nodes = self.msg_function_node(node_state)
        return nodes * gates


class Interaction(nn.Cell):
    def __init__(self, node_size, edge_size, cutoff, include_receiver=False):
        super().__init__()

        self.message_sum_module = MessageSum(
            node_size, edge_size, cutoff, include_receiver
        )

        self.state_transition_function = nn.SequentialCell(
            nn.Dense(node_size, node_size),
            ShiftedSoftplus(),
            nn.Dense(node_size, node_size)
        )

    def construct(self, node_state, edges, edge_state, edges_distance):
        # Compute sum of messages
        message_sum = self.message_sum_module(
            node_state, edges, edge_state, edges_distance
        )
        # State transition
        new_state = node_state + self.state_transition_function(message_sum)

        return new_state


class MessageSum(nn.Cell):
    def __init__(self, node_size, edge_size, cutoff, include_receiver):
        super().__init__()

        self.include_receiver = include_receiver

        if include_receiver:
            input_size = node_size * 2
        else:
            input_size = node_size

        self.message_function = SchnetMessageFunction(
            input_size, edge_size, node_size, cutoff
        )

    def construct(self, node_state, edges, edge_state, edges_distance, receiver_nodes=None):
        # Compute all messages
        if edges.shape[0]:
            if self.include_receiver:
                if receiver_nodes is not None:
                    senders = node_state[edges[:, 0]]
                    receivers = receiver_nodes[edges[:, 1]]
                    nodes = ops.cat((senders, receivers), axis=1)
                else:
                    num_edges = edges.shape[0]
                    nodes = ops.reshape(node_state[edges], (num_edges, -1))
            else:
                nodes = node_state[edges[:, 0]]  # Only include sender in messages
            messages = self.message_function(nodes, edge_state, edges_distance)

            # Sum messages
            if receiver_nodes is not None:
                message_sum = ops.zeros_like(receiver_nodes)
            else:
                message_sum = ops.zeros_like(node_state)

            message_sum = ops.tensor_scatter_add(message_sum, ops.expand_dims(edges[:, 1], axis=1), messages)

        else:
            if receiver_nodes is not None:
                message_sum = ops.zeros_like(receiver_nodes)
            else:
                message_sum = ops.zeros_like(node_state)
        return message_sum


class EdgeUpdate(nn.Cell):
    def __init__(self, edge_size, node_size):
        super().__init__()

        self.node_size = node_size
        self.edge_update_mlp = nn.SequentialCell(
            nn.Dense(2 * node_size + edge_size, 2 * edge_size),
            ShiftedSoftplus(),
            nn.Dense(2 * edge_size, edge_size),
        )

    def construct(self, edge_state, edges, node_state):
        combined = ops.cat(
            (node_state[edges].view(-1, 2 * self.node_size), edge_state), axis=1
        )
        return self.edge_update_mlp(combined)


class PaiNNUpdate(nn.Cell):
    """PaiNN style update network. Models the interaction between scalar and vectorial part"""
    def __init__(self, node_size):
        super().__init__()
        self.DenseU = nn.Dense(node_size, node_size, has_bias=False)
        self.DenseV = nn.Dense(node_size, node_size, has_bias=False)
        self.combined_mlp = nn.SequentialCell(
            nn.Dense(2 * node_size, node_size),
            nn.SiLU(),
            nn.Dense(node_size, 3 * node_size),
        )

    def construct(self, node_state_scalar, node_state_vector):
        Uv = self.DenseU(node_state_vector)  # num_nodes, 3, node_size
        Vv = self.DenseV(node_state_vector)  # num_nodes, 3, node_size
        # Vv_norm = ops.norm(Vv, dim=1, keepdim=False)  # num_nodes, node_size
        Vv_norm = ops.LpNorm(axis=1, keep_dims=False)(Vv)
        mlp_input = ops.cat(
            (node_state_scalar, Vv_norm), axis=1
        )  # num_nodes, node_size*2
        mlp_output = self.combined_mlp(mlp_input)
        a_ss, a_sv, a_vv = ops.split(
            mlp_output, node_state_scalar.shape[1], axis=1
        )  # num_nodes, node_size
        inner_prod = ops.sum(Uv * Vv, dim=1)  # num_nodes, node_size
        delta_v = ops.unsqueeze(a_vv, dim=1) * Uv  # num_nodes, 3, node_size
        delta_s = a_ss + a_sv * inner_prod  # num_nodes, node_size
        return node_state_scalar + delta_s, node_state_vector + delta_v


class PaiNNInteraction(nn.Cell):
    """Interaction network"""
    def __init__(self, node_size, edge_size, cutoff):
        super().__init__()
        self.filter_layer = nn.Dense(edge_size, 3 * node_size)
        self.cutoff = cutoff
        self.scalar_message_mlp = nn.SequentialCell(
            nn.Dense(node_size, node_size),
            nn.SiLU(),
            nn.Dense(node_size, 3 * node_size),
        )

    def construct(
        self,
        node_state_scalar,
        node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        # Compute all messages
        edge_vector_normalised = edge_vector / ops.maximum(
            # ops.norm(edge_vector, dim=1, keepdim=True), ms.Tensor(1e-12)
            ops.LpNorm(axis=1, keep_dims=True)(edge_vector), ms.Tensor(1e-12)
        )  # num_edges, 3
        filter_weight = self.filter_layer(edge_state)  # num_edges, 3*node_size
        filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)
        scalar_output = self.scalar_message_mlp(
            node_state_scalar
        )  # num_nodes, 3*node_size
        scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
        filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

        gate_state_vector, gate_edge_vector, gate_node_state = ops.split(
            filter_output, node_state_scalar.shape[1], axis=1
        )
        gate_state_vector = ops.unsqueeze(
            gate_state_vector, 1
        )  # num_edges, 1, node_size
        gate_edge_vector = ops.unsqueeze(
            gate_edge_vector, 1
        )  # num_edges, 1, node_size
        # Only include sender in messages
        messages_scalar = node_state_scalar[edges[:, 0]] * gate_node_state
        messages_state_vector = node_state_vector[
            edges[:, 0]
        ] * gate_state_vector + gate_edge_vector * ops.unsqueeze(
            edge_vector_normalised, 2
        )

        # Sum messages
        message_sum_scalar = ops.zeros_like(node_state_scalar)
        message_sum_scalar = ops.tensor_scatter_add(message_sum_scalar, ops.expand_dims(edges[:, 1], axis=1), messages_scalar)
        message_sum_vector = ops.zeros_like(node_state_vector)
        message_sum_vector = ops.tensor_scatter_add(message_sum_vector, ops.expand_dims(edges[:, 1], axis=1), messages_state_vector)

        # State transition
        new_state_scalar = node_state_scalar + message_sum_scalar
        new_state_vector = node_state_vector + message_sum_vector

        return new_state_scalar, new_state_vector


class PaiNNInteractionOneWay(nn.Cell):
    """Same as Interaction network, but the receiving nodes are differently indexed from the sending nodes"""
    def __init__(self, node_size, edge_size, cutoff):
        super().__init__()
        self.filter_layer = nn.Dense(edge_size, 3 * node_size)
        self.cutoff = cutoff
        self.scalar_message_mlp = nn.SequentialCell(
            nn.Dense(node_size, node_size),
            nn.SiLU(),
            nn.Dense(node_size, 3 * node_size),
        )
        # Ignore messages gate (not part of original PaiNN network)
        self.update_gate_mlp = nn.SequentialCell(
            nn.Dense(node_size, 2 * node_size),
            nn.SiLU(),
            nn.Dense(2 * node_size, 2 * node_size),
            nn.Sigmoid()
        )

    def construct(
        self,
        sender_node_state_scalar,
        sender_node_state_vector,
        receiver_node_state_scalar,
        receiver_node_state_vector,
        edge_state,
        edge_vector,
        edge_distance,
        edges,
    ):
        # Compute all messages
        if edge_state.shape[0]:
            edge_vector_normalised = edge_vector / ops.maximum(
                # ops.norm(edge_vector, dim=1, keepdim=True), ms.Tensor(1e-12)
                ops.LpNorm(axis=1, keep_dims=True)(edge_vector), ms.Tensor(1e-12)
            )  # num_edges, 3

            filter_weight = self.filter_layer(edge_state)  # num_edges, 3*node_size
            filter_weight = filter_weight * cosine_cutoff(edge_distance, self.cutoff)

            scalar_output = self.scalar_message_mlp(
                sender_node_state_scalar
            )  # num_nodes, 3*node_size
            scalar_output = scalar_output[edges[:, 0]]  # num_edges, 3*node_size
            filter_output = filter_weight * scalar_output  # num_edges, 3*node_size

            gate_state_vector, gate_edge_vector, gate_node_state = ops.split(
                filter_output, sender_node_state_scalar.shape[1], axis=1
            )

            gate_state_vector = ops.unsqueeze(
                gate_state_vector, dim=1
            )  # num_edges, 1, node_size
            gate_edge_vector = ops.unsqueeze(
                gate_edge_vector, dim=1
            )  # num_edges, 1, node_size

            # Only include sender in messages
            messages_scalar = sender_node_state_scalar[edges[:, 0]] * gate_node_state
            messages_state_vector = sender_node_state_vector[
                edges[:, 0]
            ] * gate_state_vector + gate_edge_vector * ops.unsqueeze(
                edge_vector_normalised, dim=2
            )

            # Sum messages
            message_sum_scalar = ops.zeros_like(receiver_node_state_scalar)
            message_sum_scalar = ops.tensor_scatter_add(
                message_sum_scalar,
                ops.expand_dims(edges[:, 1], axis=1),
                messages_scalar
            )
            message_sum_vector = ops.zeros_like(receiver_node_state_vector)
            message_sum_vector = ops.tensor_scatter_add(
                message_sum_vector,
                ops.expand_dims(edges[:, 1], axis=1),
                messages_state_vector
            )
        else:
            message_sum_scalar = ops.zeros_like(receiver_node_state_scalar)
            message_sum_vector = ops.zeros_like(receiver_node_state_vector)

        # State transition
        update_gate_scalar, update_gate_vector = ops.split(
            self.update_gate_mlp(message_sum_scalar),
            receiver_node_state_scalar.shape[1],
            axis=1,
        )
        update_gate_vector = ops.unsqueeze(
            update_gate_vector, dim=1
        )  # num_nodes, 1, node_size
        new_state_scalar = (
            update_gate_scalar * receiver_node_state_scalar
            + (1.0 - update_gate_scalar) * message_sum_scalar
        )
        new_state_vector = (
            update_gate_vector * receiver_node_state_vector
            + (1.0 - update_gate_vector) * message_sum_vector
        )

        return new_state_scalar, new_state_vector



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
"""density model"""
from typing import List, Dict
import ase
import mindspore as ms
from mindspore import nn, ops
from . import layer

class PainnDensityModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size=30,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_model = PainnAtomRepresentationModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

        self.probe_model = PainnProbeMessageModel(
            num_interactions,
            hidden_state_size,
            cutoff,
            distance_embedding_size,
        )

    def construct(self, input_dict):
        atom_representation_scalar, atom_representation_vector = self.atom_model(input_dict)
        probe_result = self.probe_model(input_dict, atom_representation_scalar, atom_representation_vector)
        return probe_result


class PainnAtomRepresentationModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.interactions = nn.CellList(
            [
                layer.PaiNNInteraction(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.CellList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Atom embeddings
        self.atom_embeddings = nn.Embedding(
            len(ase.data.atomic_numbers), self.hidden_state_size
        )

    def construct(self, input_dict):
        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["atom_edges_displacement"], input_dict["num_atom_edges"]
        )
        edge_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),  # device=input_dict["num_nodes"].device
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["atom_edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_atom_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = ops.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            atom_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_atom_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        nodes_list_scalar = []
        nodes_list_vector = []
        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)
            nodes_list_scalar.append(nodes_scalar)
            nodes_list_vector.append(nodes_vector)

        return nodes_list_scalar, nodes_list_vector


class PainnProbeMessageModel(nn.Cell):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        distance_embedding_size,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = distance_embedding_size

        # Setup interaction networks
        self.message_layers = nn.CellList(
            [
                layer.PaiNNInteractionOneWay(
                    hidden_state_size, self.distance_embedding_size, self.cutoff
                )
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.CellList(
            [layer.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_function = nn.SequentialCell(
            nn.Dense(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            nn.Dense(hidden_state_size, 1),
        )

    def construct(
        self,
        input_dict: Dict[str, ms.Tensor],
        atom_representation_scalar: List[ms.Tensor],
        atom_representation_vector: List[ms.Tensor],
    ):
        # Unpad and concatenate edges and features into batch (0th) dimension
        atom_xyz = layer.unpad_and_cat(input_dict["atom_xyz"], input_dict["num_nodes"])
        probe_xyz = layer.unpad_and_cat(input_dict["probe_xyz"], input_dict["num_probes"])
        edge_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),  # device=input_dict["num_nodes"].device
                    input_dict["num_nodes"][:-1],
                )
            ),
            axis=0,
        )
        edge_offset = edge_offset[:, None, None]

        # Unpad and concatenate probe edges into batch (0th) dimension
        probe_edges_displacement = layer.unpad_and_cat(
            input_dict["probe_edges_displacement"], input_dict["num_probe_edges"]
        )
        edge_probe_offset = ops.cumsum(
            ops.cat(
                (
                    ms.Tensor([0], dtype=ms.int32),  # device=input_dict["num_probes"].device
                    input_dict["num_probes"][:-1],
                )
            ),
            axis=0,
        )
        edge_probe_offset = edge_probe_offset[:, None, None]
        edge_probe_offset = ops.cat((edge_offset, edge_probe_offset), axis=2)
        probe_edges = input_dict["probe_edges"]
        probe_edges = probe_edges + edge_probe_offset if probe_edges.shape[1] else probe_edges
        probe_edges = layer.unpad_and_cat(probe_edges, input_dict["num_probe_edges"])

        # Compute edge distances
        probe_edges_distance, probe_edges_diff = layer.calc_distance_to_probe(
            atom_xyz,
            probe_xyz,
            input_dict["cell"],
            probe_edges,
            probe_edges_displacement,
            input_dict["num_probe_edges"],
            return_diff=True,
        )

        # Expand edge features in sinc basis
        edge_state = layer.sinc_expansion(
            probe_edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        # Apply interaction layers
        probe_state_scalar_size = (
            ops.sum(input_dict["num_probes"]).asnumpy().item(),
            self.hidden_state_size
        )
        probe_state_vector_size = (
            ops.sum(input_dict["num_probes"]).asnumpy().item(),
            3,
            self.hidden_state_size
        )
        probe_state_scalar = ops.zeros(probe_state_scalar_size)
        probe_state_vector = ops.zeros(probe_state_vector_size)

        for msg_layer, update_layer, atom_nodes_scalar, atom_nodes_vector in zip(
            self.message_layers,
            self.scalar_vector_update,
            atom_representation_scalar,
            atom_representation_vector,
        ):

            probe_state_scalar, probe_state_vector = msg_layer(
                atom_nodes_scalar,
                atom_nodes_vector,
                probe_state_scalar,
                probe_state_vector,
                edge_state,
                probe_edges_diff,
                probe_edges_distance,
                probe_edges,
            )
            probe_state_scalar, probe_state_vector = update_layer(
                probe_state_scalar, probe_state_vector
            )

        # Restack probe states
        probe_output = self.readout_function(probe_state_scalar).squeeze(1)
        probe_output = layer.pad_and_stack(
            ops.split(
                probe_output,
                input_dict["num_probes"].asnumpy().tolist(),
                axis=0,
            )
        )

        return probe_output


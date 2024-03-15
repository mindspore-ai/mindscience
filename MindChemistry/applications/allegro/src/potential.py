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
# ==============================================================================
"""
Main program of Cybertron
"""
import os
from typing import Union, List, Tuple

import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell
from mindspore.train import save_checkpoint
from mindspore.train._utils import _make_directory

from mindchemistry.cell.basic_block import MLPMixPrecision as MLP
from mindchemistry.cell.geonet import Allegro
from mindchemistry.graph.graph import AggregateEdgeToNode, AggregateNodeToGlobal
from src.allegro_embedding import AllegroEmbedding

_cur_dir = os.getcwd()


class PotentialForce(Cell):
    """Potential_Force
    """

    def __init__(self, potential_net):
        super().__init__()
        self.potential_net = potential_net
        self.grad = ms.grad(self.potential_net, grad_position=1)

    def construct(self, x, pos, edge_index, batch, batch_size):
        atom_types = x.reshape(-1, 1)
        pos = pos.reshape(-1, 3)
        total_energy = self.potential_net(atom_types, pos, edge_index, batch, batch_size)
        forces = ops.neg(self.grad(atom_types, pos, edge_index, batch, batch_size))
        return total_energy, forces


class Potential(Cell):
    """Potential
    """

    # pylint: disable=W0102
    def __init__(
            self,
            model: Union[Allegro, dict, str],
            embedding: Union[AllegroEmbedding, dict, str] = None,
            avg_num_neighbor: float = 1.0,
            edge_eng_mlp_latent_dimensions: List = [32, 1]
    ):

        super().__init__()

        self.scatter = AggregateEdgeToNode(dim=1)
        self.scatter_node_to_global = AggregateNodeToGlobal()

        # build embedding and model
        self.model = model
        self.embedding = embedding

        self.cutoff = None
        self.large_dis = 5e4
        self.avg_num_neighbor = avg_num_neighbor

        self.edge_eng = MLP(
            input_dim=self.model.latent_dim,
            hidden_dims=edge_eng_mlp_latent_dimensions,
            activation_fn=None,
            weight_init="uniform",
            dtype=ms.float32
        )

    def save_checkpoint(self, ckpt_file_name: str, directory: str = None, append_dict: str = None):
        """save checkpoint file"""
        if directory is not None:
            directory = _make_directory(directory)
        else:
            directory = _cur_dir
        ckpt_file = os.path.join(directory, ckpt_file_name)
        if os.path.exists(ckpt_file):
            os.remove(ckpt_file)
        save_checkpoint(self, ckpt_file, append_dict=append_dict)
        return self

    def construct(
            self,
            x: Tensor = None,
            pos: Tensor = None,
            edge_index: Tensor = None,
            batch: Tensor = None,
            batch_size: int = 1,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """_summary_

        Args:
            x (Tensor, optional): x. Defaults to None.
            pos (Tensor, optional): pos. Defaults to None.
            edge_index (Tensor, optional): edge_index. Defaults to None.
            batch (Tensor, optional): batch. Defaults to None.
            batch_size (int, optional): batch_size. Defaults to 1.

        Returns:
            Union[Tensor, Tuple[Tensor]]: total_energy
        """
        atom_types = x.reshape(-1, 1)
        pos = pos.reshape(-1, 3)
        embedding_out = self.embedding(
            atom_types=atom_types,
            pos=pos,
            edge_index=edge_index,
        )

        edge_features = self.model(embedding_out, edge_index, atom_types)

        edge_energy = self.edge_eng(edge_features)

        atoms_shape = list(edge_energy.shape)
        atoms_shape[0] = atom_types.shape[0]
        atom_energy = self.scatter(
            edge_attr=edge_energy, edge_index=edge_index, out=ops.zeros(tuple(atoms_shape), dtype=edge_energy.dtype)
        )

        factor = ops.sqrt(Tensor(self.avg_num_neighbor))
        atom_energy = ops.div(atom_energy, factor)

        total_energy = self.scatter_node_to_global(
            node_attr=atom_energy, batch=batch, out=ops.zeros((batch_size, 1), dtype=atom_energy.dtype)
        )

        return total_energy

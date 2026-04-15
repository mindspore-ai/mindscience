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
"""flow file"""
import math

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops
from mindchemistry.graph.graph import (AggregateNodeToGlobal, LiftGlobalToNode)

from models.lattice import LatticePolarDecomp


def replace_nan_with_zero(tensor):
    """Replace nan in tensor with zero to avoid numerical errors.
    """
    is_nan = ops.IsNan()(tensor)
    zeros = ops.Fill()(ms.float32, tensor.shape, 0.0)
    result = ops.Select()(is_nan, zeros, tensor)
    return result


class SinusoidalTimeEmbeddings(nn.Cell):
    """ Embedding for the time step in flow.
        Referring the implementation details in the paper Attention is all you need. """

    def __init__(self, dim):
        super(SinusoidalTimeEmbeddings, self).__init__()
        self.dim = dim

    def construct(self, time):
        """construct

        Args:
            time (Tensor): flow time step

        Returns:
            Tensor: Time embedding
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = ops.Exp()(mnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = ops.Concat(axis=-1)(
            (ops.Sin()(embeddings), ops.Cos()(embeddings)))
        return embeddings


def lattice_params_to_matrix_mindspore(lengths, angles):
    """Batched MindSpore version to compute lattice matrix from params.

    Args:
        lengths (Tensor): Tensor of shape (N, 3), unit A
        angles (Tensor):: Tensor of shape (N, 3), unit degree
    Returns:
        Tensor: Tensor of shape (N, 3, 3)
    """
    angles_r = ops.deg2rad(angles)
    coses = ops.cos(angles_r)
    sins = ops.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = ops.clip_by_value(val, -1., 1.)
    gamma_star = ops.acos(val)

    zero_tensor = ops.zeros((lengths.shape[0],))

    vector_a = ops.stack(
        [lengths[:, 0] * sins[:, 1], zero_tensor, lengths[:, 0] * coses[:, 1]],
        axis=1)

    vector_b = ops.stack([
        -lengths[:, 1] * sins[:, 0] * ops.cos(gamma_star), lengths[:, 1] *
        sins[:, 0] * ops.sin(gamma_star), lengths[:, 1] * coses[:, 0]
    ],
                         axis=1)

    vector_c = ops.stack([zero_tensor, zero_tensor, lengths[:, 2]], axis=1)

    return ops.stack([vector_a, vector_b, vector_c], axis=1)


class CSPFlow(nn.Cell):
    """Flow model used in CrystalFlow
    """

    def __init__(self,
                 decoder,
                 time_dim=256,
                 sigma=0.1):
        """Initialization

        Args:
            decoder (nn.cell): Nerual network as denoiser for flow.
            time_dim (int): The dimension of time embedding. Defaults to 256.
            sigma (float): the standard deviation of Gaussian prior where lattice_polar_0 is sampled
        """
        super(CSPFlow, self).__init__()
        self.time_dim = time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.lattice_model = LatticePolarDecomp()
        self.lift_node = LiftGlobalToNode()
        self.aggre_graph = AggregateNodeToGlobal('mean')
        self.decoder = decoder
        self.sigma = sigma
        self.relu = nn.ReLU()

    def construct(self, batch_num_graphs, batch_atom_types, batch_lengths,
                  batch_angles_step, batch_lattice_polar, batch_num_atoms_step,
                  batch_frac_coords, batch_node2graph,
                  batch_edge_index, node_mask, edge_mask, batch_mask):
        """Training process for diffution.

        Args:
            batch_num_graphs (Tensor): Batch size with shape (1,)
            batch_atom_types (Tensor): Atom types of nodes in a batch of graph. Shape: (num_atoms,)
            batch_lengths (Tensor): Lattices lengths in a batch of graph. Shape: (batchsize, 3)
            batch_angles (Tensor): Lattice angles in a batch of graph. Shape: (batchsize, 3)
            batch_frac_coords (Tensor): Fractional coordinates of nodes in
                a batch of graph. (num_atoms, 3)
            batch_node2graph (Tensor): Graph index for each node. Shape: (num_atoms,)
            batch_edge_index (Tensor): Beginning and ending node index for each edge.
                Shape: (2, num_edges)
            node_mask (Tensor): Node mask for padded tensor. Shape: (num_atoms,)
            edge_mask (Tensor): Edge mask for padded tensor. Shape: (num_edges,)
            batch_mask (Tensor): Graph mask for padded tensor. Shape: (batchsize,)

        Returns:
            Tuple(Tensor, Tensor, Tensor, Tensor): Return the ground truth
                and predicted flow terms of lattice polar and fractional
                coordinates respectively.
        """
        _, _, _, _ = batch_num_graphs, batch_angles_step, batch_num_atoms_step, batch_mask
        times = ops.rand(batch_lengths.shape[0])
        time_emb = self.time_embedding(times)

        lattice_polar = batch_lattice_polar
        frac_coords = batch_frac_coords

        lattice_polar_0 = self.lattice_model.sample(batch_lengths.shape[0], self.sigma)
        frac_coords_0 = ops.rand_like(frac_coords)

        tar_l = lattice_polar - lattice_polar_0
        tar_f = (frac_coords - frac_coords_0 - 0.5) % 1 - 0.5

        tar_f = ops.mul(tar_f, ops.reshape(node_mask, (-1, 1)))

        l_expand_dim = (slice(None),) + (None,) * (tar_l.dim() - 1)  # in this case is (:, None, None)
        input_lattice = lattice_polar_0 + times[l_expand_dim] * tar_l
        input_frac_coords = frac_coords_0 + self.lift_node(times[:, None], batch_node2graph) * tar_f



        #flow
        pred_l, pred_f = self.decoder(time_emb, batch_atom_types,
                                      input_frac_coords, input_lattice,
                                      batch_node2graph, batch_edge_index,
                                      node_mask, edge_mask)



        return pred_l, tar_l, pred_f, tar_f

    #sample and evaluate

    def get_anneal_factor(self, t, slope: float = 0.0, offset: float = 0.0):
        if not isinstance(t, ms.Tensor):
            t = ms.tensor(t)
        return 1 + slope * self.relu(t - offset)

    def post_decoder_on_sample(
            self, pred, t,
            anneal_slope=0.0, anneal_offset=0.0,
    ):
        """apply anneal to pred_f"""

        pred_l, pred_f = pred
        anneal_factor = self.get_anneal_factor(t, anneal_slope, anneal_offset)

        pred_f *= anneal_factor

        return pred_l, pred_f

    def sample(self,
               batch_node2graph,
               node_mask,
               edge_mask,
               batch_mask,
               batch_atom_types,
               batch_edge_index,
               batch_num_atoms,
               n=1000,
               anneal_slope=0.0,
               anneal_offset=0.0):
        """Generation process of flow. Note: For simplicity, we use x instead of frac_coords and
            l instead of lattice.

        Args:
            batch_atom_types (Tensor): Atom types of nodes in a batch of graph. Shape: (num_atoms,)
            batch_node2graph (Tensor): Graph index for each node. Shape: (num_atoms,)
            batch_edge_index (Tensor): Beginning and ending node index for each edge.
                Shape: (2, num_edges)
            node_mask (Tensor): Node mask for padded tensor. Shape: (num_atoms,)
            edge_mask (Tensor): Edge mask for padded tensor. Shape: (num_edges,)
            batch_mask (Tensor): Graph mask for padded tensor. Shape: (batchsize,)
            N (int): the steps of flow
            anneal_slope(float):
            anneal_offset(float):
        Returns:
            Tuple(dict, Tensor, Tensor): Return the traj of flow process, the fractional coordinates and
                generated lattice matrix for the input atom types of each crystal.
        """
        batch_size_pad = batch_mask.shape[0]
        num_node_pad = node_mask.shape[0]    #shape: (2819,) where 2819 is the largest numbers of atoms in evry batches

        l_0 = self.lattice_model.sample(batch_size_pad, self.sigma)
        x_0 = ops.UniformReal()((num_node_pad, 3)) % 1.0


        l_t = l_0
        x_t = x_0
        l_mat_t = LatticePolarDecomp().build(l_t)
        traj = {
            0: {
                'num_atoms': batch_num_atoms,
                'atom_types': batch_atom_types,
                'frac_coords': x_t,
                'lattices': l_mat_t,
            }
        }
        for t in range(1, n+1):
            t_stamp = t / n
            times = ops.Fill()(ms.float32, (batch_size_pad,), t_stamp)
            time_emb = self.time_embedding(times)
            # ========= pred each step start =========

            pred = self.decoder(time_emb, batch_atom_types, x_t, l_t,
                                batch_node2graph, batch_edge_index,
                                node_mask, edge_mask)

            pred = self.post_decoder_on_sample(
                pred,
                t=t_stamp,
                anneal_slope=anneal_slope, anneal_offset=anneal_offset,
            )

            pred_l, pred_f = pred

            # ========= pred each step end =========


            # ========= update each step start =========

            l_t = l_t + pred_l / n
            x_t = x_t + pred_f / n
            x_t = x_t % 1.0

            # ========= update each step end =========

            # ========= build trajectory start =========
            l_mat_t = LatticePolarDecomp().build(l_t)
            traj[t] = {
                t: {
                    'num_atoms': batch_num_atoms,
                    'atom_types': batch_atom_types,
                    'frac_coords': x_t,
                    'lattices': l_mat_t,
                }
            }

        return traj, x_t, l_mat_t

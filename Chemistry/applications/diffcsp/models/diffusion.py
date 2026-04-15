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
"""diffution file"""
import math

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops
from mindchemistry.graph.graph import (AggregateNodeToGlobal, LiftGlobalToNode)

from models.diff_utils import (BetaScheduler, SigmaScheduler,
                               d_log_p_wrapped_normal_ms)


def replace_nan_with_zero(tensor):
    """Replace nan in tensor with zero to avoid numerical errors.
    """
    is_nan = ops.IsNan()(tensor)
    zeros = ops.Fill()(ms.float32, tensor.shape, 0.0)
    result = ops.Select()(is_nan, zeros, tensor)
    return result


class SinusoidalTimeEmbeddings(nn.Cell):
    """ Embedding for the time step in diffution.
        Referring the implementation details in the paper Attention is all you need. """

    def __init__(self, dim):
        super(SinusoidalTimeEmbeddings, self).__init__()
        self.dim = dim

    def construct(self, time):
        """construct

        Args:
            time (Tensor): diffution time step

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


class CSPDiffusion(nn.Cell):
    """Diffution model used in DiffCSP
    """

    def __init__(self,
                 decoder,
                 time_dim=256,
                 timesteps=1000,
                 scheduler_mode='cosine',
                 sigma_begin=0.005,
                 sigma_end=0.5):
        """Initialization

        Args:
            decoder (nn.cell): Nerual network as denoiser for diffution.
            time_dim (int): The dimension of time embedding. Defaults to 256.
            timesteps (int): The number of time steps that diffution model used. Defaults to 1000.
            scheduler_mode (str): The scheduler mode for lattice DDPM to get the beta
                in each time step. Defaults to 'cosine'.
            sigma_begin (float): The beginning sigma used in fractiaonal coordinates SDEs.
                Defaults to 0.005.
            sigma_end (float): The ending sigma used in fractiaonal coordinates SDEs.
                Defaults to 0.5.
        """
        super(CSPDiffusion, self).__init__()
        self.beta_scheduler = BetaScheduler(timesteps=timesteps,
                                            scheduler_mode=scheduler_mode)
        self.sigma_scheduler = SigmaScheduler(timesteps=timesteps,
                                              sigma_begin=sigma_begin,
                                              sigma_end=sigma_end)
        self.time_dim = time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.lift_node = LiftGlobalToNode()
        self.aggre_graph = AggregateNodeToGlobal('mean')
        self.decoder = decoder

    def construct(self, batch_num_graphs, batch_atom_types, batch_lengths,
                  batch_angles, batch_frac_coords, batch_node2graph,
                  batch_edge_index, node_mask, edge_mask, batch_mask):
        """Training process for diffution.

        Args:
            batch_num_graphs (Tensor): Batch size with shape (1,)
            batch_atom_types (Tensor): Atom types of nodes in a batch of graph. Shape: (num_atoms,)
            batch_lengths (Tensor): Lattice lengths in a batch of graph. Shape: (batchsize, 3)
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
                and predicted denoising terms of lattice matrix and fractional
                coordinates respectively.
        """
        _ = batch_num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_lengths.shape[0])
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = ops.Sqrt()(alphas_cumprod)
        c1 = ops.Sqrt()(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_mindspore(batch_lengths,
                                                      batch_angles)
        lattices = replace_nan_with_zero(lattices)
        lattices = ops.mul(lattices, ops.reshape(batch_mask, (-1, 1, 1)))

        frac_coords = batch_frac_coords

        rand_l, rand_x = ops.StandardNormal()(
            lattices.shape), ops.StandardNormal()(frac_coords.shape)

        rand_l = ops.mul(rand_l, ops.reshape(batch_mask, (-1, 1, 1)))
        rand_x = ops.mul(rand_x, ops.reshape(node_mask, (-1, 1)))

        input_lattice = c0[:, None, None] * lattices + c1[:, None,
                                                          None] * rand_l

        sigmas_per_atom = self.lift_node(sigmas[:, None], batch_node2graph)
        sigmas_norm_per_atom = self.lift_node(sigmas_norm[:, None],
                                              batch_node2graph)

        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.
        input_frac_coords = ops.mul(input_frac_coords,
                                    ops.reshape(node_mask, (-1, 1)))

        pred_l, pred_x = self.decoder(time_emb, batch_atom_types,
                                      input_frac_coords, input_lattice,
                                      batch_node2graph, batch_edge_index,
                                      node_mask, edge_mask)

        tar_x = d_log_p_wrapped_normal_ms(
            sigmas_per_atom * rand_x,
            sigmas_per_atom) / ops.Sqrt()(sigmas_norm_per_atom)
        tar_x = ops.mul(tar_x, ops.reshape(node_mask, (-1, 1)))

        return pred_l, rand_l, pred_x, tar_x

    @ms.jit
    def denoise(self, time_emb, batch_atom_types, x_t, l_t, batch_node2graph,
                batch_edge_index, node_mask, edge_mask):
        """
        Running denoiser model by graph mode of mindspore for quicker sample.
        """
        pred_l, pred_x = self.decoder(time_emb, batch_atom_types, x_t, l_t,
                                      batch_node2graph, batch_edge_index,
                                      node_mask, edge_mask)
        return pred_l, pred_x

    def sample(self,
               batch_atom_types,
               batch_node2graph,
               batch_edge_index,
               node_mask,
               edge_mask,
               batch_mask,
               step_lr=1e-5):
        """Generation process of diffution. Note: For simplicity, we use x instead of frac_coords and
            l instead of lattice.

        Args:
            batch_atom_types (Tensor): Atom types of nodes in a batch of graph. Shape: (num_atoms,)
            batch_node2graph (Tensor): Graph index for each node. Shape: (num_atoms,)
            batch_edge_index (Tensor): Beginning and ending node index for each edge.
                Shape: (2, num_edges)
            node_mask (Tensor): Node mask for padded tensor. Shape: (num_atoms,)
            edge_mask (Tensor): Edge mask for padded tensor. Shape: (num_edges,)
            batch_mask (Tensor): Graph mask for padded tensor. Shape: (batchsize,)
            step_lr (float): Langevin dynamics. Defaults to 1e-5.

        Returns:
            Tuple(Tensor, Tensor): Return the generated lattice matrix and fractional
                coordinates for the input atom types of each crystal.
        """

        batch_size_pad = batch_mask.shape[0]
        num_node_pad = node_mask.shape[0]

        # For simplicity, we use x instead of frac_coords and l instead of lattice
        l_init = ops.StandardNormal()((batch_size_pad, 3, 3))
        x_init = ops.UniformReal()((num_node_pad, 3)) % 1.0

        time_start = self.beta_scheduler.timesteps

        l_t = l_init
        x_t = x_init % 1.0

        for t in range(time_start, 0, -1):
            times = ops.Fill()(ms.float32, (batch_size_pad,), t)

            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / ops.Sqrt()(alphas)
            c1 = (1 - alphas) / ops.Sqrt()(1 - alphas_cumprod)

            # Corrector
            rand_l = ops.StandardNormal()(
                l_init.shape) if t > 1 else ops.ZerosLike()(l_init)
            rand_x = ops.StandardNormal()(
                x_init.shape) if t > 1 else ops.ZerosLike()(x_init)

            step_size = step_lr * (sigma_x /
                                   self.sigma_scheduler.sigma_begin)**2
            std_x = ops.Sqrt()(2 * step_size)

            l_t = ops.mul(l_t, ops.reshape(batch_mask, (-1, 1, 1)))
            x_t = ops.mul(x_t, ops.reshape(node_mask, (-1, 1)))
            pred_l, pred_x = self.denoise(time_emb, batch_atom_types, x_t, l_t,
                                          batch_node2graph, batch_edge_index,
                                          node_mask, edge_mask)

            pred_x = pred_x * ops.Sqrt()(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            l_t_minus_05 = l_t

            # Predictor
            rand_l = ops.StandardNormal()(
                l_init.shape) if t > 1 else ops.ZerosLike()(l_init)
            rand_x = ops.StandardNormal()(
                x_init.shape) if t > 1 else ops.ZerosLike()(x_init)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = ops.Sqrt()(
                (adjacent_sigma_x**2 *
                 (sigma_x**2 - adjacent_sigma_x**2)) / (sigma_x**2))

            l_t_minus_05 = ops.mul(l_t_minus_05,
                                   ops.reshape(batch_mask, (-1, 1, 1)))
            x_t_minus_05 = ops.mul(x_t_minus_05,
                                   ops.reshape(node_mask, (-1, 1)))
            pred_l, pred_x = self.denoise(time_emb, batch_atom_types,
                                          x_t_minus_05, l_t_minus_05,
                                          batch_node2graph, batch_edge_index,
                                          node_mask, edge_mask)

            pred_x = pred_x * ops.Sqrt()(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l

            x_t = x_t_minus_1 % 1.0
            l_t = l_t_minus_1

        return x_t, l_t

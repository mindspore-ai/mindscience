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
"""test mindchemistry DiffCSP"""

import os
import math
import pytest

import mindspore as ms
from mindspore import context, Tensor, ops, nn, load_checkpoint, load_param_into_net
import mindspore.numpy as mnp

from mindchemistry.cell import CSPNet
from mindchemistry.graph.graph import LiftGlobalToNode
from mindchemistry.graph.loss import L2LossMask

LTOL = 0.6
FTOL = 0.6

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

def p_wrapped_normal_ms(x, sigma, n=10, t=1.0):
    """Utils for calcatating the score of wrapped normal distribution.
    """
    p_ = 0
    for i in range(-n, n + 1):
        p_ += ops.Exp()(-(x + t * i) ** 2 / 2 / sigma ** 2)
    return p_

def d_log_p_wrapped_normal_ms(x, sigma, n=10, t=1.0):
    """The score of wrapped normal distribution, which is parameterized by sigma,
       for the input value x. See details in Appendix B.1 in the paper of DiffCSP.

    Args:
        x (Tensor): Input noise.
        sigma (Tensor): The variance of wrapped normal distribution.
        n (int): The approximate parameter of the score of wrapped normal distribution. Defaults to 10.
        t (int): The period of wrapped normal distribution.  Defaults to 1.0.

    Returns:
        Tensor: The score for the input value x.
    """
    p_ = 0
    for i in range(-n, n + 1):
        p_ += (x + t * i) / sigma ** 2 * ops.Exp()(-(x + t * i) ** 2 / 2 / sigma ** 2)
    return p_ / p_wrapped_normal_ms(x, sigma, n, t)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_diffcsp():
    """
    Feature: Test DiffCSP in platform ascend.
    Description: The forward output should has expected shape and accuracy.
    Expectation: Success or throw AssertionError.
    """
    os.environ['MS_JIT_MODULES'] = "mindchemistry"
    context.set_context(mode=context.GRAPH_MODE)

    diffcsp_model = CSPNet(
        hidden_dim=512,
        latent_dim=256,
        num_layers=6,
        max_atoms=100,
        num_freqs=128
        )

    mindspore_ckpt = load_checkpoint("/home/workspace/mindspore_ckpt/ckpt/cspnet.ckpt")
    load_param_into_net(diffcsp_model, mindspore_ckpt)

    time_embedding = SinusoidalTimeEmbeddings(256)

    batch_size = 2
    num_atoms_total = 25
    # predict structures of Na3MnCoNiO6 and Nd(Al2Cu)4
    t = Tensor([919, 481], ms.int32)
    time_emb = time_embedding(t)
    atom_types = Tensor([11, 11, 11, 25, 27, 28, 8, 8, 8, 8, 8, 8, 60, 13, 13, 13, 13, 13,
                         13, 13, 13, 29, 29, 29, 29], ms.int32)
    lattices = Tensor([[[2.97418404e+00, 0.00000000e+00, -5.75143814e-01],
                        [-3.28006029e-01, 5.36585712e+00, -1.69618487e+00],
                        [0.00000000e+00, 0.00000000e+00, 7.97762775e+00]],
                       [[4.80539989e+00, 0.00000000e+00, 1.98468173e+00],
                        [2.40269971e+00, 6.29416704e+00, 9.92341042e-01],
                        [0.00000000e+00, 0.00000000e+00, 6.80986357e+00]]], ms.float32)
    frac_coords = Tensor([[6.6651100e-01, 1.7850000e-03, 3.3302200e-01],
                          [1.2300001e-04, 9.9972498e-01, 2.4600001e-04],
                          [3.3202499e-01, 9.9566197e-01, 6.6404998e-01],
                          [4.9982300e-01, 5.0073302e-01, 9.9964601e-01],
                          [1.7373601e-01, 4.9349201e-01, 3.4747201e-01],
                          [8.3296400e-01, 5.0580001e-01, 6.6592801e-01],
                          [7.2316997e-02, 6.9008100e-01, 1.4463399e-01],
                          [4.1867900e-01, 6.9827998e-01, 8.3735800e-01],
                          [7.4119502e-01, 7.1081799e-01, 4.8238999e-01],
                          [9.2924601e-01, 3.0546999e-01, 8.5849202e-01],
                          [2.4877100e-01, 2.9142499e-01, 4.9754199e-01],
                          [5.8461100e-01, 3.0672801e-01, 1.6922200e-01],
                          [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                          [7.2122198e-01, 7.7877802e-01, 7.7877802e-01],
                          [5.0000000e-01, 2.2122200e-01, 7.7877802e-01],
                          [3.4894001e-01, 6.5105999e-01, 6.5105999e-01],
                          [5.7936511e-17, 3.4894001e-01, 6.5105999e-01],
                          [4.5203392e-17, 6.5105999e-01, 3.4894001e-01],
                          [6.5105999e-01, 3.4894001e-01, 3.4894001e-01],
                          [2.7877799e-01, 2.2122200e-01, 2.2122200e-01],
                          [5.0000000e-01, 7.7877802e-01, 2.2122200e-01],
                          [5.0000000e-01, 5.0000000e-01, 0.0000000e+00],
                          [5.5514202e-17, 5.0000000e-01, 1.0000000e+00],
                          [1.4521202e-17, 1.0000000e+00, 5.0000000e-01],
                          [5.0000000e-01, 1.0000000e+00, 5.0000000e-01]], ms.float32)
    node2graph = Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ms.int32)
    edge_index = Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                          4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
                          5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
                          6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9,
                          9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
                          10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                          12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13,
                          13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14,
                          14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                          15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                          16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18,
                          18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19,
                          19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20,
                          20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
                          21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23,
                          23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24,
                          24, 24, 24, 24, 24, 24, 24, 24, 24],
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
                          4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7,
                          8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
                          4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7,
                          8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
                          4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7,
                          8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14,
                          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 17,
                          18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                          21, 22, 23, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                          24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16,
                          17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 17, 18, 19,
                          20, 21, 22, 23, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                          23, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12,
                          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 12, 13, 14, 15,
                          16, 17, 18, 19, 20, 21, 22, 23, 24]], ms.int32)

    node_mask = Tensor([1]*atom_types.shape[0], ms.int32)
    edge_mask = Tensor([1]*edge_index.shape[1], ms.int32)
    batch_mask = Tensor([1]*batch_size, ms.int32)

    c0 = Tensor([1.25875384e-01, 7.23357856e-01], ms.float32)
    c1 = Tensor([9.92046058e-01, 6.90473258e-01], ms.float32)
    sigmas = Tensor([3.44197601e-01, 4.57015522e-02], ms.float32)
    sigmas_norm = Tensor([7.40092039e-01, 4.88875946e+02], ms.float32)
    lift_node = LiftGlobalToNode()
    sigmas_per_atom = lift_node(sigmas[:, None], node2graph)
    sigmas_norm_per_atom = lift_node(sigmas_norm[:, None], node2graph)

    rand_l, rand_x = ops.StandardNormal()(lattices.shape), ops.StandardNormal()(frac_coords.shape)

    input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
    input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

    lattice_out, coord_out = diffcsp_model(
        time_emb, atom_types, input_frac_coords, input_lattice, node2graph, edge_index, node_mask, edge_mask)

    tar_x = d_log_p_wrapped_normal_ms(sigmas_per_atom * rand_x, sigmas_per_atom) / ops.Sqrt()(sigmas_norm_per_atom)

    loss_func_mse = L2LossMask(reduction='mean')
    mseloss_l = loss_func_mse(lattice_out, rand_l, mask=batch_mask, num=batch_size)
    mseloss_x = loss_func_mse(coord_out, tar_x, mask=node_mask, num=num_atoms_total)

    assert lattice_out.shape == (batch_size, 3, 3), f"For `DiffCSP`, the lattice output should be\
         ({batch_size}, 3, 3), but got {lattice_out.shape}."

    assert coord_out.shape == (num_atoms_total, 3), f"For `DiffCSP`, the coordinates output should be\
         ({num_atoms_total}, 3), but got {coord_out.shape}."

    assert mseloss_l <= LTOL, "The denoising of lattice accuracy is not successful."
    assert mseloss_x <= FTOL, "The denoising of fractional coordinates accuracy is not successful."

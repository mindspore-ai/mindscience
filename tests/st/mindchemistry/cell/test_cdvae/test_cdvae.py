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
"""test mindchemistry CDVAE"""

import os
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, mint

from mindchemistry.cell import CDVAE
from mindchemistry.cell.dimenet.dimenet_wrap import DimeNetWrap
from mindchemistry.cell.gemnet.gemnet_wrap import GemNetWrap
from mindchemistry.cell.gemnet.data_utils import StandardScalerMindspore

ms.set_seed(1234)
np.random.seed(1234)
os.environ["MS_JIT_MODULES"] = "mindchemistry"
context.set_context(mode=context.PYNATIVE_MODE)

config_path = "./configs.yaml"
data_config_path = "./perov_5.yaml"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_cdvae():
    """
    Feature: Test CDVAE in platform ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """

    cdvae_model = CDVAE(config_path, data_config_path)

    # input data
    batch_size = 2
    atom_types = Tensor([6, 7, 6, 8], ms.int32)
    dist = Tensor([1.4, 1.7, 1.8, 1.9, 2.0, 2.1, 1.8, 1.6], ms.float32)
    idx_kj = Tensor([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 7, 6, 6, 7], ms.int32)
    idx_ji = Tensor([1, 0, 3, 2, 5, 4, 4, 5, 2, 3, 0, 1, 6, 7, 7, 6], ms.int32)
    edge_j = Tensor([0, 1, 1, 0, 2, 3, 3, 2], ms.int32)
    edge_i = Tensor([1, 0, 0, 1, 3, 2, 2, 3], ms.int32)
    batch = Tensor([0, 0, 1, 1], ms.int32)
    lengths = Tensor([[2.5, 2.5, 2.5],
                      [2.5, 2.5, 2.5]], ms.float32)
    angles = Tensor([[90.0, 90.0, 90.0],
                     [90.0, 90.0, 90.0]], ms.float32)
    frac_coords = Tensor([[0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.5],
                          [0.7, 0.7, 0.7],
                          [0.5, 0.5, 0.5]], ms.float32)
    num_atoms = Tensor([2, 2], ms.int32)
    y = Tensor([0.08428, 0.01353], ms.float32)
    total_atoms = 4
    np.random.seed(1234)
    sbf = Tensor(np.random.randn(16, 42), ms.float32)
    cdvae_model.lattice_scaler = StandardScalerMindspore(
        Tensor([2.5, 2.5, 2.5, 90.0, 90.0, 90.0], ms.float32),
        Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ms.float32))
    cdvae_model.scaler = StandardScalerMindspore(
        Tensor([2.62], ms.float32),
        Tensor([1.0], ms.float32))

    out = cdvae_model(atom_types, dist,
                      idx_kj, idx_ji, edge_j, edge_i,
                      batch, lengths, num_atoms,
                      angles, frac_coords, y, batch_size,
                      sbf, total_atoms, False, True)

    assert mint.isclose(out, ms.Tensor(29.453514), rtol=1e-4, atol=1e-4), f"For `CDVAE`, the output should be\
         29.4535, but got {out}."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dimenet():
    """
    Feature: Test Dimenet in CDVAE in platform ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    dimenet = DimeNetWrap(config_path, data_config_path)

    # input data
    atom_types = np.array([6, 7, 6, 8], np.int32)
    lengths = np.array([[2.5, 2.5, 2.5],
                        [2.5, 2.5, 2.5]], np.float32)
    angles = np.array([[90.0, 90.0, 90.0],
                       [90.0, 90.0, 90.0]], np.float32)
    frac_coords = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [0.7, 0.7, 0.7],
                            [0.5, 0.5, 0.5]], np.float32)
    edge_index = np.array([[0, 1, 1, 0, 2, 3, 3, 2],
                           [1, 0, 0, 1, 3, 2, 2, 3]], np.int32)
    to_jimages = np.zeros((edge_index.shape[1], 3), np.int32)
    num_bonds = np.array([4, 4], np.int32)
    num_atoms = np.array([2, 2], np.int32)

    out = dimenet.evaluation(angles, lengths, num_atoms, edge_index, frac_coords,
                             num_bonds, to_jimages, atom_types)
    assert out.shape == (2, dimenet.latent_dim), f"For `DimeNetPlusPlus`, the output shape should be\
          (2, {dimenet.latent_dim}), but got {out.shape}."
    assert mint.isclose(out.sum(), ms.Tensor(0.0, dtype=ms.float32)), f"For `CDVAE`, the summary output\
         should be smaller than 29.45, but got {out.sum()}."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_gemnet():
    """
    Feature: Test Gemnet in CDVAE in platform ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    gemnet = GemNetWrap(config_path)

    # input data
    batch_size = 2
    atom_types = Tensor([6, 7, 6, 8], ms.int32)
    batch = Tensor([0, 0, 1, 1], ms.int32)
    lengths = np.array([[2.5, 2.5, 2.5],
                        [2.5, 2.5, 2.5]], np.float32)
    angles = np.array([[90.0, 90.0, 90.0],
                       [90.0, 90.0, 90.0]], np.float32)
    frac_coords = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5],
                            [0.7, 0.7, 0.7],
                            [0.5, 0.5, 0.5]], np.float32)
    num_atoms = np.array([2, 2], np.int32)
    total_atoms = 4
    h, f_t = gemnet.evaluation(
        frac_coords, num_atoms, lengths, angles, atom_types, batch,
        total_atoms, batch_size)

    assert h.shape == (4, 128), f"For `CDVAE Decoder`, the shape of h\
         should be smaller than (4, 100), but got {h.shape}."
    assert f_t.shape == (4, 3), f"For `CDVAE Decoder`, the shape of\
         f_t should be (4, 3) , but got {f_t.shape}."

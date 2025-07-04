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

from mindchemistry.cell.dimenet.dimenet_wrap import DimeNetWrap
from mindchemistry.cell.gemnet.gemnet_wrap import GemNetWrap

ms.set_seed(1234)
np.random.seed(1234)
os.environ["MS_JIT_MODULES"] = "mindchemistry"
context.set_context(mode=context.PYNATIVE_MODE)

config_path = "./configs.yaml"
data_config_path = "./perov_5.yaml"

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

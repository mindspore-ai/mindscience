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
"""test mindchemistry Allegro"""

import os
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor

from mindchemistry.cell.geonet import Allegro

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_allegro():
    """
    Feature: Test Allegro in platform ascend.
    Description: The forward output should has expected shape.
    Expectation: Success or throw AssertionError.
    """
    os.environ['MS_JIT_MODULES'] = "mindchemistry"
    context.set_context(mode=context.GRAPH_MODE)
    allegro_model = Allegro(
        l_max=3,
        irreps_in={'pos': '1x1o', 'edge_index': None, 'node_attrs': '4x0e', 'node_features': '4x0e',
                   'edge_embedding': '8x0e'},
        avg_num_neighbor=11.0,
        num_layers=3,
        env_embed_multi=128,
        two_body_kwargs={'hidden_dims': [128, 256, 512, 1024], 'activation': 'silu', 'weight_init': 'uniform'},
        latent_kwargs={'hidden_dims': [1024, 1024, 1024], 'activation': 'silu', 'weight_init': 'uniform'},
        env_embed_kwargs={'hidden_dims': [], 'activation': None, 'weight_init': 'uniform'}
        )

    edges = 660
    final_latent_out = 1024
    embedding_out = (
        Tensor(np.random.rand(60, 4), ms.float32),
        Tensor(np.random.rand(60, 4), ms.float32),
        Tensor(np.random.rand(660, 3), ms.float32),
        Tensor(np.random.rand(660), ms.float32),
        Tensor(np.random.rand(660, 8), ms.float32),
        Tensor(np.random.rand(660), ms.float32),
        Tensor(np.ones(660), ms.bool_)
        )
    edge_index = Tensor(np.ones((2, 660)), ms.int32)
    atom_types = Tensor(np.ones((60, 1)), ms.int32)

    out = allegro_model(embedding_out, edge_index, atom_types)
    assert out.shape == (edges, final_latent_out), f"For `Allegro`, the output should be\
         ({edges}, {final_latent_out}), but got {out.shape}."


# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Data processing"""
import os

import yaml
import numpy as np

from sciai.utils import to_tensor, parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def flatten(vectors):
    """concat and flatten vectors into 1D"""
    return np.concatenate([v.flatten() for v in vectors])


def generate_test_data(dom_coords, nn=100):
    """generate test data"""
    x1 = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x2 = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    x1, x2 = np.meshgrid(x1, x2)
    x_star = np.hstack((x1.flatten()[:, None], x2.flatten()[:, None]))
    return x1, x2, x_star


def get_model_inputs(bcs_samplers, res_sampler, model, batch_size, adaptive_constant):
    """get model inputs"""
    inputs = sample_model_inputs(bcs_samplers, res_sampler, model, batch_size)
    inputs.append(adaptive_constant)
    return to_tensor(tuple(inputs))


def sample_model_inputs(bcs_samplers, res_sampler, model, batch_size):
    """sample model inputs"""
    # Fetch boundary mini-batches
    x_bc1_batch, u_bc1_batch = bcs_samplers[0].fetch_minibatch(batch_size, model.mu_x, model.sigma_x)
    x_bc2_batch, u_bc2_batch = bcs_samplers[1].fetch_minibatch(batch_size, model.mu_x, model.sigma_x)
    x_bc3_batch, u_bc3_batch = bcs_samplers[2].fetch_minibatch(batch_size, model.mu_x, model.sigma_x)
    x_bc4_batch, u_bc4_batch = bcs_samplers[3].fetch_minibatch(batch_size, model.mu_x, model.sigma_x)

    # Fetch residual mini-batch
    x_res_batch, f_res_batch = res_sampler.fetch_minibatch(batch_size, model.mu_x, model.sigma_x)

    inputs = [
        x_bc1_batch[:, 0:1], x_bc1_batch[:, 1:2], u_bc1_batch,
        x_bc2_batch[:, 0:1], x_bc2_batch[:, 1:2], u_bc2_batch,
        x_bc3_batch[:, 0:1], x_bc3_batch[:, 1:2], u_bc3_batch,
        x_bc4_batch[:, 0:1], x_bc4_batch[:, 1:2], u_bc4_batch,
        x_res_batch[:, 0:1], x_res_batch[:, 1:2], f_res_batch
    ]

    return inputs

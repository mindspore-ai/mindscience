# Copyright 2025 Huawei Technologies Co., Ltd
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
"""evaluate_utils"""
import logging
import mindspore as ms
import mindspore.mint as mint
from mindspore.nn import Adam
from tqdm import tqdm

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def get_real_samples(loader):
    """
    reconstruct the crystals in <loader>.
    """
    gt_frac_coords = []
    groundtruth_num_atoms = []
    groundtruth_atom_types = []
    gt_lengths = []
    gt_angles = []
    for idx, data in enumerate(loader):
        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        (atom_types, dist, _, idx_kj, idx_ji,
         edge_j, edge_i, batch, lengths, num_atoms,
         angles, frac_coords, _, batch_size, sbf,
         total_atoms) = data
        gt_frac_coords.append(frac_coords.asnumpy())
        gt_angles.append(angles.asnumpy())
        gt_lengths.append(lengths.asnumpy())
        groundtruth_atom_types.append(atom_types.asnumpy())
        groundtruth_num_atoms.append(num_atoms.asnumpy())

    return (
        gt_frac_coords, groundtruth_num_atoms, groundtruth_atom_types,
        gt_lengths, gt_angles)


def get_generation_res(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    """
    generate new crystals based on randomly sampled z.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    result_frac_coords = []
    result_num_atoms = []
    result_atom_types = []
    result_lengths = []
    result_angles = []

    for _ in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = ms.ops.randn(batch_size, model.hidden_dim)

        for _ in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs, batch_size)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples["frac_coords"].asnumpy())
            batch_num_atoms.append(samples["num_atoms"].asnumpy())
            batch_atom_types.append(samples["atom_types"].asnumpy())
            batch_lengths.append(samples["lengths"].asnumpy())
            batch_angles.append(samples["angles"].asnumpy())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples["all_frac_coords"][::down_sample_traj_step].asnumpy())
                batch_all_atom_types.append(
                    samples["all_atom_types"][::down_sample_traj_step].asnumpy())

        # collect sampled crystals for this z.
        result_frac_coords.append(batch_frac_coords)
        result_num_atoms.append(batch_num_atoms)
        result_atom_types.append(batch_atom_types)
        result_lengths.append(batch_lengths)
        result_angles.append(batch_angles)
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                batch_all_frac_coords)
            all_atom_types_stack.append(
                batch_all_atom_types)

    return (result_frac_coords, result_num_atoms, result_atom_types,
            result_lengths, result_angles,
            all_frac_coords_stack, all_atom_types_stack)

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


def get_reconstructon_res(loader, model, ld_kwargs, num_evals,
                          force_num_atoms=False, force_atom_types=False):
    """
    reconstruct the crystals in <loader>.
    """
    result_frac_coords = []
    result_num_atoms = []
    result_atom_types = []
    result_lengths = []
    result_angles = []
    gt_frac_coords = []
    groundtruth_num_atoms = []
    groundtruth_atom_types = []
    gt_lengths = []
    gt_angles = []
    for idx, data in enumerate(loader):
        logging.info("Reconstructing %d", int(idx * data[-3]))
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

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
        _, _, z = model.encode(atom_types, dist,
                               idx_kj, idx_ji, edge_j, edge_i,
                               batch, total_atoms, batch_size, sbf)
        for _ in range(num_evals):
            gt_num_atoms = num_atoms if force_num_atoms else None
            gt_atom_types = atom_types if force_atom_types else None
            outputs = model.langevin_dynamics(
                z, ld_kwargs, batch_size, total_atoms, gt_num_atoms, gt_atom_types)
            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs["frac_coords"].asnumpy())
            batch_num_atoms.append(outputs["num_atoms"].asnumpy())
            batch_atom_types.append(outputs["atom_types"].asnumpy())
            batch_lengths.append(outputs["lengths"].asnumpy())
            batch_angles.append(outputs["angles"].asnumpy())
        # collect sampled crystals for this z.
        result_frac_coords.append(batch_frac_coords)
        result_num_atoms.append(batch_num_atoms)
        result_atom_types.append(batch_atom_types)
        result_lengths.append(batch_lengths)
        result_angles.append(batch_angles)

    return (
        result_frac_coords, result_num_atoms, result_atom_types,
        result_lengths, result_angles,
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


def get_optimization_res(model, ld_kwargs, data_loader,
                         num_starting_points=128, num_gradient_steps=5000,
                         lr=1e-3, num_saved_crys=10):
    """
    optimize the structure based on specific proprety.
    """
    model.set_train(True)
    if data_loader is not None:
        data = next(iter(data_loader))
        (atom_types, dist, _, idx_kj, idx_ji,
         edge_j, edge_i, batch, _, num_atoms,
         _, _, _, batch_size, sbf,
         total_atoms) = data
        _, _, z = model.encode(atom_types, dist,
                               idx_kj, idx_ji, edge_j, edge_i,
                               batch, total_atoms, batch_size, sbf)
        z = mint.narrow(z, 0, 0, num_starting_points)
        z = ms.Parameter(z, requires_grad=True)
    else:
        z = mint.randn(num_starting_points, model.hparams.hidden_dim)
        z = ms.Parameter(z, requires_grad=True)

    opt = Adam([z], learning_rate=lr)
    freeze_model(model)

    loss_fn = model.fc_property

    def forward_fn(data):
        loss = loss_fn(data)
        return loss
    grad_fn = ms.value_and_grad(forward_fn, None, opt.parameters)

    def train_step(data):
        loss, grads = grad_fn(data)
        opt(grads)
        return loss

    all_crystals = []
    total_atoms = mint.sum(mint.narrow(
        num_atoms, 0, 0, num_starting_points)).item()
    interval = num_gradient_steps // (num_saved_crys - 1)
    for i in tqdm(range(num_gradient_steps)):
        loss = mint.mean(train_step(z))
        logging.info("Task opt step: %d, loss: %f", i, loss)
        if i % interval == 0 or i == (num_gradient_steps - 1):
            crystals = model.langevin_dynamics(
                z, ld_kwargs, batch_size, total_atoms)
            all_crystals.append(crystals)
    return {k: mint.cat([d[k] for d in all_crystals]).unsqueeze(0).asnumpy() for k in
            ["frac_coords", "atom_types", "num_atoms", "lengths", "angles"]}


def freeze_model(model):
    """ The model is fixed, only optimize z"""
    for param in model.get_parameters():
        param.requires_grad = False

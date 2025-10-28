# Copyright 2022 Huawei Technologies Co., Ltd
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

"""
Dataset utilities for NequIP-based molecular dynamics.
Provides data loading, preprocessing, and batching logic for the RMD17 dataset.
"""

import numpy as np
import mindspore as ms
from mindscience.e3nn import radius_graph_full


class RMD17:
    """
    Represents the RMD17 dataset for molecular property prediction.

    Args:
        rmd_data (dict): Loaded `.npz` dataset dictionary.
        start (int): Starting index for slicing dataset.
        end (int): Ending index for slicing dataset.
        get_force (bool): Whether to include forces in the labels.
        dtype (ms.dtype): MindSpore tensor dtype.

    Attributes:
        charges (ndarray): Atomic number indices for each atom.
        coords (ndarray): Atomic coordinates.
        energies (ndarray): Potential energies.
        forces (ndarray): Atomic forces.
        label (ndarray): Normalized energy or energy+force labels.
    """

    def __init__(self, rmd_data, start=None, end=None, get_force=False, dtype=ms.float32):
        dtype_map = {
            ms.float16: np.float16,
            ms.float32: np.float32,
            ms.float64: np.float64
        }
        np_dtype = dtype_map[dtype]

        self.charges = rmd_data['nuclear_charges'].astype(np.int32)
        self.coords = rmd_data['coords'][start:end].astype(np_dtype)
        self.energies = rmd_data['energies'][start:end].astype(np_dtype)
        self.forces = rmd_data['forces'][start:end].astype(np_dtype)
        self.forces_all = rmd_data['forces'].astype(np_dtype)

        allowed_species = np.unique(self.charges)
        self.num_type = allowed_species.shape[0]
        self.charges = self.data_index(allowed_species, self.charges)

        stats = self.statistics(stride=1, end=end)
        energies_mean, energies_std = stats[0]
        self.energies_mean = energies_mean
        self.energies_std = energies_std
        self.force_rms = stats[1][0]

        scale_by = self.force_rms
        shift_by = self.energies_mean

        label = (self.energies - shift_by) / scale_by
        forces_norm = self.forces.reshape((self.forces.shape[0], -1)) / scale_by
        label = label.reshape((label.shape[0], 1))

        if get_force:
            label = np.concatenate((label, forces_norm), axis=-1)

        self.label = label

    def statistics(self, stride: int = 1, end=None):
        """
        Compute dataset-level mean and standard deviation of energy and force.

        Args:
            stride (int): Step for subsampling frames.
            end (int): Optional end index for slicing.

        Returns:
            list[tuple]: [(energy_mean, energy_std), (force_rms,)]
        """
        if end is not None:
            indices = np.arange(end)
            selector = ms.Tensor(indices)[::stride]
        else:
            selector = ms.ops.arange(0, end, stride)

        _ = selector  # placeholder to avoid unused-variable warning

        out = []
        mean = np.mean(self.energies, dtype=np.float64)
        std = np.std(self.energies, dtype=np.float64)
        out.append((mean, std))

        arr = self.forces.reshape(-1, 3)
        out.append((np.sqrt(np.mean(arr * arr)),))
        return out

    def data_index(self, allowed_species_np, atomic_nums):
        """
        Map atomic numbers to type indices based on unique allowed species.

        Args:
            allowed_species_np (ndarray): Unique atomic numbers.
            atomic_nums (ndarray): Atomic numbers for current molecule.

        Returns:
            ndarray: Type indices.
        """
        num_species = allowed_species_np.shape[0]
        min_z = np.amin(allowed_species_np).astype(np.int64)
        max_z = np.amax(allowed_species_np).astype(np.int64)
        z_to_index = np.full((1 + max_z - min_z,), -1, dtype=np.int32)
        z_to_index[allowed_species_np - min_z] = np.arange(num_species)
        return z_to_index[atomic_nums - min_z]

    def __getitem__(self, index):
        """Return atomic charges, coordinates, and label for given index."""
        return self.charges, self.coords[index], self.label[index]

    def __len__(self):
        """Return dataset length."""
        return len(self.label)


def generate_dataset(raw_data, batch_size=1, embed=False):
    """
    Build a MindSpore dataset and construct graph edges.

    Args:
        raw_data (Dataset): Dataset object (e.g., RMD17 instance).
        batch_size (int): Number of samples per batch.
        embed (bool): Whether to use one-hot embedding.

    Returns:
        tuple: (dataset, edge_index, batch)
    """
    dataset = ms.dataset.GeneratorDataset(raw_data, column_names=['x', 'pos', 'label'], shuffle=False)
    dataset = dataset.batch(batch_size=batch_size)

    def one_hot(arr):
        """Convert integer array to one-hot encoding."""
        x = np.zeros((arr.size, arr.max() + 1), dtype=np.float32)
        x[np.arange(arr.size), arr] = 1
        return x

    def reshape_fn(x, pos, label):
        """Reshape dataset tensors for NequIP model."""
        if embed:
            node_feature = one_hot(x.flatten())
            return node_feature, pos.reshape((-1, pos.shape[-1])), label.reshape((-1, label.shape[-1]))

        if label.shape[-1] <= 1:
            return x.flatten(), pos.reshape((-1, pos.shape[-1])), label, np.array(0., dtype=label.dtype)

        energy = label[:, :1]
        force = label[:, 1:].reshape(-1, 3)
        return x.flatten(), pos.reshape((-1, pos.shape[-1])), energy, force

    _, pos_sample, _ = next(dataset.create_tuple_iterator())
    edge_index, batch = radius_graph_full(pos_sample)

    dataset = dataset.map(
        operations=reshape_fn,
        input_columns=['x', 'pos', 'label'],
        output_columns=['x', 'pos', 'energy', 'force']
    )
    return dataset, ms.Tensor(edge_index), ms.Tensor(batch)


def _unpack(data):
    """Unpack dataset dictionary into model input/output tuples."""
    return (data['x'], data['pos']), (data['energy'], data['force'])


def get_num_type(rmd_data):
    """
    Count unique atomic species in the dataset.

    Args:
        rmd_data (dict): Dataset loaded from .npz file.

    Returns:
        int: Number of unique atom types.
    """
    charges = rmd_data['nuclear_charges'].astype(np.int32)
    num_type = np.unique(charges).shape[0]
    return num_type


def create_training_dataset(config, dtype, pred_force):
    """
    Create MindSpore datasets for training and evaluation.

    Args:
        config (dict): Dataset configuration dictionary.
        dtype (ms.dtype): Data type for tensors.
        pred_force (bool): Whether to predict atomic forces.

    Returns:
        tuple: (trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch, num_type)
    """
    with np.load(config['path']) as rmd_data:
        num_type = get_num_type(rmd_data)

        trainset, train_edge_index, train_batch = generate_dataset(
            RMD17(rmd_data, end=config['n_train'], get_force=pred_force, dtype=dtype),
            embed=False,
            batch_size=config['batch_size']
        )

        evalset, eval_edge_index, eval_batch = generate_dataset(
            RMD17(
                rmd_data,
                start=config['n_train'],
                end=config['n_train'] + config['n_val'],
                get_force=pred_force,
                dtype=dtype
            ),
            embed=False,
            batch_size=config['batch_size']
        )

    return trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch, num_type

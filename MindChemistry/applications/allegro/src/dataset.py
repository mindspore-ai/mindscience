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
"""
dataset
"""
import numpy as np
import mindspore as ms
from mindspore.communication import get_rank, get_group_size

from mindchemistry.e3 import radius_graph_full


class RMD17:
    """RMD17"""

    def __init__(
            self, rmd_data, start=None, end=None, get_force=False, dtype=ms.float32, split_random=False, normalize=False
    ):
        inner_dtype = {ms.float16: np.float16, ms.float32: np.float32, ms.float64: np.float64}.get(dtype, None)
        self.charges = rmd_data['nuclear_charges'].astype(np.int32)

        coords = rmd_data['coords']
        energies = rmd_data['energies']
        forces = rmd_data['forces']
        if split_random:
            np.random.shuffle(coords)
            np.random.shuffle(energies)
            np.random.shuffle(forces)

        self._coords = coords[start:end].astype(inner_dtype)
        self._energies = energies[start:end].astype(inner_dtype)
        self._forces = forces[start:end].astype(inner_dtype)
        self._forces_all = forces.astype(inner_dtype)

        allowed_species = np.unique(self.charges)
        self._num_type = allowed_species.shape[0]
        self.charges = RMD17.data_index(allowed_species, self.charges)

        dataset_statistics_stride = 1
        stats = RMD17.statistics(self, stride=dataset_statistics_stride, end=end)
        (energies_mean, energies_std) = stats[:1][0]
        self.energies_mean = energies_mean
        self.energies_std = energies_std
        self.force_rms = stats[1][0]

        if normalize:
            scale_by = self.force_rms
            shift_by = self.energies_mean
            self._forces = self._forces.reshape((self._forces.shape[0], -1))
            if scale_by != 0:
                self._label = (self._energies - shift_by) / scale_by
                self._forces = self._forces / scale_by
            else:
                raise ValueError
            self._label = self._label.reshape((self._label.shape[0], 1))
        else:
            self._forces = self._forces.reshape((self._forces.shape[0], -1))
            self._label = self._energies.reshape((self._energies.shape[0], 1))

        if get_force:
            self._label = np.concatenate((self._label, self._forces), axis=-1)

    def __getitem__(self, index):
        return self.charges, self._coords[index], self._label[index]

    def __len__(self):
        return len(self._label)

    # pylint: disable=E0213
    def data_index(allowed_species_np, atomic_nums):
        """data_index"""
        num_species = allowed_species_np.shape[0]
        min_z = np.amin(allowed_species_np)
        min_z = min_z.astype(np.int64)
        max_z = np.amax(allowed_species_np)
        max_z = max_z.astype(np.int64)
        allowed_species = allowed_species_np
        z_to_index = np.full((1 + max_z - min_z,), -1, dtype=np.int32)
        z_to_index[allowed_species - min_z] = np.arange(num_species)
        out = z_to_index[atomic_nums - min_z]
        return out

    def statistics(self, stride: int = 1, end=None):
        """statistics"""
        if end is not None:
            indices = np.arange(end)
            # pylint: disable=W0612
            selector = ms.Tensor(indices)[::stride]
        else:
            # pylint: disable=W0612
            selector = ms.ops.arange(0, end, stride)

        atom_number = self.charges.shape[0]
        data_size = self._forces_all.shape[0]
        # pylint: disable=W0612
        batch = np.repeat(np.arange(data_size), atom_number).tolist()

        out = []
        arr = self._energies
        mean = np.mean(arr, dtype=np.float64)
        std = np.std(arr, dtype=np.float64)
        out.append((mean, std))

        arr = self._forces.reshape(-1, 3)
        out.append((np.sqrt(np.mean(arr * arr)),))

        return out


def generate_dataset(raw_data, batch_size=1, embed=False, shuffle=False, parallel_mode="NONE"):
    """generate_dataset"""
    literal_pos = 'pos'
    literal_label = 'label'
    if parallel_mode == "DATA_PARALLEL":
        rank_id = get_rank()
        rank_size = get_group_size()
        dataset = ms.dataset.GeneratorDataset(
            raw_data, column_names=['x', literal_pos, literal_label], shuffle=shuffle, num_shards=rank_size,
            shard_id=rank_id
        )
    else:
        dataset = ms.dataset.GeneratorDataset(raw_data, column_names=['x', literal_pos, literal_label], shuffle=shuffle)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    def _one_hot(arr):
        x = np.zeros((arr.size, arr.max() + 1), dtype=np.float32)
        x[np.arange(arr.size), arr] = 1
        return x

    def _reshape(x, pos, label):
        if embed:
            node_feature = _one_hot(x.flatten())
            return node_feature, pos.reshape((-1, pos.shape[-1])), label.reshape((-1, label.shape[-1]))
        if label.shape[-1] <= 1:
            return HandleData(x.flatten(), pos.reshape((-1, pos.shape[-1])), label,
                              np.array(0., dtype=label.dtype)).get_tuple_data()
        energy = label[:, :1]
        force = label[:, 1:].reshape(-1, 3)
        return HandleData(x.flatten(), pos.reshape((-1, pos.shape[-1])), energy, force).get_tuple_data()

    _, pos, _ = next(dataset.create_tuple_iterator())
    edge_index, batch = radius_graph_full(pos)
    dataset = dataset.map(
        operations=_reshape, input_columns=['x', literal_pos, literal_label],
        output_columns=['x', literal_pos, 'energy', 'force']
    )
    return dataset, ms.Tensor(edge_index), ms.Tensor(batch)


def _unpack(data):
    return (data['x'], data[literal_pos]), (data['energy'], data['force'])


def get_num_type(rmd_data):
    charges = rmd_data['nuclear_charges'].astype(np.int32)
    num_type = np.unique(charges).shape[0]
    return num_type


def create_training_dataset(config, dtype, pred_force, parallel_mode="NONE"):
    """create_training_dataset"""
    literal_n_train = 'n_train'
    with np.load(config['path']) as rmd_data:
        shift = 0
        num_type = get_num_type(rmd_data)
        trainset, train_edge_index, train_batch = generate_dataset(
            RMD17(
                rmd_data,
                start=shift,
                end=config[literal_n_train] + shift,
                get_force=pred_force,
                dtype=dtype,
                split_random=config['split_random']
            ),
            embed=False,
            batch_size=config['batch_size'],
            shuffle=config['shuffle'],
            parallel_mode=parallel_mode
        )
        evalset, eval_edge_index, eval_batch = generate_dataset(
            RMD17(
                rmd_data,
                start=config[literal_n_train] + shift,
                end=config[literal_n_train] + config['n_val'] + shift,
                get_force=pred_force,
                dtype=dtype,
                split_random=config['split_random']
            ),
            embed=False,
            batch_size=config['batch_size_eval']
        )
    return HandleDataSet(trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch,
                         num_type).get_tuple_data()


class HandleData:
    """CommonData"""

    def __init__(self, x, pos, energy, force):
        self.tuple_data = (x, pos, energy, force)

    def get_tuple_data(self):
        """get_tuple_data"""
        return self.tuple_data


class HandleDataSet:
    """CommonData"""

    def __init__(self, trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch, num_type):
        self.tuple_data = (trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch, num_type)

    def get_tuple_data(self):
        """get_tuple_data"""
        return self.tuple_data

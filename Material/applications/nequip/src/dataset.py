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
# """
# dataset
# """
import numpy as np
import mindspore as ms
from mindchemistry.e3 import radius_graph_full


class RMD17:
    def __init__(self, rmd_data, start=None, end=None, get_force=False, dtype=ms.float32):
        _dtype = {
            ms.float16: np.float16,
            ms.float32: np.float32,
            ms.float64: np.float64
        }[dtype]
        self.charges = rmd_data['nuclear_charges'].astype(np.int32)
        self._coords = rmd_data['coords'][start:end].astype(_dtype)
        self._energies = rmd_data['energies'][start:end].astype(_dtype)
        self._forces = rmd_data['forces'][start:end].astype(_dtype)
        self._forces_all = rmd_data['forces'].astype(_dtype)

        allowed_species = np.unique(self.charges)
        self._num_type = allowed_species.shape[0]
        self.charges = RMD17.data_Index(allowed_species, self.charges)

        dataset_statistics_stride = 1
        stats = RMD17.statistics(self, stride=dataset_statistics_stride, end=end)
        (energies_mean, energies_std) = stats[:1][0]
        self.energies_mean = energies_mean
        self.energies_std = energies_std
        self.force_rms = stats[1][0]

        scale_by = self.force_rms
        shift_by = self.energies_mean

        self._label = (self._energies - shift_by) / scale_by

        self._forces = self._forces.reshape((self._forces.shape[0], -1))
        self._forces = self._forces / scale_by
        self._label = self._label.reshape((self._label.shape[0], 1))
        if get_force:
            self._label = np.concatenate((self._label, self._forces), axis=-1)

    def statistics(self, stride: int = 1, end=None):
        if end is not None:
            _indices = np.arange(end)
            selector = ms.Tensor(_indices)[::stride]
        else:
            selector = ms.ops.arange(0, end, stride)

        atom_number = self.charges.shape[0]
        data_size = self._forces_all.shape[0]
        batch = np.repeat(np.arange(data_size), atom_number).tolist()
        node_selector = ms.Tensor(np.in1d(batch, selector.numpy()))

        out = []
        arr = self._energies
        mean = np.mean(arr, dtype=np.float64)
        std = np.std(arr, dtype=np.float64)
        out.append((mean, std))

        arr = self._forces.reshape(-1, 3)
        out.append((np.sqrt(np.mean(arr * arr)),))

        return out

    def data_Index(allowed_species_np, atomic_nums):
        num_species = allowed_species_np.shape[0]
        _min_Z = np.amin(allowed_species_np)
        _min_Z = _min_Z.astype(np.int64)
        _max_Z = np.amax(allowed_species_np)
        _max_Z = _max_Z.astype(np.int64)
        allowed_species = allowed_species_np
        Z_to_index = np.full((1 + _max_Z - _min_Z,), -1, dtype=np.int32)
        Z_to_index[allowed_species - _min_Z] = np.arange(num_species)
        out = Z_to_index[atomic_nums - _min_Z]
        return out

    def __getitem__(self, index):
        return self.charges, self._coords[index], self._label[index]

    def __len__(self):
        return len(self._label)


def generate_dataset(raw_data, batch_size=1, embed=False):
    dataset = ms.dataset.GeneratorDataset(raw_data, column_names=['x', 'pos', 'label'], shuffle=False)
    dataset = dataset.batch(batch_size=batch_size)

    def _one_hot(arr):
        x = np.zeros((arr.size, arr.max() + 1), dtype=np.float32)
        x[np.arange(arr.size), arr] = 1
        return x

    def _reshape(x, pos, label):
        if embed:
            node_feature = _one_hot(x.flatten())
            return node_feature, pos.reshape((-1, pos.shape[-1])), label.reshape((-1, label.shape[-1]))
        else:
            if label.shape[-1] <= 1:
                return x.flatten(), pos.reshape((-1, pos.shape[-1])), label, np.array(0., dtype=label.dtype)
            else:
                _energy = label[:, :1]
                _force = label[:, 1:].reshape(-1, 3)
                return x.flatten(), pos.reshape((-1, pos.shape[-1])), _energy, _force

    _x, _pos, _label = next(dataset.create_tuple_iterator())
    edge_index, batch = radius_graph_full(_pos)
    dataset = dataset.map(operations=_reshape, input_columns=['x', 'pos', 'label'],
                          output_columns=['x', 'pos', 'energy', 'force'])
    return dataset, ms.Tensor(edge_index), ms.Tensor(batch)


def _unpack(data):
    return (data['x'], data['pos']), (data['energy'], data['force'])


def get_num_type(rmd_data):
    charges = rmd_data['nuclear_charges'].astype(np.int32)
    num_type = np.unique(charges).shape[0]
    return num_type


def create_training_dataset(config, dtype, pred_force):
    with np.load(config['path']) as rmd_data:
        num_type = get_num_type(rmd_data)
        trainset, train_edge_index, train_batch = generate_dataset(
            RMD17(rmd_data, end=config['n_train'], get_force=pred_force, dtype=dtype), embed=False,
            batch_size=config['batch_size'])
        evalset, eval_edge_index, eval_batch = generate_dataset(
            RMD17(rmd_data, start=config['n_train'], end=config['n_train'] + config['n_val'], get_force=pred_force,
                  dtype=dtype), embed=False, batch_size=config['batch_size'])
    return trainset, train_edge_index, train_batch, evalset, eval_edge_index, eval_batch, num_type

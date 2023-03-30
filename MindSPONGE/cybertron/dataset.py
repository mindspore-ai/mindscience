# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
Dataset Pre-processor
"""

from operator import itemgetter
import numpy as np
from numpy import ndarray

from mindsponge.function import Units

class DatasetProcessor:
    r"""A dataset pre-processor

        Args:

            name (str):             Name of dataset

            atom_types (ndarray):   Numpy array of shape (N, A) or (A). Data type is int
                                    Atom types (atomic number).

            position (ndarray):     Numpy array of shape (N, A, D). Data type is float
                                    Position coordinates

            label (ndarray):        Numpy array of shape (N, E). Data type is float
                                    Dataset label

            force (ndarray):        Numpy array of shape (N, A, D). Data type is float
                                    Force of each atom. Default: None

            type_ref (ndarray):     Numpy array of shape (A, E). Data type is float
                                    Reference value for each atom type. Default: None

            length_unit (str):      Unit of position coordinates. Default: 'nm'

            energy_unit (str):      Unit of energy. Default: None

        Symbols:

            N:  Number of dataset.

            A:  Maximum number of atoms in molecule.

            E:  Number of label.

            D:  Dimension of position coordinate, usually is 3.

        """

    def __init__(self,
                 name: str,
                 atom_types: ndarray,     # (N,A) or (A)
                 position: ndarray,       # (N,A,D)
                 label: ndarray,          # (N,X)
                 force: ndarray = None,     # (N,A,D)
                 type_ref: ndarray = None,
                 length_unit: str = 'nm',
                 energy_unit: str = None,
                 ):

        self.name = name
        self.atom_types = None
        self.single_molecule = False

        self.units = Units(length_unit, energy_unit)
        self.length_unit = self.units.length_unit
        self.energy_unit = self.units.energy_unit

        if self.length_unit == 'none':
            raise ValueError('The length unit cannot be None!')

        # (N,A)
        self.atom_types = np.array(atom_types, np.int32)
        if self.atom_types.ndim == 1:
            self.atom_types = np.expand_dims(self.atom_types, 0)
        if self.atom_types.ndim != 2:
            raise ValueError(
                'The shape of atom_types must be (N_data,N_atoms) or (N_atoms)!')
        self.n_atom = self.atom_types.shape[-1]
        self.max_atomic_id = np.max(self.atom_types)
        # (N,A)
        self.atom_mask = self.atom_types > 0
        # (N,1)
        self.num_atoms = np.sum(self.atom_mask.astype(int), -1, keepdims=True)
        # ()
        self.tot_atoms = np.sum(self.num_atoms)
        self.single_molecule = False
        if self.atom_types.shape[0] == 1:
            self.single_molecule = True

        # (N,A,D)
        self.position = np.array(position, np.float64)
        print('='*105)
        if self.position.ndim != 3:
            raise ValueError(
                'The shape of position must be (N_data,N_atoms,Dim) but got '+str(self.position.shape)+'!')
        if self.position.shape[-2] != self.n_atom:
            raise ValueError('The number of in posiition ({:d}) does not match that in atom_types ({:d})!'
                             .format(self.position.shape[-2], self.n_atom))
        self.n_data = self.position.shape[0]
        if self.single_molecule:
            self.tot_atoms *= self.n_data
        elif self.atom_types.shape[0] != self.n_data:
            raise ValueError('The number of atom_types ({:d}) does not match the number of position ({:d})!'
                             .format(self.atom_types.shape[0], self.n_data))
        self.n_dim = self.position.shape[-1]

        print('Number of atoms: '+str(self.n_atom))
        print('Number of data: '+str(self.n_data))
        print('Dimension of space: '+str(self.n_dim))
        print('Total number of effective atoms: '+str(self.tot_atoms))
        print('Shape of atom types (Z): '+str(self.atom_types.shape))
        print('Shape of position (R): '+str(self.position.shape))

        # np.random.seed(seed)
        self.data_index = np.random.shuffle(np.arange(self.n_data))

        # (N,E)
        self.label = None
        self.n_label = 0
        if isinstance(label, (tuple, list)):
            self.label = np.concatenate(
                tuple(self._get_scalar(l, 'label') for l in label), axis=-1)
        else:
            self.label = self._get_scalar(label, 'label')
        self.n_label = self.label.shape[-1]
        print('Shape of label (E): '+str(self.label.shape))

        # (N_type,E)
        self.type_ref = None
        if type_ref is not None:
            if isinstance(type_ref, (tuple, list)):
                type_ref = np.concatenate(
                    tuple(self._get_ref(l, self.n_label, 'label') for l in type_ref), axis=-1)
            else:
                self.type_ref = self._get_ref(type_ref, self.n_label, 'label')
            print('Reference value for atom types:')
            info = '   Type '
            for i in range(self.n_label):
                info += '{:>10s}'.format('Label'+str(i))
            print(info)
            for i, ref in enumerate(self.type_ref):
                info = '   {:<7s} '.format(str(i)+':')
                for j in range(self.n_label):
                    info += '{:>10.2e}'.format(ref[j])
                print(info)

        # (N,A,D)
        self.force = None
        if force is not None:
            self.force = np.array(force, np.float64)
            if self.force.ndim != 3:
                raise ValueError(
                    'The shape of force must be (N_data,N_atoms,Dim)!')
            if self.force.shape[0] != self.n_data:
                raise ValueError(
                    'The number of force does not match the number of position!')
            if self.force.shape[-1] != self.n_dim:
                raise ValueError(
                    'The dimension of force does not match the dimension of position!')
            if self.force.shape[-2] != self.n_atom:
                raise ValueError(
                    'The number of atoms in force does not match the number in atom_type!')
            print('Shape of force (F): '+str(self.force.shape))

        print('Length unit: '+self.units.length_unit_name)
        print('Energy unit: '+self.units.energy_unit_name)
        print('-'*105)

        self.mol_avg = None
        self.mol_std = None
        self.mol_min = None
        self.mol_max = None
        self.mol_mid = None
        self.mol_range = None

        self.atom_avg = None
        self.atom_std = None
        self.atom_min = None
        self.atom_max = None
        self.atom_mid = None
        self.atom_range = None

        self.force_avg = None
        self.force_std = None
        self.force_max = None
        self.force_mid = None

        self.atomwise_mode = {'atomwise', 'atom', 'a'}
        self.graph_mode = {'graph', 'g', 'molecule', 'mol', 'm'}

        self.scaleshift_mode = {
            'atomwise': True,
            'atom': True,
            'a': True,
            'graph': False,
            'g': False,
            'molecule': False,
            'mol': False,
            'm': False,
        }

        self.mode_scaleshift = {
            True: 'atomwise',
            False: 'graph',
        }

        self._do_data_analysis()

        self.data_index = np.arange(self.n_data, dtype=np.int32)

    def set_atom_types(self, atom_types: ndarray):
        """set atom types"""
        self.atom_types = np.array(atom_types, np.int32)
        if self.atom_types.ndim == 1:
            self.atom_types = np.expand_dims(self.atom_types, 0)
        if self.atom_types.ndim != 2:
            raise ValueError(
                'The shape of atom_types must be (N_data,N_atoms) or (N_atoms)!')
        self.n_atom = self.atom_types.shape[-1]
        self.max_atomic_id = np.max(self.atom_types)
        # (N,A)
        self.atom_mask = self.atom_types > 0
        # (N,1)
        self.num_atoms = np.sum(self.atom_mask.astype(int), -1, keepdims=True)
        # ()
        self.tot_atoms = np.sum(self.num_atoms)
        self.single_molecule = False
        if self.atom_types.shape[0] == 1:
            self.single_molecule = True
            self.tot_atoms *= self.n_atom
        return self

    def shuffle_dataset(self):
        """shuffle the order of dataset"""
        index = np.arange(self.n_data, dtype=np.int32)
        np.random.shuffle(index)
        self.data_index = self.data_index[index]
        self._gather_dataset(index)
        return index

    def _gather_dataset(self, index: ndarray):
        """gather data by index"""
        if not self.single_molecule:
            self.set_atom_types(self.atom_types[index])
        self.position = self.position[index]
        self.label = self.label[index]
        if self.force is not None:
            self.force = self.force[index]
        return self

    def exclude_data(self, index: ndarray):
        """exclude data by index"""
        ex_idx = set(np.array(index).tolist())
        idx_ = set(self.data_index.tolist()) - ex_idx
        self.data_index = np.array(list(idx_), np.int32)
        origin_num = self.n_data
        self.n_data = self.data_index.size
        exculed_num = origin_num - self.n_data
        self._gather_dataset(self.data_index)
        print('Remove {:d} samples from {:d} dataset.'.format(exculed_num, self.n_data))
        print('-'*105)
        self._do_data_analysis()
        return self.data_index

    def _get_broadcast_value(self, value: float, name: str):
        """broadcast value"""
        value = np.array(value).reshape(-1)
        if value.size != self.n_label:
            if value.size == 1:
                value = value.repeat(self.n_label)
            else:
                raise ValueError('The number of  "'+name+'" ('+str(value.size)+
                                 ') does not match the number of label ('+str(self.n_label)+')')
        return value

    def set_graph_scaleshift(self, scale: float, shift: float):
        """set graph scale and shift"""
        self.graph_scale = self._get_broadcast_value(scale, 'scale')
        self.graph_shift = self._get_broadcast_value(shift, 'shift')
        return self

    def set_atomwise_scaleshift(self, scale: float, shift: float):
        """set atomwise scale and shift"""
        self.atom_scale = self._get_broadcast_value(scale, 'scale')
        self.atom_shift = self._get_broadcast_value(shift, 'shift')
        return self

    def get_scaleshift_mode(self, mode):
        """get the mode of scale and shift"""
        if isinstance(mode, str):
            mode = [self.scaleshift_mode[mode]]
        mode = np.array(mode)
        if mode.size != self.n_label:
            if mode.size == 1:
                mode = mode.repeat(self.n_label)
            else:
                raise ValueError('The number of "mode" ({:d}) '
                                 'does not match the number of label ({:d}).'.
                                 format(len(mode), self.n_label))
        if mode.dtype != bool:
            mode = np.array(itemgetter(*mode)(self.scaleshift_mode))
        return mode

    def get_scaleshift(self, atom_scale, atom_shift, graph_scale, graph_shift, mode):
        """set scale and shift"""
        atomwise_scaleshift = self.get_scaleshift_mode(mode)
        scale = np.select([atomwise_scaleshift], [atom_scale], graph_scale)
        shift = np.select([atomwise_scaleshift], [atom_shift], graph_shift)
        return scale, shift

    def get_normalized_dataset(self,
                               mode: str = 'atomwise',
                               scale: ndarray = None,
                               shift: ndarray = None,
                               index: list = Ellipsis,
                               potential_index: int = 0,
                               dtype: type = np.float32,
                               ) -> dict:

        """get normalized dataset"""

        atomwise_scaleshift = self.get_scaleshift_mode(mode)
        if scale is None:
            scale = np.where(atomwise_scaleshift, self.atom_std, self.mol_std)
        if shift is None:
            shift = np.where(atomwise_scaleshift, self.atom_avg, self.mol_avg)

        dataset = {}
        dataset['name'] = self.name
        dataset['num_atoms'] = self.n_atom
        data_index = self.data_index[index]
        dataset['num_dataset'] = data_index.size
        dataset['data_index'] = data_index
        dataset['normed_dataset'] = False
        dataset['length_unit'] = self.length_unit
        dataset['energy_unit'] = self.energy_unit
        if self.type_ref is not None:
            dataset['type_ref'] = self.type_ref.astype(dtype)

        atom_types = self.atom_types if self.single_molecule else self.atom_types[index]
        dataset['Z'] = atom_types
        dataset['R'] = self.position[index].astype(dtype)

        atom_mask = atom_types > 0
        num_atoms = np.sum(atom_mask.astype(int), -1, keepdims=True)
        label = self.label[index] - self.get_label_ref(atom_types, atom_mask)
        label = self.label_normalization(mode, label, scale, shift, num_atoms)
        dataset['E'] = label.astype(dtype)

        if self.force is not None:
            fscale = scale
            if self.n_label > 1:
                fscale = scale[potential_index]
            force = self.force[index] / fscale
            dataset['F'] = force.astype(dtype)

        dataset['scale'] = scale.astype(dtype)
        dataset['shift'] = shift.astype(dtype)
        dataset['scaleshift_mode'] = itemgetter(
            *atomwise_scaleshift)(self.mode_scaleshift)
        dataset['use_atomwise_scaleshift'] = atomwise_scaleshift
        if self.force is not None:
            # R_avg_force = E_avg / F_avg = E_avg / <dE/dR>
            dataset['avg_force_dis'] = (fscale / self.force_avg).astype(dtype)
        return dataset

    def get_origin_dataset(self, index: list = Ellipsis, dtype: type = np.float64, analysis_info: bool = True) -> dict:
        """get original dataset"""
        dataset = {}
        dataset['name'] = self.name
        dataset['num_atoms'] = self.n_atom
        data_index = self.data_index[index]
        dataset['num_dataset'] = data_index.size
        dataset['data_index'] = data_index
        dataset['normed_dataset'] = False
        dataset['length_unit'] = self.length_unit
        dataset['energy_unit'] = self.energy_unit
        if self.type_ref is not None:
            dataset['type_ref'] = self.type_ref.astype(dtype)

        dataset['Z'] = self.atom_types if self.single_molecule else self.atom_types[index]
        dataset['R'] = self.position[index].astype(dtype)
        dataset['E'] = self.label[index].astype(dtype)

        if self.force is not None:
            dataset['F'] = self.force[index].astype(dtype)

        if analysis_info:
            dataset['mol_avg'] = self.mol_avg.astype(dtype)
            dataset['mol_std'] = self.mol_std.astype(dtype)
            dataset['mol_min'] = self.mol_min.astype(dtype)
            dataset['mol_max'] = self.mol_max.astype(dtype)
            dataset['mol_mid'] = self.mol_mid.astype(dtype)

            dataset['atom_avg'] = self.atom_avg.astype(dtype)
            dataset['atom_std'] = self.atom_std.astype(dtype)
            dataset['atom_min'] = self.atom_min.astype(dtype)
            dataset['atom_max'] = self.atom_max.astype(dtype)
            dataset['atom_mid'] = self.atom_mid.astype(dtype)

            if self.force is not None:
                dataset['force_avg'] = self.force_avg.astype(dtype)
                dataset['force_std'] = self.force_std.astype(dtype)
                dataset['force_max'] = self.force_max.astype(dtype)
                dataset['force_mid'] = self.force_mid.astype(dtype)

        return dataset

    def get_dataset(self,
                    num_train=1024,
                    num_valid=128,
                    num_test=None,
                    norm_train=True,
                    norm_valid=True,
                    norm_test=False,
                    scale: ndarray = None,
                    shift: ndarray = None,
                    mode: str = 'atomwise',
                    potential_index: int = 0,
                    shuffle: bool = False,
                    dtype: type = np.float32,
                    ):
        """get dataset"""

        if num_test is None:
            num_test = self.n_data - num_train - num_valid
        if num_test < 0 or num_train + num_valid + num_test > self.n_data:
            raise ValueError('The total number of "num_train" ({:d}), "num_valid" ({:d}) and '
                             '"num_test" ({:d}) is larger than the number of dataset ({:d}).'.
                             format(num_train, num_valid, num_test, self.n_data))

        atomwise_scaleshift = self.get_scaleshift_mode(mode)
        if scale is None:
            scale = np.where(atomwise_scaleshift, self.atom_std, self.mol_std)
        if shift is None:
            shift = np.where(atomwise_scaleshift, self.atom_avg, self.mol_avg)

        ss_data = {}
        ss_data['scale'] = scale
        ss_data['shift'] = shift
        mode = itemgetter(*atomwise_scaleshift)(self.mode_scaleshift)
        ss_data['scaleshift_mode'] = mode
        ss_data['use_atomwise_scaleshift'] = atomwise_scaleshift
        if self.type_ref is not None:
            ss_data['type_ref'] = self.type_ref
        if self.force is not None:
            fscale = scale
            if self.n_label > 1:
                fscale = scale[potential_index]
            # average of unit displacement of force: <s_F> = <E> / <F> = <E> / <dE/dR>
            ss_data['avg_force_dis'] = fscale / self.force_avg

        if self.n_label == 1:
            print('   {:>16s}{:>16s}{:>12s}'.format('Scale', 'Shift', 'Mode'))
            print('   {:>16.6e}{:>16.6e}{:>12s}'.format(
                scale[0], shift[0], mode))
        else:
            print('   {:>2s}. {:>16s}{:>16s}{:>12s}'.format(
                'ID', 'Scale', 'Shift', 'Mode'))
            for i in range(self.n_label):
                print('   {:>2d}. {:>16.6e}{:>16.6e}{:>12s}'.format(
                    i, scale[i], shift[i], mode[i]))

        if shuffle:
            self.shuffle_dataset()

        idx_train = np.arange(num_train)
        idx_valid = np.arange(num_valid) + num_train
        idx_test = np.arange(num_test) + num_train + num_valid

        if norm_train:
            ds_train = self.get_normalized_dataset(
                atomwise_scaleshift, scale, shift, idx_train, potential_index, dtype)
        else:
            ds_train = self.get_origin_dataset(idx_train, dtype, False)
            ds_train.update(ss_data)

        if norm_valid:
            ds_valid = self.get_normalized_dataset(
                atomwise_scaleshift, scale, shift, idx_valid, potential_index, dtype)
        else:
            ds_valid = self.get_origin_dataset(idx_valid, dtype, False)
            ds_valid.update(ss_data)

        if norm_test:
            ds_test = self.get_normalized_dataset(
                atomwise_scaleshift, scale, shift, idx_test, potential_index, dtype)
        else:
            ds_test = self.get_origin_dataset(idx_test, dtype, False)
            ds_test.update(ss_data)

        print('-'*105)
        print('Number of Training, Validation and Test dataset: {:d}, {:d}, {:d}.'
              .format(num_train, num_valid, num_test))
        print('-'*105)

        return ds_train, ds_valid, ds_test

    def save_dataset(self,
                     prefix,
                     num_train=1024,
                     num_valid=128,
                     num_test=None,
                     norm_train=True,
                     norm_valid=True,
                     norm_test=False,
                     scale: ndarray = None,
                     shift: ndarray = None,
                     mode: str = 'atomwise',
                     potential_index: int = 0,
                     shuffle: bool = False,
                     ):
        """save dataset"""

        ds_train, ds_valid, ds_test = self.get_dataset(
            num_train,
            num_valid,
            num_test,
            norm_train,
            norm_valid,
            norm_test,
            scale,
            shift,
            mode,
            potential_index,
            shuffle
        )

        train_file = prefix + \
            ('_normed' if norm_train else '_origin') + \
            '_trainset_' + str(num_train) + '.npz'
        valid_file = prefix + \
            ('_normed' if norm_valid else '_origin') + \
            '_validset_' + str(num_valid) + '.npz'
        test_file = prefix + \
            ('_normed' if norm_test else '_origin') + \
            '_testset_' + str(num_test) + '.npz'

        print('Dataset file:')
        print('   Training dataset   {:>13s}:   {:s}'
              .format(('(normalized)' if norm_train else '(origin)'), train_file))
        print('   Validation dataset {:>13s}:   {:s}'
              .format(('(normalized)' if norm_valid else '(origin)'), valid_file))
        print('   Test dataset       {:>13s}:   {:s}'
              .format(('(normalized)' if norm_test else '(origin)'), test_file))
        print('='*105)

        np.savez(train_file, **ds_train)
        np.savez(valid_file, **ds_valid)
        np.savez(test_file, **ds_test)

        return ds_train, ds_valid, ds_test

    def convert_units(self, length_unit: str = None, energy_unit: str = None, energy_index: list = Ellipsis):
        """convert units of dataset"""
        if length_unit is not None:
            scale = self.units.convert_length_to(length_unit)
            self.position *= scale
            if self.force is not None:
                self.force /= scale
            old_unit = self.units.length_unit_name
            self.units.set_length_unit(length_unit)
            self.length_unit = self.units.length_unit
            print('Change the length unit from "'+old_unit+'" to "' +
                  self.units.length_unit_name+'"')

        if energy_unit is None:
            if length_unit is not None:
                print('='*105)
        else:
            scale = self.units.convert_energy_to(energy_unit)
            self.label[:, energy_index] *= scale
            if self.type_ref is not None:
                self.type_ref[:, energy_index] *= scale

            if self.force is not None:
                self.force *= scale

            old_unit = self.units.energy_unit_name
            self.units.set_energy_unit(energy_unit)
            self.energy_unit = self.units.energy_unit
            print('Change the energy unit from "'+old_unit+'" to "' +
                  self.units.energy_unit_name+'"')
            print('-'*105)
            self._do_data_analysis()

    def label_analysis(self, label, num_atoms=None):
        """analyse label"""
        def _label_analysis(label):
            lavg = np.mean(label, 0)
            lstd = np.std(label, 0)
            lmin = np.min(label, 0)
            lmax = np.max(label, 0)
            lmid = np.median(label, 0)
            return lavg, lstd, lmin, lmax, lmid

        def _show_label_analysis(lavg, lstd, lmin, lmax, lmid):
            lrange = lmax - lmin
            if self.n_label == 1:
                print('   {:>16s}{:>16s}{:>16s}{:>16s}{:>16s}{:>16s}'.format(
                    'Mean', 'STD', 'Min', 'Max', 'Median', 'Range'))
                for i in range(self.n_label):
                    print('   {:>16.6e}{:>16.6e}{:>16.6e}{:>16.6e}{:>16.6e}{:>16.6e}'.format(
                        lavg[0], lstd[0], lmin[0], lmax[0], lmid[0], lrange[0]))
            else:
                print('   ID  {:>16s}{:>16s}{:>16s}{:>16s}{:>16s}{:>16s}'.format(
                    'Mean', 'STD', 'Min', 'Max', 'Median', 'Range'))
                for i in range(self.n_label):
                    print('   {:>2d}: {:>16.6e}{:>16.6e}{:>16.6e}{:>16.6e}{:>16.6e}{:>16.6e}'.format(
                        i, lavg[i], lstd[i], lmin[i], lmax[i], lmid[i], lrange[i]))
            print('-'*105)

        print('Label analysis according to graph:')
        mol_info = _label_analysis(label)
        _show_label_analysis(*mol_info)
        print('Label analysis according to each atom:')
        atom_info = _label_analysis(label/num_atoms)
        _show_label_analysis(*atom_info)
        return mol_info, atom_info

    def force_analysis(self, force: ndarray, atom_mask: ndarray = None, tot_atoms: ndarray = None):
        """analyse force"""
        # (N,A)
        if atom_mask is None:
            fnorm = np.linalg.norm(force, axis=-1)
            favg = np.mean(fnorm)
            fstd = np.std(fnorm)
        else:
            fnorm = np.linalg.norm(force, axis=-1) * atom_mask
            if tot_atoms is None:
                tot_atoms = np.sum(atom_mask.astype(int))
            favg = np.sum(fnorm) / tot_atoms
            favg2 = np.sum(fnorm*fnorm) / tot_atoms
            fstd = np.sqrt(favg2-favg*favg)

        fmax = np.max(fnorm)
        fmid = np.median(fnorm)

        print('Force norm analysis:')
        print('   {:>16s}{:>16s}{:>16s}{:>16s}'.format(
            'Mean', 'STD', 'Max', 'Median'))
        print('   {:>16.6e}{:>16.6e}{:>16.6e}{:>16.6e}'.format(
            favg, fstd, fmax, fmid))
        print('-'*105)

        return favg, fstd, fmax, fmid

    def _do_data_analysis(self):
        """analyse label"""
        label = self.label - \
            self.get_label_ref(self.atom_types, self.atom_mask)
        mol_info, atom_info = self.label_analysis(label, self.num_atoms)
        (self.mol_avg, self.mol_std, self.mol_min,
         self.mol_max, self.mol_mid) = mol_info
        (self.atom_avg, self.atom_std, self.atom_min,
         self.atom_max, self.atom_mid) = atom_info
        if self.force is not None:
            (self.force_avg, self.force_std, self.force_max, self.force_mid) = \
                self.force_analysis(self.force, self.atom_mask, self.tot_atoms)
        print('Number of data: '+str(label.shape[0]))
        print('='*105)
        return self

    def get_label_ref(self, atom_types: ndarray, atom_mask: ndarray = None) -> ndarray:
        """get reference of each label"""
        if self.type_ref is None:
            return 0

        if atom_mask is None:
            atom_mask = atom_types > 0
        # (N,A,E) = (N,A,E) * (N,A,1)
        ref = self.type_ref[atom_types] * np.expand_dims(atom_mask, -1)
        # (N,E)
        return np.sum(ref, axis=-2)

    def label_normalization(self, mode: str, label: ndarray, scale: float, shift: float, num_atoms: int) -> ndarray:
        """normalize label"""
        atomwise_scaleshift = self.get_scaleshift_mode(mode)
        if atomwise_scaleshift.all():
            return (label - shift * num_atoms) / scale
        if not atomwise_scaleshift.any():
            return (label - shift) / scale

        atomwsie_norm = (label - shift * num_atoms) / scale
        graph_norm = (label - shift) / scale
        return np.select([atomwise_scaleshift], [atomwsie_norm], graph_norm)

    def _get_scalar(self, data: ndarray, name: str):
        """get scalar from input"""
        scalar = np.array(data, np.float64)
        if scalar.ndim == 1:
            scalar = np.expand_dims(data, 0)
        if scalar.ndim != 2:
            raise ValueError('The shape of '+name +
                             ' must be (N_data,N_'+name+') or (N_data)')
        if scalar.shape[0] != self.n_data:
            raise ValueError('The number of '+name +
                             ' does not match the number of position!')
        return scalar

    def _get_ref(self, data: ndarray, n_ref: int, name: str) -> ndarray:
        """get reference value"""
        ref = np.array(data, np.float64)
        if ref.ndim == 1:
            ref = np.expand_dims(ref, -1)
        if ref.ndim != 2:
            raise ValueError('The shape of '+name +'_ref should be (N_type, N_'+name+')')
        if ref.shape[0] <= self.max_atomic_id:
            raise ValueError('The size of "'+name+'_ref" '+str(ref.shape[0]) +
                             ' is over the range of maximum atomic index ('+str(self.max_atomic_id)+')')
        if ref.shape[-1] != n_ref:
            name_ref = name + '_ref'
            raise ValueError('The number of "{:s}" in "{:s}" ({:d}) mismatch ({:d}).'.
                             format(name, name_ref, ref.shape[-1], n_ref))
        return ref

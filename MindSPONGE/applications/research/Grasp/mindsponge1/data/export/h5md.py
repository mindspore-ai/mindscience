# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
Export H5MD file.
"""

import os
import numpy as np
from numpy import ndarray
import h5py
from h5py import Group
from mindspore.train._utils import _make_directory

from ...system import Molecule
from ...function.units import Units, global_units

_cur_dir = os.getcwd()


class H5MD:
    r"""write HDF5 molecular data (H5MD) hdf5_file

    Reference:

        de Buyl, P.; Colberg, P. H., Höfling, F.,
        H5MD: A structured, efficient, and portable file format for molecular data [J].
        Computer Physics Communications, 2014, 185(6): 1546-1553.

    Args:

        system (Molecule):      Simulation system

        filename (str):         Name of output H5MD hdf5_file.

        directory (str):        Directory of the output hdf5_file. Default: None

        write_velocity (bool):  Whether to write the velocity of the system to the H5MD file.
                                Default:  False

        write_force (bool):     Whether to write the forece of the system to the H5MD file.
                                Default: False

        length_unit (str):      Length unit for coordinates.
                                If given "None", it will be equal to the length unit of the system.
                                Default: None

        energy_unit (str):      Energy unit.
                                If given "None", it will be equal to the global energy unit.
                                Default: None.

        compression (str):      Compression strategy for HDF5. Default: 'gzip'

        compression_opts (int): Compression settings for HDF5. Default: 4

        mode (str):             I/O mode for HDF5. Default: 'w'

    """

    def __init__(self,
                 system: Molecule,
                 filename: str,
                 directory: str = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 compression: str = 'gzip',
                 compression_opts: int = 4,
                 mode: str = 'w',
                 ):

        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir
        self.filename = os.path.join(self._directory, filename)

        self.hdf5_file = h5py.File(self.filename, mode)

        self.h5md = self.hdf5_file.create_group('h5md')
        self.h5md.attrs['version'] = [1, 1]

        self.h5md_author = self.h5md.create_group('author')
        self.h5md_author.attrs['name'] = 'AIMM Group @ Shenzhen Bay Laboratory & Peking University'
        self.h5md_author.attrs['email'] = 'yangyi@szbl.ac.cn'

        self.h5md_creator = self.h5md.create_group('creator')
        self.h5md_creator.attrs['name'] = 'MindSPONGE'
        self.h5md_creator.attrs['version'] = '0.5'

        if length_unit is None:
            length_unit = system.length_unit
        if energy_unit is None:
            energy_unit = global_units.energy_unit
        self.units = Units(length_unit, energy_unit)

        self.num_walker = system.num_walker
        self.num_atoms = system.num_atoms
        self.dimension = system.dimension
        self.coordinate = system.coordinate.asnumpy()
        self.crd_shape = (None, self.num_atoms, self.dimension)

        self.pbc_box = system.pbc_box
        self.use_pbc = False
        if self.pbc_box is not None:
            self.pbc_box = system.pbc_box.asnumpy()
            self.use_pbc = True

        self.compression = compression
        self.compression_opts = compression_opts

        atomic_number = None
        if system.atomic_number is not None:
            atomic_number = system.atomic_number.asnumpy()[0]

        self.length_unit_scale = self.units.convert_length_from(
            system.units)
        self.force_unit_scale = self.units.convert_energy_from(
            system.units) / self.length_unit_scale

        self.time_unit = 'ps'

        atom_name = None
        if system.atom_name is not None:
            atom_name = [s.encode('ascii', 'ignore')
                         for s in system.atom_name[0].tolist()]

        atom_type = None
        if system.atom_type is not None:
            atom_type = [s.encode('ascii', 'ignore')
                         for s in system.atom_type[0].tolist()]

        resname = None
        if system.residue_name is not None:
            resname = [s.encode('ascii', 'ignore')
                       for s in system.residue_name.tolist()]

        resid = None
        if system.atom_resid is not None:
            resid = system.atom_resid.asnumpy()

        bond_from = None
        bond_to = None
        if system.bond is not None:
            bond_from = system.bond[0][..., 0].asnumpy() + 1
            bond_to = system.bond[0][..., 1].asnumpy() + 1

        species = np.arange(self.num_atoms, dtype=np.int32)

        self.parameters = self.hdf5_file.create_group('parameters')
        self.vmd_structure = self.create_vmd_structure(species, atomic_number, atom_name, atom_type,
                                                       resid, resname, bond_from, bond_to)

        self.shape = (self.num_atoms, self.dimension)
        self.particles = self.hdf5_file.create_group('particles')

        if self.num_walker > 1:
            self.position = []
            self.velocity = []
            self.force = []
            self.box = []
            self.trajectory = []
            for i in range(self.num_walker):
                name = 'trajectory' + str(i)
                trajectory = self.create_trajectory(species, name)
                self.trajectory.append(trajectory)

                self.position.append(self.create_position(
                    self.trajectory[i], self.shape))
                self.box.append(self.create_box(
                    self.trajectory[i], self.shape))

        else:
            self.trajectory = self.create_trajectory(species, 'trajectory')
            self.position = self.create_position(self.trajectory, self.shape)
            self.box = self.create_box(self.trajectory, self.use_pbc)

        self.image = None
        self.edges = None
        self.velocity = None
        self.force = None

        self.observables = self.hdf5_file.create_group('observables')
        self.obs_group = None

    def create_element(self, group: h5py.Group, name: str, shape: tuple, dtype: str, unit: str = None) -> h5py.Group:
        """create element"""
        element = group.create_group(name)
        element.create_dataset('step', shape=(0,), dtype='int32', maxshape=(None,),
                               compression=self.compression, compression_opts=self.compression_opts)
        element.create_dataset('time', shape=(0,), dtype='float32', maxshape=(None,),
                               compression=self.compression, compression_opts=self.compression_opts)
        element.create_dataset('value', shape=(0,)+shape, dtype=dtype, maxshape=(None,)+shape,
                               compression=self.compression, compression_opts=self.compression_opts)
        element['time'].attrs['unit'] = self.time_unit
        if unit is not None:
            element['value'].attrs['unit'] = unit.encode('ascii', 'ignore')
        return element

    def create_vmd_structure(self,
                             species: ndarray,
                             atomic_number: ndarray = None,
                             atom_name: ndarray = None,
                             atom_type: ndarray = None,
                             resid: ndarray = None,
                             resname: ndarray = None,
                             bond_from: ndarray = None,
                             bond_to: ndarray = None,
                             ):
        """create the group 'vmd_structure'"""

        vmd_structure = self.parameters.create_group('vmd_structure')
        vmd_structure.create_dataset(
            'indexOfSpecies', dtype='int32', data=species,
            compression=self.compression, compression_opts=self.compression_opts)

        if atomic_number is not None:
            vmd_structure.create_dataset('atomicnumber', dtype='int32', data=atomic_number,
                                         compression=self.compression, compression_opts=self.compression_opts)
        if atom_name is not None:
            vmd_structure.create_dataset('name', data=atom_name,
                                         compression=self.compression, compression_opts=self.compression_opts)
        if atom_type is not None:
            vmd_structure.create_dataset('type', data=atom_type,
                                         compression=self.compression, compression_opts=self.compression_opts)
        if resid is not None:
            vmd_structure.create_dataset('resid', dtype='int32', data=resid,
                                         compression=self.compression, compression_opts=self.compression_opts)
        if resname is not None:
            vmd_structure.create_dataset('resname', data=resname,
                                         compression=self.compression, compression_opts=self.compression_opts)
        if bond_from is not None:
            vmd_structure.create_dataset('bond_from', dtype='int32', data=bond_from,
                                         compression=self.compression, compression_opts=self.compression_opts)
            vmd_structure.create_dataset('bond_to', dtype='int32', data=bond_to,
                                         compression=self.compression, compression_opts=self.compression_opts)

        return vmd_structure

    def create_trajectory(self, species: ndarray, name: str = 'trajectory') -> h5py.Group:
        """create the group 'trajectory'"""
        trajectory = self.particles.create_group(name)
        trajectory.create_dataset('species', dtype='int32', data=species,
                                  compression=self.compression, compression_opts=self.compression_opts)
        return trajectory

    def create_position(self, trajectory: h5py.Group, shape: tuple) -> h5py.Group:
        """create the group 'position'"""
        return self.create_element(trajectory, 'position', shape, 'float32', self.units.length_unit_name)

    def create_box(self, trajectory: h5py.Group, use_pbc: ndarray = None) -> h5py.Group:
        """create the group 'box'"""
        box = trajectory.create_group('box')
        box.attrs['dimension'] = self.dimension
        if use_pbc is None:
            box.attrs['boundary'] = ['none'] * self.dimension
        else:
            box.attrs['boundary'] = ['periodic'] * self.dimension
        return box

    def create_edges(self, box: h5py.Group, pbc_box: ndarray = None):
        """create edges"""
        if pbc_box is None:
            edges = self.create_element(
                box, 'edges', (self.dimension,), 'float32', self.units.length_unit_name)
        else:
            pbc_box *= self.length_unit_scale
            edges = box.create_dataset('edges', data=pbc_box, dtype='float32',
                                       compression=self.compression, compression_opts=self.compression_opts)
            edges.attrs['unit'] = self.units.length_unit_name.encode(
                'ascii', 'ignore')
        return edges

    def create_image(self, trajectory: h5py.Group, shape: tuple) -> h5py.Group:
        """create the group 'image'"""
        return self.create_element(trajectory, 'image', shape, 'int8')

    def create_velocity(self, trajectory: h5py.Group, shape: tuple) -> h5py.Group:
        """create the group 'velocity'"""
        return self.create_element(trajectory, 'velocity', shape, 'float32', self.units.velocity_unit_name)

    def create_force(self, trajectory: h5py.Group, shape: tuple) -> h5py.Group:
        """create the group 'force'"""
        return self.create_element(trajectory, 'force', shape, 'float32', self.units.force_unit_name)

    def create_obs_group(self, name: str = 'trajectory') -> h5py.Group:
        obs_group = self.observables.create_group(name)
        obs_group.attrs['dimension'] = self.dimension
        obs_group.create_dataset('particle_number', dtype='int32', data=[self.num_atoms],
                                 compression=self.compression, compression_opts=self.compression_opts)
        return obs_group

    def set_box(self, constant_volume: bool = True):
        """set PBC box information"""
        if self.pbc_box is not None:
            if self.num_walker > 1:
                self.edges = []
                for i in range(self.num_walker):
                    if constant_volume:
                        self.edges.append(self.create_edges(
                            self.box, self.pbc_box[i]))
                    else:
                        self.edges.append(self.create_edges(self.box))
            else:
                if constant_volume:
                    self.edges = self.create_edges(self.box, self.pbc_box[0])
                else:
                    self.edges = self.create_edges(self.box)
        return self

    def set_image(self):
        """set group 'image'"""
        if self.num_walker > 1:
            self.image = []
            for i in range(self.num_walker):
                self.force.append(self.create_image(
                    self.trajectory[i], self.shape))
        else:
            self.image = self.create_image(self.trajectory, self.shape)
        return self

    def set_velocity(self):
        """set group 'velocity'"""
        if self.num_walker > 1:
            self.velocity = []
            for i in range(self.num_walker):
                self.velocity.append(self.create_velocity(
                    self.trajectory[i], self.shape))
        else:
            self.velocity = self.create_velocity(self.trajectory, self.shape)
        return self

    def set_force(self):
        """set group 'force'"""
        if self.num_walker > 1:
            self.force = []
            for i in range(self.num_walker):
                self.force.append(self.create_force(
                    self.trajectory[i], self.shape))
        else:
            self.force = self.create_force(self.trajectory, self.shape)
        return self

    def set_observables(self, names: list, shapes: list, dtypes: list, units: list):
        """set observables"""
        if self.num_walker > 1:
            self.obs_group = []
            for i in range(self.num_walker):
                obs_group = self.create_obs_group('trajectory' + str(i))
                for name, shape, dtype, unit in zip(names, shapes, dtypes, units):
                    self.create_element(obs_group, name, shape, dtype, unit)
                self.obs_group.append(obs_group)
        else:
            self.obs_group = self.create_obs_group('trajectory')
            for name, shape, dtype, unit in zip(names, shapes, dtypes, units):
                self.create_element(self.obs_group, name, shape, dtype, unit)
        return self

    def write_element(self, group: Group, step: int, time: float, value: ndarray):
        """write the element to H5MD file"""
        ds_step = group['step']
        ds_step.resize(ds_step.shape[0]+1, axis=0)
        ds_step[-1] = step

        ds_time = group['time']
        ds_time.resize(ds_time.shape[0]+1, axis=0)
        ds_time[-1] = time

        ds_value = group['value']
        ds_value.resize(ds_value.shape[0]+1, axis=0)
        ds_value[-1] = value
        return self

    def write_position(self, step: int, time: float, position: ndarray):
        """write position"""
        position *= self.length_unit_scale
        if self.num_walker == 1:
            self.write_element(self.position, step, time, position[0])
        else:
            for i in range(self.num_walker):
                self.write_element(
                    self.position[i], step, time, position[i])
        return self

    def write_box(self, step: int, time: float, box: ndarray):
        """write box"""
        box *= self.length_unit_scale
        if self.num_walker == 1:
            self.write_element(self.edges, step, time, box[0])
        else:
            for i in range(self.num_walker):
                self.write_element(self.edges[i], step, time, box[i])
        return self

    def write_image(self, step: int, time: float, image: ndarray):
        """write image"""
        if self.num_walker == 1:
            self.write_element(self.image, step, time,
                               image[0].astype(np.int8))
        else:
            for i in range(self.num_walker):
                self.write_element(
                    self.image[i], step, time, image[i].astype(np.int8))
        return self

    def write_velocity(self, step: int, time: float, velocity: ndarray):
        """write velocity"""
        velocity *= self.length_unit_scale
        if self.num_walker == 1:
            self.write_element(self.velocity, step, time, velocity[0])
        else:
            for i in range(self.num_walker):
                self.write_element(
                    self.velocity[i], step, time, velocity[i])
        return self

    def write_force(self, step: int, time: float, force: ndarray):
        """write force"""
        force *= self.force_unit_scale
        if self.num_walker == 1:
            self.write_element(self.force, step, time, force[0])
        else:
            for i in range(self.num_walker):
                self.write_element(
                    self.force[i], step, time, force[i])
        return self

    def write_observables(self, names: list, step: int, time: float, values: list, index: int = None):
        """write observables"""
        if index is None and self.num_walker > 1:
            raise ValueError(
                'The "index" must given when using muliple walkers')
        if self.num_walker == 1:
            for name, value in zip(names, values):
                self.write_element(self.obs_group[name], step, time, value)
        else:
            for name, value in zip(names, values):
                self.write_element(
                    self.obs_group[index][name], step, time, value)
        return self

    def close(self):
        """close the HDF5 file"""
        return self.hdf5_file.close()

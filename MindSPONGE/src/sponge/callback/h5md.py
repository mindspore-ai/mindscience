# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
Callback to write H5MD trajectory file
"""

import signal
import sys
from typing import List
import numpy as np

from mindspore.train.callback import Callback, RunContext

from ..system import Molecule
from ..optimizer import Updater
from ..data.export import H5MD
from ..system.modelling.pdb_generator import gen_pdb


class WriteH5MD(Callback):
    r"""Callback to write HDF5 molecular data (H5MD) file

    Args:
        system (Molecule):      Simulation system (system module of MindSPONGE).

        filename (str):         Name of output H5MD file.

        directory (str):        Directory of the output file. Default: ``None``.

        mode (str):             I/O mode for H5MD.

                                - ``'w'``             Create file, truncate if exists.
                                - ``'w-'`` or ``'x'``:    Create file, fail if exists.
                                - ``'a'``:            Read/write if exists, create otherwise.

        write_velocity (bool):  Whether to write the velocity of the system to the H5MD file.
                                Default: ``False``.

        write_force (bool):     Whether to write the forece of the system to the H5MD file.
                                Default: ``False``.

        wiite_image (bool):     Whether to write the image of the position of system to the H5MD file.
                                Default: ``False``.

        length_unit (str):      Length unit for coordinates. Default: ``None``.

        energy_unit (str):      Energy unit. Default: ``None``.

        dtype (str):            Data type for H5MD. Default: ``'float32'``.

        compression (str):      Compression strategy for HDF5. Default: ``'gzip'``.

        compression_opts (int): Compression settings for HDF5. Default: ``4``.

        auto_close (bool):      Whether to automatically close the writing of H5MD files at the end of
                                the simulation process. Default: ``True``.
        save_last_pdb (str):    Decide to store the last crd in a pdb format file or not. If choose to store the pdb,
                                the value should be string format pdb file name. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 system: Molecule,
                 filename: str,
                 save_freq: int = 1,
                 directory: str = None,
                 mode: str = 'w',
                 write_velocity: bool = False,
                 write_force: bool = False,
                 write_image: bool = True,
                 write_metrics: bool = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 dtype: str = 'float32',
                 compression: str = 'gzip',
                 compression_opts: int = 4,
                 auto_close: bool = True,
                 save_last_pdb: str = None
                 ):

        if mode not in ['w', 'w-', 'x', 'a']:
            raise ValueError(f'WriteH5MD only support "w", "w-", "x" and "a" but got: {mode}')

        self.system = system
        self.num_walker = system.num_walker
        self.units = system.units
        self.h5md = H5MD(self.system, filename, directory, mode=mode,
                         length_unit=length_unit, energy_unit=energy_unit,
                         compression=compression, compression_opts=compression_opts)

        self.convert_to_angstram = self.system.units.convert_length_to('A')
        self.use_pbc = system.pbc_box is not None
        self.const_volume = True
        self.dtype = dtype
        self.save_freq = save_freq
        self.auto_close = auto_close

        self.potential = 0
        self.energies = 0
        self.kinetics = 0
        self.tot_energy = 0
        self.temperature = 0
        self.pressure = 0
        self.volume = 0
        self.bias = 0
        self.biases = 0

        self.write_image = write_image
        self.write_velocity = write_velocity
        self.write_force = write_force

        if save_last_pdb is None:
            self.last_pdb_name = None
            self.save_pdb = False
        else:
            self.save_pdb = True
            self.last_pdb_name = save_last_pdb

        if mode == 'a':
            self.init_h5md = False
        else:
            self.init_h5md = True
            if self.use_pbc and self.write_image:
                self.h5md.set_image()

            if self.write_velocity:
                self.h5md.set_velocity()

            if self.write_force:
                self.h5md.set_force()

            self.h5md.add_observables('potential_energy', (), self.dtype, self.units.energy_unit_name)
            self.h5md.add_observables('total_energy', (), self.dtype, self.units.energy_unit_name)

        self.use_updater = None

        self.write_metrics = write_metrics
        self.write_energies = False
        self.write_bias = False

        self.count = 0
        self.count_records = 0

        # Detect process kill signal and save the data.
        signal.signal(signal.SIGINT, self.__stop__)
        signal.signal(signal.SIGTERM, self.__stop__)

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def __stop__(self, signal_, frame_):
        """
        Save data when process killed.
        """
        # pylint: disable=unused-argument
        print(f'\n\033[33mProgram process terminated. {self.count_records} steps saved in H5MD file.\033[0m\n')
        self.close()
        sys.exit(0)

    def close(self):
        if self.save_pdb:
            self.save_to_pdb()
        self.h5md.close()
        return self

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()

        def _init_energies(num_energies: int, energy_names: List[str]):
            if num_energies == 0:
                return

            self.write_energies = True
            if self.init_h5md:
                energy_names = [name.encode('ascii', 'ignore') for name in energy_names]
                energies = self.h5md.add_observables(
                    'energies', (num_energies,), self.dtype, self.units.energy_unit_name)
                self.h5md.create_dataset(energies, 'labels', shape=(1, num_energies), data=energy_names)
            return

        def _init_biases(num_biases: int, bias_names: List[str]):
            if num_biases == 0:
                return

            self.write_bias = True
            if self.init_h5md:
                self.h5md.add_observables('bias_potential', (), self.dtype, self.units.energy_unit_name)
                bias_names = [name.encode('ascii', 'ignore') for name in bias_names]
                biases = self.h5md.add_observables(
                    'biases', (num_biases,), self.dtype, self.units.energy_unit_name)
                self.h5md.create_dataset(biases, 'labels', shape=(1, num_biases), data=bias_names)
            return

        _init_energies(cb_params.num_energies, cb_params.energy_names)
        _init_biases(cb_params.num_biases, cb_params.bias_names)

        if isinstance(cb_params.optimizer, Updater):
            self.use_updater = True
            if self.init_h5md:
                self.h5md.add_observables('kinetic_energy', (), self.dtype, self.units.energy_unit_name)
                self.h5md.add_observables('temperature', (), self.dtype, 'K')
            if self.use_pbc:
                self.const_volume = cb_params.barostat is None
                if self.init_h5md:
                    self.h5md.add_observables('pressure', (), self.dtype, 'bar')
                    self.h5md.add_observables('volume', (), self.dtype, self.units.volume_unit_name)
                    self.h5md.set_box(self.const_volume)
        else:
            self.use_updater = False
            if self.use_pbc and self.init_h5md:
                self.h5md.set_box(True)
            if self.write_velocity and not isinstance(cb_params.optimizer, Updater):
                self.write_velocity = False
                print(f'Warning! The optimizer "{cb_params.optimizer}" does not has the attribute "velocity".')

        if cb_params.metrics is None:
            if self.write_metrics:
                print('[WARNING] No Metrics found. Metrics will not be output to H5MD file.')
            self.write_metrics = False
        else:
            if self.write_metrics is None:
                self.write_metrics = True
            if self.init_h5md:
                for name, shape, unit in \
                    zip(cb_params.metrics, cb_params.metrics_shape, cb_params.metrics_units):
                    shape_ = () if shape == (1,) else shape
                    self.h5md.add_observables(name, shape_, self.dtype, unit)

        self.init_h5md = False

    def epoch_begin(self, run_context: RunContext):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_end(self, run_context: RunContext):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_begin(self, run_context: RunContext):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        if self.count % self.save_freq == 0:
            cb_params = run_context.original_args()
            step = cb_params.cur_step
            self.h5md.write_step(step)
            time = cb_params.cur_time
            self.h5md.write_time(time)
            if self.use_updater:
                self.kinetics = cb_params.kinetics.copy().asnumpy().squeeze()
                self.h5md.write_observables('kinetic_energy', self.kinetics)
                self.temperature = cb_params.temperature.copy().asnumpy().squeeze()
                self.h5md.write_observables('temperature', self.temperature)
            cb_params = run_context.original_args()
            coordinate = cb_params.coordinate.copy().asnumpy().squeeze()
            self.h5md.write_position(coordinate)

            if self.use_pbc:
                if not self.const_volume:
                    pbc_box = cb_params.pbc_box.copy().asnumpy().squeeze()
                    self.h5md.write_box(pbc_box)
                if self.write_image:
                    image = self.system.calc_image().asnumpy().squeeze()
                    self.h5md.write_image(image)

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        if self.count % self.save_freq == 0:
            cb_params = run_context.original_args()

            self.potential = cb_params.potential.copy().asnumpy().squeeze()
            self.h5md.write_observables('potential_energy', self.potential)

            if self.write_energies:
                self.energies = cb_params.energies.copy().asnumpy().squeeze()
                self.h5md.write_observables('energies', self.energies)

            if self.write_bias:
                self.bias = cb_params.bias.copy().asnumpy().squeeze()
                self.h5md.write_observables('bias_potential', self.bias)
                self.biases = cb_params.biases.copy().asnumpy().squeeze()
                self.h5md.write_observables('biases', self.biases)

            self.tot_energy = self.potential
            if self.use_updater:
                self.tot_energy += + self.kinetics
                if self.use_pbc:
                    self.pressure = cb_params.pressure.copy().asnumpy().squeeze()
                    self.h5md.write_observables('pressure', self.pressure)
                    self.volume = cb_params.volume.copy().asnumpy().squeeze()
                    self.h5md.write_observables('volume', self.volume)
            self.h5md.write_observables('total_energy', self.tot_energy)

            if self.write_metrics:
                metrics: dict = cb_params.analyse()
                for metric, value in metrics.items():
                    self.h5md.write_observables(metric, value)

            if self.write_velocity:
                velocity = cb_params.velocity.copy().asnumpy().squeeze()
                self.h5md.write_velocity(velocity)
            if self.write_force:
                force = cb_params.force.copy().asnumpy().squeeze()
                self.h5md.write_force(force)

            self.count_records += 1

        self.count += 1

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        #pylint: disable=unused-argument
        if self.auto_close:
            self.close()

    def save_to_pdb(self):
        """ Save the system information into a pdb file.
        """
        last_resname = self.system.residue_name
        if len(last_resname[0]) == 4:
            last_resname[0] = last_resname[0][1:]
        if len(last_resname[-1]) == 4:
            last_resname[-1] = last_resname[-1][1:]
        gen_pdb(self.system.coordinate.asnumpy() * self.convert_to_angstram,
                self.system.atom_name[0],
                np.take(last_resname, self.system.atom_resid),
                self.system.atom_resid.asnumpy() + 1,
                pdb_name=self.last_pdb_name)

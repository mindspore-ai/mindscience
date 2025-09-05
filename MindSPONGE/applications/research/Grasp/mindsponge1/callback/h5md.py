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
Callback to write H5MD trajectory file
"""

from mindspore.train.callback import Callback, RunContext

from ..system import Molecule
from ..optimizer import Updater
from ..data.export import H5MD


class WriteH5MD(Callback):
    r"""
    Callback to write HDF5 molecular data (H5MD) file.

    Args:
        system (Molecule):      Simulation system
        filename (str):         Name of output H5MD file.
        save_freq(int):         Saved frequency. Default: 1
        directory (str):        Directory of the output file. Default: None
        write_velocity (bool):  Whether to write the velocity of the system to the H5MD file.
                                Default:  False
        write_force (bool):     Whether to write the forece of the system to the H5MD file.
                                Default: False
        write_image (bool):     Whether to write the image of the position of system to the H5MD file.
                                Default: False
        length_unit (str):      Length unit for coordinates. Default: None.
        energy_unit (str):      Energy unit. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 system: Molecule,
                 filename: str,
                 save_freq: int = 1,
                 directory: str = None,
                 write_velocity: bool = False,
                 write_force: bool = False,
                 write_image: bool = True,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):

        self.system = system
        self.units = system.units
        self.h5md = H5MD(self.system, filename, directory,
                         length_unit, energy_unit)

        self.use_pbc = system.pbc_box is not None
        self.const_volume = True

        self.write_image = write_image
        if self.use_pbc and self.write_image:
            self.h5md.set_image()

        self.save_freq = save_freq

        self.write_velocity = write_velocity
        if self.write_velocity:
            self.h5md.set_velocity()

        self.write_force = write_force
        if self.write_force:
            self.h5md.set_force()

        self.potential = 0
        self.kinetics = 0
        self.tot_energy = 0
        self.temperature = 0
        self.pressure = 0
        self.volume = 0

        self.observables = [
            'potential_energy',
            'kinetic_energy',
            'total_energy',
            'temperature',
            'pressure',
            'volume',
        ]

        self.obs_units = [
            self.units.energy_unit_name,
            self.units.energy_unit_name,
            self.units.energy_unit_name,
            'K',
            'bar',
            self.units.volume_unit_name,
        ]

        self.obs_dtypes = [
            'float32',
            'float32',
            'float32',
            'float32',
            'float32',
            'float32',
        ]

        self.obs_shapes = [
            (),
            (),
            (),
            (),
            (),
            (),
        ]

        self.h5md.set_observables(
            self.observables, self.obs_shapes, self.obs_dtypes, self.obs_units)

        self.use_updater = None

        self.count = 0

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        if isinstance(cb_params.optimizer, Updater):
            self.use_updater = True
            if self.use_pbc:
                self.const_volume = cb_params.barostat is None
                self.h5md.set_box(self.const_volume)
        else:
            self.use_updater = False
            if self.use_pbc:
                self.h5md.set_box(True)
            if self.write_velocity and not isinstance(cb_params.optimizer, Updater):
                self.write_velocity = False
                print('Warning! The optimizer "'+str(cb_params.optimizer) +
                      '" does not has the attribute "velocity".')

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
            if self.use_updater:
                self.kinetics = cb_params.kinetics.asnumpy().squeeze()
                self.temperature = cb_params.temperature.asnumpy().squeeze()
            cb_params = run_context.original_args()
            step = cb_params.cur_step
            time = cb_params.cur_time
            coordinate = cb_params.coordinate.asnumpy()
            self.h5md.write_position(step, time, coordinate)

            if self.use_pbc:
                if not self.const_volume:
                    pbc_box = cb_params.pbc_box.asnumpy()
                    self.h5md.write_box(step, time, pbc_box)
                if self.write_image:
                    image = self.system.calc_image().asnumpy()
                    self.h5md.write_image(step, time, image)

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        if self.count % self.save_freq == 0:
            cb_params = run_context.original_args()
            step = cb_params.cur_step
            time = cb_params.cur_time

            self.potential = cb_params.energy.asnumpy().squeeze()
            if self.use_updater:
                self.tot_energy = self.potential + self.kinetics
                if self.use_pbc:
                    self.pressure = cb_params.pressure.asnumpy().squeeze()
                    self.volume = cb_params.volume.asnumpy().squeeze()

            obs_values = [
                self.potential,
                self.kinetics,
                self.tot_energy,
                self.temperature,
                self.pressure,
                self.volume,
            ]

            self.h5md.write_observables(self.observables, step, time, obs_values)

            if self.write_velocity:
                velocity = cb_params.velocity[0].asnumpy()
                self.h5md.write_velocity(step, time, velocity)
            if self.write_force:
                force = cb_params.force.asnumpy()
                self.h5md.write_force(step, time, force)

        self.count += 1

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        #pylint: disable=unused-argument
        self.h5md.close()

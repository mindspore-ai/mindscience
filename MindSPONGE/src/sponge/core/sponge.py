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
Core engine of MindSPONGE
"""

import os
from typing import Union, List
import time
import datetime
from collections.abc import Iterable

from mindspore import nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.common import Tensor
from mindspore.nn.optim import Optimizer

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback, RunContext, _InternalCallbackParam, _CallbackManager
from mindspore.parallel._utils import _get_parallel_mode, _get_device_num, _get_global_rank, \
    _get_parameter_broadcast, _device_number_check
from mindspore.parallel._ps_context import _is_role_pserver
from mindspore.train.model import _StepSync, _transfer_tensor_to_tuple
from mindspore.dataset.engine.datasets import Dataset
from mindspore.train.dataset_helper import DatasetHelper
from mindspore.dataset.engine.datasets import _set_training_dataset

from .simulation import WithEnergyCell, WithForceCell
from .simulation import RunOneStepCell
from .analysis import AnalysisCell
from ..function import any_not_none, get_arguments
from ..potential import PotentialCell, ForceCell
from ..optimizer import Updater, UpdaterMD
from ..system.molecule import Molecule
from ..metrics import MetricCV, get_metrics


class Sponge():
    r"""
    Core engine of MindSPONGE for simulation and analysis.

    This Cell is the top-level wrapper for the three modules system (
    :class:`sponge.system.Molecule`),potential (:class:`sponge.potential.
    PotentialCell`) and
    optimizer (`mindspore.nn.Optimizer`) in MindSPONGE.

    There are three ways to wrap the modules:

    1)  Wraps `system`, `potential` and `optimizer` directly into :class:`sponge.core.Sponge`.

    .. code-block::

        from sponge import Sponge
        from sponge.system import Molecule
        from sponge.potential.forcefield import ForceField
        from sponge.optimizer import Updater
        system = Molecule(template='water.tip3p.yaml')
        potential = ForceField(system, parameters='SPCE')
        optimizer = Updater(system, controller=None, time_step=1e-3)
        md = Sponge(system, potential, optimizer)

    In this way ordinary simulations can be achieved

    2)  Wrap `system` and `potential` with :class:`sponge.core.WithEnergyCell` first,
    then wrap :class:`sponge.core.WithEnergyCell` and `optimizer`
    with :class:`sponge.core.Sponge`.

    .. code-block::

        from sponge import WithEnergyCell, Sponge
        from sponge.system import Molecule
        from sponge.potential.forcefield import ForceField
        from sponge.optimizer import Updater
        system = Molecule(template='water.tip3p.yaml')
        potential = ForceField(system, parameters='SPCE')
        optimizer = Updater(system, controller=None, time_step=1e-3)
        sys_with_ene = WithEnergyCell(system, potential)
        md = Sponge(sys_with_ene, optimizer=optimizer)

    In this case, the adjustment of the potential can be achieved by
    adjusting the :class:`sponge.core.WithEnergyCell`,
    for example by setting the `neighbour_list` and the `bias` in
    :class:`sponge.core.WithEnergyCell`.

    3)  Wrap `system` and `potential` with
    :class:`sponge.core.WithEnergyCell` first,
    then wrap :class:`sponge.core.WithEnergyCell` and `optimizer`
    with :class:`sponge.core.RunOneStepCell`, and finally pass the
    :class:`sponge.core.RunOneStepCell` into :class:`sponge.core.Sponge`.

    .. code-block::

        from sponge import WithEnergyCell, RunOneStepCell, Sponge
        from sponge.system import Molecule
        from sponge.potential.forcefield import ForceField
        from sponge.optimizer import Updater
        system = Molecule(template='water.tip3p.yaml')
        potential = ForceField(system, parameters='SPCE')
        optimizer = Updater(system, controller=None, time_step=1e-3)
        sys_with_ene = WithEnergyCell(system, potential)
        one_step = RunOneStepCell(sys_with_ene, optimizer=optimizer)
        md = Sponge(one_step)

    In this case, the adjustment of the force can be achieved by
    adjusting the :class:`sponge.core.RunOneStepCell`, for example by
    adding a `sponge.potential.ForceCell` to the :class:`sponge.core.RunOneStepCell`.

    For simulations:

    Simulation can be performed by executing the member function
    :func:`sponge.core.Sponge.run`.

    .. code-block::

        from sponge import Sponge
        from sponge.system import Molecule
        from sponge.potential.forcefield import ForceField
        from sponge.optimizer import Updater
        system = Molecule(template='water.tip3p.yaml')
        potential = ForceField(system, parameters='SPCE')
        optimizer = Updater(system, controller=None, time_step=1e-3)
        md = Sponge(system, potential, optimizer)
        md.run(100)

    For analysis:

    :class:`sponge.core.Sponge` can also analyse the simulation
    system by `metrics`. The `metrics` should be a dictionary of
    :class:`sponge.metrics.Metric` or :class:`sponge.colvar.Colvar`.
    The value of the `metrics` can be calculated by executing the
    member function :func:`sponge.core.Sponge.analyse`.

    .. code-block::

        from sponge import Sponge
        from sponge.colvar import Torsion
        from sponge import Protein
        from sponge.potential.forcefield import ForceField
        from sponge.optimizer import SteepestDescent
        # You can find alad.pdb file under MindSPONGE/tutorials/advanced/alad.pdb
        system = Protein(pdb='alad.pdb')
        potential = ForceField(system, 'AMBER.FF14SB')
        optimizer = SteepestDescent(system.trainable_params(), 1e-7)
        phi = Torsion([4, 6, 8, 14])
        psi = Torsion([6, 8, 14, 16])
        md = Sponge(system, potential, optimizer, metrics={'phi': phi, 'psi': psi})
        metrics = md.analyse()
        for k, v in metrics.items():
            print(k, v)

    Args:
        network (Union[Molecule, WithEnergyCell, RunOneStepCell]): Cell of the simulation system.
          Data type refers to
          :class:`sponge.system.Molecule`, :class:`sponge.core.WithEnergyCell` and :class:`sponge.core.RunOneStepCell`
        potential (:class:`sponge.potential.PotentialCell`, optional): Potential energy.
          Default: ``None``.
        optimizer (`mindspore.nn.Optimizer`, optional): Optimizer.
          Default: ``None``.
        metrics (dict, optional): A Dictionary of metrics for system analysis.
          The key type of the `dict` should be `str`, and the value type of
          the `dict` should be :class:`sponge.metrics.Metric` or
          :class:`sponge.colvar.Colvar`. Default: ``None``.
        analysis (:class:`sponge.core.AnalysisCell`, optional): Analysis network.
          Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 network: Union[Molecule, WithEnergyCell, RunOneStepCell],
                 potential: PotentialCell = None,
                 optimizer: Optimizer = None,
                 metrics: dict = None,
                 analysis: AnalysisCell = None,
                 **kwargs
                 ):
        self._kwargs = get_arguments(locals(), kwargs)

        self._parallel_mode = _get_parallel_mode()
        self._device_number = _get_device_num()
        self._global_rank = _get_global_rank()
        self._parameter_broadcast = _get_parameter_broadcast()
        self._create_time = int(time.time() * 1e9)

        self._force_function = None
        if optimizer is None:
            if potential is not None:
                raise ValueError('When optimizer is None, potential must also be None!')
            # network is RunOneStepCell: Sponge = RunOneStepCell
            self._simulation_network: RunOneStepCell = network
            self._system_with_energy: WithEnergyCell = self._simulation_network.system_with_energy
            self._system_with_force: WithForceCell = self._simulation_network.system_with_force
            self._optimizer: Optimizer = self._simulation_network.optimizer
            self._system: Molecule = self._simulation_network.system

            self._potential_function = None
            if self._system_with_energy is not None:
                self._potential_function: PotentialCell = self._system_with_energy.potential_function

            if self._system_with_force is not None:
                self._force_function: ForceCell = self._system_with_force.force_function
        else:
            self._system_with_force = None
            self._optimizer = optimizer
            if potential is None:
                # network is WithEnergyCell: Sponge = WithEnergyCell + optimizer
                self._system_with_energy: WithEnergyCell = network
                self._simulation_network = RunOneStepCell(
                    energy=self._system_with_energy, optimizer=self._optimizer)
                self._system: Molecule = self._system_with_energy.system
                self._potential_function: PotentialCell = self._system_with_energy.potential_function
            else:
                # network is system: Sponge = system + potential + optimizer
                self._system: Molecule = network
                self._potential_function: PotentialCell = potential
                self._system_with_energy = WithEnergyCell(self._system, self._potential_function)
                self._simulation_network = RunOneStepCell(
                    energy=self._system_with_energy, optimizer=self._optimizer)

        self._metrics = metrics

        self._num_energies = self._simulation_network.num_energies
        self._num_biases = self._simulation_network.num_biases
        self._energy_names = self._simulation_network.energy_names

        self._check_for_graph_cell()

        self.use_updater = False
        if isinstance(self._optimizer, Updater):
            self.use_updater = True

        if isinstance(self._simulation_network, RunOneStepCell):
            self._simulation_network.set_pbc_grad(self.use_updater)

        self.units = self._system.units

        lr = self._optimizer.learning_rate
        if self._optimizer.dynamic_lr:
            if self._optimizer.is_group_lr:
                lr = ()
                for learning_rate in self._optimizer.learning_rate:
                    current_dynamic_lr = learning_rate(0)
                    lr += (current_dynamic_lr,)
            else:
                lr = self._optimizer.learning_rate(0)
        self.time_step = lr.asnumpy()

        self.coordinate = self._system.coordinate
        self.pbc_box = self._system.pbc_box

        self.energy_neighbour_list = None
        if self._system_with_energy is not None:
            self.energy_neighbour_list = self._system_with_energy.neighbour_list

        self.force_neighbour_list = None
        if self._system_with_force is not None:
            self.force_neighbour_list = self._system_with_force.neighbour_list

        self.neighbour_list_pace = self._simulation_network.neighbour_list_pace

        self.energy_cutoff = self._simulation_network.energy_cutoff
        self.force_cutoff = self._simulation_network.force_cutoff

        self.update_nl = any_not_none([self.energy_cutoff, self.force_cutoff])

        self._analysis_network = analysis

        self._metric_fns = None
        self._metric_key = []
        self._metric_shape = []
        self._metric_units = []
        if metrics is not None:
            self._metric_fns = get_metrics(metrics)
            for k, v in self._metric_fns.items():
                self._metric_key.append(k)
                if isinstance(v, MetricCV):
                    self._metric_shape.append(v.shape)
                    self._metric_units.append(v.get_unit(self.units))
                else:
                    self._metric_shape.append(tuple())
                    self._metric_units.append(None)
            if analysis is None:
                self._analysis_network = AnalysisCell(self._system, self._potential_function,
                                                      self.energy_neighbour_list)

        self.sim_step = 0
        self.sim_time = 0.0

        self._potential = None
        self._force = None

        self._use_bias = False

        self.reduce_mean = ops.ReduceMean()

    @property
    def energy_names(self) -> List[str]:
        r"""
        Names of energy terms

        Returns:
            list of str, names of energy terms

        """
        return self._energy_names

    @property
    def num_energies(self) -> int:
        r"""
        Number of energy terms

        Returns:
            int, number of energy terms

        """
        return self._num_energies

    @property
    def num_biases(self) -> int:
        r"""
        Number of bias potential energies V

        Returns:
            int, number of bias potential energies

        """
        return self._num_biases

    def recompile(self):
        r"""
        Recompile the simulation network
        """
        self._simulation_network.compile_cache.clear()
        return self

    def update_neighbour_list(self):
        r"""
        Update neighbour list
        """
        self._simulation_network.update_neighbour_list()
        return self

    def update_bias(self, step: int):
        r"""
        Update bias potential.

        Args:
            step (int): step of the simulation.
        """
        self._simulation_network.update_bias(step)

    def update_wrapper(self, step: int):
        r"""
        Update energy wrapper.

        Args:
            step (int): step of the simulation.
        """
        self._simulation_network.update_wrapper(step)

    def update_modifier(self, step: int):
        r"""
        Update force modifier

        Args:
            step (int): step of the simulation.
        """
        self._simulation_network.update_modifier(step)

    def change_optimizer(self, optimizer: Optimizer):
        r"""
        Change optimizer.

        Args:
            optimizer (:class:`mindsponge.optimizer.Optimizer`): Optimizer will be used.
        """
        if self._optimizer is None:
            raise ValueError('Cannot change the optimizer, because the initial optimizer is None '
                             'or the network is not a RunOneStepCell type.')

        self._optimizer = optimizer

        if isinstance(self._optimizer, Updater):
            self.use_updater = True
        else:
            self.use_updater = False

        self._simulation_network = RunOneStepCell(
            energy=self._system_with_energy, optimizer=self._optimizer)
        self._simulation_network.set_pbc_grad(self.use_updater)

        lr = self._optimizer.learning_rate
        if self._optimizer.dynamic_lr:
            if self._optimizer.is_group_lr:
                lr = ()
                for learning_rate in self._optimizer.learning_rate:
                    current_dynamic_lr = learning_rate(0)
                    lr += (current_dynamic_lr,)
            else:
                lr = self._optimizer.learning_rate(0)
        self.time_step = lr.asnumpy()

        return self

    def change_potential(self, potential: PotentialCell):
        r"""
        Change potential energy.

        Args:
            potential (:class:`sponge.potential.PotentialCell`): Potential energy will be used.
        """
        if self._potential_function is None:
            raise ValueError('Cannot change the potential, because the initial potential is None '
                             'or the network is not a WithEnergyCell type.')
        if self._optimizer is None:
            raise ValueError('Cannot change the potential, because the initial optimizer is None '
                             'or the network is not a RunOneStepCell type.')

        self._potential_function = potential
        self._system_with_energy = WithEnergyCell(self._system, self._potential_function)
        self._simulation_network = RunOneStepCell(
            energy=self._system_with_energy, optimizer=self._optimizer)
        self._simulation_network.set_pbc_grad(self.use_updater)

        return self

    def calc_energy(self) -> Tensor:
        r"""
        Calculate the total potential energy (potential energy and bias
        potential) of the simulation system.

        Returns:
            energy (Tensor), Tensor of shape :math:`(B, 1)`.
            Here :math:`B` is the batch size, i.e. the number of walkers of the simulation.
            Data type is float. Total potential energy.

        """
        if self._system_with_energy is None:
            return None
        return self._system_with_energy()

    def calc_energies(self) -> Tensor:
        r"""
        Calculate the energy terms of the potential energy.

        Returns:
            energies (Tensor), Tensor of shape :math:`(B, U)`.
              Energy terms.
              Here :math:`B` is the batch size, i.e. the number of walkers of
              the simulation, `U` is the number of potential energy terms.
              Data type is float.
        """
        if self._system_with_energy is None:
            return None
        return self._system_with_energy.calc_energies()

    def calc_biases(self) -> Tensor:
        r"""
        Calculate the bias potential terms.

        Returns:
            biases (Tensor), Tensor of shape :math:`(B, V)`.
              Bias terms.
              Here :math:`B` is the batch size,
              :math:`V` is the number of bias potential terms.
              Data type is float.
        """
        if self._system_with_energy is None:
            return None
        return self._system_with_energy.calc_biases()

    def run(self,
            steps: int,
            callbacks: Union[Callback, List[Callback]] = None,
            dataset: Dataset = None,
            show_time: bool = True,
            ):
        r"""
        Simulation API.

        Args:
            steps (int): Simulation steps.
            callbacks (Union[`mindspore.train.Callback`, List[`mindspore.train.Callback`]]):
              Callback function(s) to obtain the information of the system
              during the simulation. Default: ``None``.
            dataset (Dataset): Dataset used at simulation process. Default: ``None``.
            show_time (bool): Whether to show the time of the simulation. Default: ``True``.

        Examples:
            >>> from sponge import Sponge
            >>> from sponge.system import Molecule
            >>> from sponge.potential.forcefield import ForceField
            >>> from sponge.optimizer import Updater
            >>> from sponge.callback import RunInfo
            >>> system = Molecule(template='water.tip3p.yaml')
            >>> potential = ForceField(system, parameters='SPCE')
            >>> optimizer = Updater(system, controller=None, time_step=1e-3)
            >>> md = Sponge(system, potential, optimizer)
            >>> md.run(100, callbacks=[RunInfo(10)])
        """
        if self.neighbour_list_pace == 0 or steps < self.neighbour_list_pace:
            epoch = 1
            cycle_steps = steps
            rest_steps = 0
        else:
            epoch = steps // self.neighbour_list_pace
            cycle_steps = self.neighbour_list_pace
            rest_steps = steps - epoch * cycle_steps

        cb_params = _InternalCallbackParam()
        cb_params.sim_network = self._simulation_network
        cb_params.analyse = None
        cb_params.metrics = None
        cb_params.metrics_shape = None
        cb_params.metrics_units = None
        if self._analysis_network is not None:
            cb_params.analyse = self.analyse
            cb_params.metrics = self._metric_key
            cb_params.metrics_shape = self._metric_shape
            cb_params.metrics_units = self._metric_units

        cb_params.with_energy = self._system_with_energy is not None
        cb_params.with_force = self._system_with_force is not None

        cb_params.num_steps = steps
        cb_params.time_step = self.time_step
        cb_params.num_epoch = epoch
        cb_params.cycle_steps = cycle_steps
        cb_params.rest_steps = rest_steps

        cb_params.mode = "simulation"
        cb_params.sim_network = self._simulation_network
        cb_params.system = self._system
        cb_params.potential_network = self._potential_function
        cb_params.optimizer = self._optimizer
        cb_params.parallel_mode = self._parallel_mode
        cb_params.device_number = self._device_number
        cb_params.simulation_dataset = dataset
        cb_params.list_callback = self._transform_callbacks(callbacks)
        if context.get_context("mode") == context.PYNATIVE_MODE:
            cb_params.list_callback.insert(0, _StepSync())
            callbacks = cb_params.list_callback

        cb_params.coordinate = self.coordinate
        cb_params.pbc_box = self.pbc_box
        cb_params.energy = 0
        cb_params.force = 0
        cb_params.potential = 0
        cb_params.energies = 0
        cb_params.num_energies = self._num_energies
        cb_params.energy_names = self._energy_names

        cb_params.num_biases = self._num_biases

        if self._num_biases > 0:
            cb_params.bias = 0
            cb_params.biases = 0
            cb_params.bias_names = self._simulation_network.bias_names
        else:
            cb_params.bias = None
            cb_params.biases = None
            cb_params.bias_names = None

        cb_params.volume = self._system.get_volume()
        if self.use_updater:
            self._optimizer.set_step(0)
            cb_params.velocity = self._optimizer.velocity
            kinetics = F.reduce_sum(self._optimizer.kinetics, -1)
            cb_params.kinetics = kinetics
            cb_params.temperature = self._optimizer.temperature
            pressure = self._optimizer.pressure
            if pressure is not None:
                # (B) <- (B,D)
                pressure = self.reduce_mean(pressure, -1)
            cb_params.pressure = pressure

            cb_params.thermostat = None
            cb_params.barostat = None
            cb_params.constraint = None
            if isinstance(self._optimizer, UpdaterMD):
                cb_params.thermostat = self._optimizer.thermostat
                cb_params.barostat = self._optimizer.barostat
                cb_params.constraint = self._optimizer.constraint

        beg_time = datetime.datetime.now()
        if show_time:
            print('[MindSPONGE] Started simulation at', beg_time.strftime('%Y-%m-%d %H:%M:%S'))
        # build callback list
        with _CallbackManager(callbacks) as list_callback:
            self._simulation_process(epoch, cycle_steps, rest_steps, list_callback, cb_params)
        end_time = datetime.datetime.now()

        if show_time:
            print('[MindSPONGE] Finished simulation at', end_time.strftime('%Y-%m-%d %H:%M:%S'))
            used_time = end_time - beg_time
            d = used_time.days
            s = used_time.seconds
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            if d >= 1:
                print('[MindSPONGE] Simulation time: %d days, %d hours, %d minutes and %d seconds.' % (d, h, m, s))
            elif h >= 1:
                print('[MindSPONGE] Simulation time: %d hours %d minutes %d seconds.' % (h, m, s))
            elif m >= 1:
                s += used_time.microseconds / 1e6
                print('[MindSPONGE] Simulation time: %d minutes %1.1f seconds.' % (m, s))
            else:
                s += used_time.microseconds / 1e6
                print('[MindSPONGE] Simulation time: %1.2f seconds.' % s)
            print('-'*80)
        return self

    def calc_potential(self) -> Tensor:
        r"""
        Calculate and return the potential energy

        Returns:
            energy, Tensor of shape :math:`(B, 1)`.
              Total potential energy.
              Here `B` is the batch size, i.e. the number of walkers of the
              simulation. Data type is float.
        """
        if self._system_with_energy is None:
            return 0
        return self._system_with_energy()

    def get_energies(self) -> Tensor:
        r"""
        Get the potential energy terms.

        Returns:
            energies, Tensor of shape :math:`(B, U)`.
              Energy terms.
              Here `B` is the batch size, i.e. the number of walkers of the
              simulation, :math:`U` is the number of potential energy terms.
              Data type is float.
        """
        return self._simulation_network.energies

    def get_biases(self) -> Tensor:
        r"""
        Get the bias potential energies.

        Returns:
            biases, Tensor of shape :math:`(B, V)`.
              Bias terms.
              Here :math:`B` is the batch size, i.e. the number of walkers of the
              simulation, :math:`V` is the number of bias potential terms.
              Data type is float.
        """
        return self._simulation_network.biases

    def get_bias(self) -> Tensor:
        r"""
        Get the total bias potential energy.

        Returns:
            bias, Tensor of shape :math:`(B, 1)`.
                Here :math:`B` is the batch size, i.e. the number of walkers of the simulation.
                Data type is float.
        """
        return self._simulation_network.bias

    def analyse(self,
                dataset: Dataset = None,
                callbacks: Union[Callback, List[Callback]] = None,
                ):
        """
        Analysis API.

        Note:
            To use this API, the `metrics` must be set at :class:`sponge.core.Sponge` initialization.

        Args:
            dataset (Dataset): Dataset of simulation to be analysed. Default: ``None``.
            callbacks (Union[`mindspore.train.Callback`, List[`mindspore.train.Callback`]]):
              List of callback objects which should be executed
              while training. Default: ``None``.

        Returns:
            Dict, the key is the metric name defined by users and the value is the metrics value for
            the model in the test mode.

        Examples:
            >>> from mindsponge.colvar import Torsion
            >>> from mindsponge.colvar import Torsion
            >>> phi = Torsion([4, 6, 8, 14])
            >>> psi = Torsion([6, 8, 14, 16])
            >>> md = Sponge(system, potential, optimizer, metrics={'phi': phi, 'psi': psi})
            >>> metrics = md.analyse()
            >>> for k, v in metrics.items():
            >>>     print(k, v)
            phi [[3.1415927]]
            psi [[3.1415927]]

        """

        _device_number_check(self._parallel_mode, self._device_number)
        if not self._metric_fns:
            raise ValueError("The Sponge argument 'metrics' can not be None or empty, "
                             "you should set the argument 'metrics' for Sponge.")

        cb_params = _InternalCallbackParam()
        cb_params.mode = "analyse"
        cb_params.analysis_network = self._analysis_network
        cb_params.cur_step_num = 0
        if dataset is not None:
            cb_params.analysis_dataset = dataset
            cb_params.batch_num = dataset.get_dataset_size()

        cb_params.list_callback = self._transform_callbacks(callbacks)

        self._clear_metrics()

        with _CallbackManager(callbacks) as list_callback:
            return self._analysis_process(dataset, list_callback, cb_params)

    def _check_for_graph_cell(self):
        r"""
        Check for graph cell
        """
        if not isinstance(self._system, nn.GraphCell):
            return

        if self._potential_function is not None or self._optimizer is not None:
            raise ValueError("For 'Model', 'loss_fn' and 'optimizer' should be None when network is a GraphCell, "
                             "but got 'loss_fn': {}, 'optimizer': {}.".format(self._potential_function,
                                                                              self._optimizer))

    @staticmethod
    def _transform_callbacks(callbacks: Callback):
        r"""
        Transform callbacks to a list.

        Args:
            callbacks (`mindspore.train.Callback`): Callback or iterable of Callback's.

        Returns:
            List, a list of callbacks.
        """
        if callbacks is None:
            return []

        if isinstance(callbacks, Iterable):
            return list(callbacks)

        return [callbacks]

    def _simulation_process(self,
                            epoch: int,
                            cycle_steps: int,
                            rest_steps: int,
                            list_callback: Callback = None,
                            cb_params: _InternalCallbackParam = None
                            ):
        r"""
        Training process. The data would be passed to network directly.

        Args:
            epoch (int): Total number of iterations on the data.
            cycle_steps (int): Number of steps in each epoch.
            rest_steps (int): Number of steps in the last epoch.
            list_callback (`minspore.train.callback.Callback`): Executor of callback list. Default: ``None``.
            cb_params (_InternalCallbackParam): Callback parameters. Default: ``None``.
        """
        self._exec_preprocess(True)

        self.sim_step = 0
        self.sim_time = 0.0
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False

        for i in range(epoch):
            cb_params.cur_epoch = i
            self.update_neighbour_list()
            should_stop = self._run_one_epoch(cycle_steps, list_callback, cb_params, run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        if rest_steps > 0:
            self.update_neighbour_list()
            self._run_one_epoch(rest_steps, list_callback,
                                cb_params, run_context)

        list_callback.end(run_context)

    def _run_one_epoch(self,
                       cycles: int,
                       list_callback: Callback,
                       cb_params: _InternalCallbackParam,
                       run_context: RunContext
                       ):
        r"""
        Run one epoch simulation

        Args:
            cycles (int): Number of steps in each epoch.
            list_callback (`mindspore.train.Callback`):
              Executor of callback list. Default: ``None``.
            cb_params (_InternalCallbackParam): Callback parameters. Default: ``None``.
            run_context (`mindspore.train.callback.RunContext`): Context of the current run.

        Returns:
            bool, whether to stop the training.
        """
        should_stop = False
        list_callback.epoch_begin(run_context)
        for _ in range(cycles):

            cb_params.cur_step = self.sim_step
            cb_params.cur_time = self.sim_time
            list_callback.step_begin(run_context)

            cb_params.volume = self._system.get_volume()

            self._potential, self._force = self._simulation_network()
            self.update_bias(self.sim_step)
            self.update_wrapper(self.sim_step)
            self.update_modifier(self.sim_step)

            cb_params.potential = self._potential
            cb_params.force = self._force

            cb_params.energies = self.get_energies()
            if self._num_biases > 0:
                cb_params.bias = self.get_bias()
                cb_params.biases = self.get_biases()

            if self.use_updater:
                cb_params.velocity = self._optimizer.velocity
                # (B) <- (B,D)
                kinetics = F.reduce_sum(self._optimizer.kinetics, -1)
                cb_params.kinetics = kinetics
                cb_params.temperature = self._optimizer.temperature
                pressure = self._optimizer.pressure
                if pressure is not None:
                    # (B) <- (B,D)
                    pressure = self.reduce_mean(pressure, -1)
                cb_params.pressure = pressure

            self.sim_step += 1
            self.sim_time += self.time_step

            list_callback.step_end(run_context)

            #pylint: disable = protected-access
            if _is_role_pserver():
                os._exit(0)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        # if param is cache enable, flush data from cache to host before epoch end
        self._flush_from_cache(cb_params)

        list_callback.epoch_end(run_context)
        return should_stop

    def _analysis_process(self,
                          dataset: Dataset = None,
                          list_callback: Callback = None,
                          cb_params: _InternalCallbackParam = None
                          ):
        r"""
        Evaluation. The data would be passed to network directly.

        Args:
            dataset (Dataset): Dataset to evaluate the model.
            list_callback (:class: `mindspore.train.Callback`):
              Executor of callback list. Default: ``None``.
            cb_params (_InternalCallbackParam): Callback parameters. Default: ``None``.

        Returns:
            Dict, which returns the metrics values for the model in the test mode.
        """
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        dataset_helper, _ = self._exec_preprocess(False)
        list_callback.epoch_begin(run_context)

        if dataset is None:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)

            inputs = (
                self.coordinate,
                self.pbc_box,
                self._potential,
                self._force,
                self.get_energies(),
                self.get_bias(),
                self.get_biases(),
            )

            outputs = self._analysis_network(*inputs)

            if self.pbc_box is None:
                outputs = (outputs[0], None) + outputs[2:]

            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)
        else:
            for next_element in dataset_helper:
                cb_params.cur_step_num += 1
                list_callback.step_begin(run_context)
                next_element = _transfer_tensor_to_tuple(next_element)
                outputs = self._analysis_network(*next_element)
                cb_params.net_outputs = outputs
                list_callback.step_end(run_context)
                self._update_metrics(outputs)

        list_callback.epoch_end(run_context)
        if dataset is not None:
            dataset.reset()
        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)
        return metrics

    def _clear_metrics(self):
        r"""
        Clear metrics local values.
        """
        for metric in self._metric_fns.values():
            metric.clear()

    def _update_metrics(self, outputs):
        r"""
        Update metrics local values.

        Args:
            outputs (Tensor): Outputs of the analysis network.
        """
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        if not isinstance(outputs, tuple):
            raise ValueError(
                f"The argument 'outputs' should be tuple, but got {type(outputs)}.")

        for metric in self._metric_fns.values():
            metric.update(*outputs)

    def _get_metrics(self):
        r"""
        Get metrics local values.

        Returns:
            Dict, which returns the metrics values.
        """
        metrics = dict()
        for key, value in self._metric_fns.items():
            metrics[key] = value.eval()
        return metrics

    def _exec_preprocess(self, is_run, dataset=None, dataset_helper=None):
        r"""
        Initializes dataset.

        Args:
            is_run (bool): Whether to run the simulation.
            dataset (Dataset): Dataset used at simulation process. Default: ``None``.
            dataset_helper (DatasetHelper): Dataset helper. Default: ``None``.
        """
        if is_run:
            network = self._simulation_network
            phase = 'simulation'
        else:
            network = self._analysis_network
            phase = 'analysis'

        if dataset is not None and dataset_helper is None:
            dataset_helper = DatasetHelper(dataset, False)

        if is_run:
            _set_training_dataset(dataset_helper)  # pylint: disable=W0212

        network.set_train(is_run)
        network.phase = phase

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()

        return dataset_helper, network

    def _flush_from_cache(self, cb_params):
        r"""
        Flush cache data to host if tensor is cache enable.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters."""
        params = cb_params.sim_network.get_parameters()
        for param in params:
            if param.cache_enable:
                Tensor(param).flush_from_cache()

    @property
    def create_time(self):
        r"""
        Create time of the Sponge instance.

        Returns:
            int, create time of the Sponge instance.
        """
        return self._create_time

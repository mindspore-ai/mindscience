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
"""Dynamic ITS"""

from typing import Tuple
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F
from mindspore.numpy.utils_const import _raise_runtime_error

from .wrapper import EnergyWrapper
from ... import function as func
from ...function import get_tensor, get_integer


class DynamicITS(EnergyWrapper):
    r"""Energy wrapper for dynammic integrated tempering sampling.

    Math:

    Args:

        sim_temp (float):       Simulation temperature.

        temp_min (float):       Minimum temperature for integration.
                                Only used when `temperature` is None.
                                Default: None

        temp_max (float):       Minimum temperature for integration.
                                Only used when `temperature` is None.
                                Default: None

        temp_bin (int):         Number of temperatures for integrationintergration.
                                Only used when `temperature` is None.
                                Default: None

        unlinear_temp (bool)    Whether to generate unlinear integration temperatures
                                Default: False

        temperatures (Tensor):  Temperatures for integration.
                                The shape of tensor is `(B, T)`, the data type is float.
                                Default: None

        update_pace (int):      Freuency for updating ITS. Default: 100

        num_walker (int):       Number of multiple walkers.
                                Cannot be None when `share_parameter` is False. Default: None

        share_parameter (bool): Whether to share ITS parameters for all walkers. Default: True

        energy_shift (float):   Initial shift value for potential energy. Default: 0

        energy_ratio (Tensor):  Ratio to select the potential energies to be enhanced
                                The shape of tensor is `(B, U)`. The data type is float.
                                Default: 1

        bias_ratio (Tensor):    Ratio to select the bias potential energies to be enhanced.
                                The shape of tensor is `(B, V)`. The data type is float.
                                Default: 1

        ratio_exponent (float): Exponent for calculating the iteration weights of
                                the neighbouring temperatures. Default: 0.5

        step_weight (float):    Weight for iteration step in calculating the iteration weights.
                                Default: 0

        weight_bias (float):    Bias value for the iteration of weighting factors.
                                Defatul: 0

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: None

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: None

    Supported Platforms:

        ``Ascend``

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation.

        T:  Number of integration temperatures.

        U:  Dimension of potential energy.

        V:  Dimension of bias potential.

    """

    def __init__(self,
                 num_walker: int,
                 ref_temp: float,
                 max_temp: float,
                 temp_bin: int = 50,
                 update_pace: int = 0,
                 energy_shift: float = 0,
                 threshold: float = 0.1,
                 temp_increase: float = 0.1,
                 memory_time: float = 0.8,
                 peshift_scale: float = 1.5,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):

        super().__init__(
            update_pace=update_pace,
            length_unit=length_unit,
            energy_unit=energy_unit,
            )

        self.num_walker = get_integer(num_walker)
        self.temp_bin = get_integer(temp_bin)

        self.boltzmann = get_tensor(self.units.boltzmann, ms.float32)

        # (1)
        self.ref_temp = get_tensor(ref_temp, ms.float32).reshape(())

        self.ref_kbt = self.boltzmann * self.ref_temp
        self.ref_beta = msnp.reciprocal(self.ref_kbt)

        self.max_temp = get_tensor(max_temp, ms.float32).reshape(())
        self.sim_temp = Parameter(self.ref_temp, name='sim_temp', requires_grad=False)

        # (T)
        temperatures = msnp.full((self.temp_bin,), self.ref_temp, ms.float32)
        temperatures[-1] = self.max_temp
        self.temperatures = Parameter(temperatures, name='temperatures', requires_grad=False)

        mask = F.fill(ms.bool_, (self.temp_bin,), False)
        mask[-1] = True
        self.temp_mask = Parameter(mask, name='temp_mask', requires_grad=False)

        # (T)
        # self.weight_factors: \log{n_k}
        self.weight_factors = Parameter(F.zeros_like(self.beta_array),
                                        name='weight_factors', requires_grad=False)

        # (B, 1)
        self.energy = Parameter(msnp.zeros((self.num_walker, 1), ms.float32), name='energy', requires_grad=False)
        self.eff_energy = Parameter(msnp.zeros((self.num_walker, 1), ms.float32),
                                    name='eff_energy', requires_grad=False)

        # (B, 1)
        self.minimum_energy = Parameter(Tensor(0, ms.float32), name='minimum_energy', requires_grad=False)

        energy_shift = get_tensor(energy_shift, ms.float32).reshape(())
        self.energy_shift = Parameter(energy_shift, name='energy_shift', requires_grad=False)

        max_index = Tensor(self.temp_bin-1, ms.int32)
        # index of current minimum temperature
        self.min_index = Parameter(max_index, name='min_index', requires_grad=False)
        # index of current maximum temperature
        self.max_index = Parameter(max_index, name='max_index', requires_grad=False)

        self.step = Parameter(Tensor(0, ms.int32), name='iteration_step', requires_grad=False)

        self.threshold = threshold
        self.memory_time = memory_time
        self.temp_increase = temp_increase
        self.peshift_scale = peshift_scale

    @property
    def sim_kbt(self) -> Tensor:
        return self.get_sim_kbt()

    @property
    def sim_beta(self) -> Tensor:
        return self.get_sim_beta()

    @property
    def kbt_array(self) -> Tensor:
        return self.get_kbt_array()

    @property
    def beta_array(self) -> Tensor:
        return self.get_beta_array()

    def get_kbt(self, temperature: Tensor) -> Tensor:
        return self.boltzmann * temperature

    def get_beta(self, temperature: Tensor) -> Tensor:
        return msnp.reciprocal(self.boltzmann * temperature)

    def get_sim_kbt(self) -> Tensor:
        return self.get_kbt(self.sim_temp)

    def get_sim_beta(self) -> Tensor:
        return self.get_beta(self.sim_temp)

    def get_kbt_array(self) -> Tensor:
        return self.get_kbt(self.temperatures)

    def get_beta_array(self) -> Tensor:
        return self.get_beta(self.temperatures)

    def set_sim_temp(self, temperature: Tensor) -> Tensor:
        sim_temp = get_tensor(temperature).reshape(())
        return F.assign(self.sim_temp, sim_temp)

    def change_energy_shift(self, peshift: Tensor) -> Tensor:
        r"""change the the shift value for potential energy

        Arg:
            peshift (Tensor):   Tensor of shape `(B, 1)`. Data type is float.
                                Energy shift.

        """
        # (1, 1) or (B, 1)
        fb0 = self.weight_factors[:, [0]] + self.ref_beta * peshift
        # (1, T) = (1, T) * (1, 1) - (1, 1) or (B, T) = (B, T) * (B, 1) - (B, 1)
        fb_add = self.get_beta_array() * peshift - fb0
        peshift = F.depend(peshift, F.assign_add(self.weight_factors, fb_add))
        return F.assign(self.energy_shift, peshift)

    def update(self):
        """update ITS"""
        min_energy = self.minimum_energy

        if min_energy + self.energy_shift < 0:
            self.change_energy_shift(-min_energy * self.peshift_scale)

        # (1, B) < (B, 1)
        energy = F.reshape(self.energy, (1, -1))
        eff_energy = F.reshape(self.eff_energy, (1, -1))

        # () * (1, B)
        beta_ene_eff = self.get_sim_beta() * eff_energy

        # (T, B) = (T, 1) * (1, B)
        betak_ene = F.reshape(self.get_beta_array(), (-1, 1)) * (energy + self.energy_shift)

        # (T) <- (T, B)
        weight_factors = F.logsumexp(beta_ene_eff - betak_ene, 1)
        # (T,):
        weight_factors -= weight_factors[0]

        wt_ema = self.memory_time * self.weight_factors + (1 - self.memory_time) * weight_factors

        index = self.identity(self.min_index)
        if index < self.temp_bin:
            wt_ema[index] = weight_factors[index]

        return F.assign(self.weight_factors, wt_ema)

    def add_low_temperature(self) -> Tensor:
        r"""add lower temperature"""
        index = self.identity(self.min_index)

        if index == 0:
            return self.identity(self.ref_temp)

        def _calc_kld(temp_old, temp_new, energy, size):
            beta_old = self.get_beta(temp_old)
            beta_new = self.get_beta(temp_new)

            exponent = (beta_old - beta_new) * energy
            exponent -= F.amax(exponent)
            wt = msnp.clip(F.exp(exponent), 1e-8, 1.)
            wt = wt / F.reduce_sum(wt)
            kld = F.reduce_sum(wt * F.log(size * wt), 0)
            return kld

        def _get_lower_temp(temperature, energy):
            factor = self.ref_temp / temperature
            increment = msnp.minimum(self.temp_increase, (1. - factor))
            temp_new = self.ref_temp / (factor + increment)
            kld = _calc_kld(temperature, temp_new, energy, self.num_walker)

            threshold = self.threshold
            if factor < 0.5:
                threshold *= 0.1

            while kld > threshold:
                increment *= 0.5
                temp_new = self.ref_temp / (factor + increment)
                kld = _calc_kld(temperature, temp_new, energy, self.num_walker)

            return temp_new

        last_temp = self.temperatures[index]
        new_temp = _get_lower_temp(last_temp, self.energy)
        index -= 1

        if new_temp <= self.ref_temp:
            new_temp = self.ref_temp
            index = 0
        elif index == 0:
            _raise_runtime_error(f'The minimum temperature added ({new_temp}) cannot reach '
                                 f'the reference temperature ({self.ref_temp}), '
                                 f'please increase the number of "temp_bin" or "threshold". ')

        self.temperatures[index] = new_temp
        self.temp_mask[index] = True

        new_temp = F.depend(new_temp, F.assign(self.min_index, index))
        new_temp = F.depend(new_temp, self.temperatures)
        new_temp = F.depend(new_temp, self.temp_mask)

        return new_temp

    def remove_high_temperature(self) -> Tensor:
        r"""remove the highest temperature"""
        index = self.identity(self.max_index)
        if index == 0:
            return self.identity(self.ref_temp)

        max_temp = self.temperatures[index]

        if self.temp_mask[index]:
            self.temperatures[index] = self.ref_temp
            self.temp_mask[index] = False
            max_temp = F.depend(max_temp, self.temperatures)
            max_temp = F.depend(max_temp, self.temp_mask)
            index -= 1
        else:
            index = 0

        max_temp = F.depend(max_temp, F.assign(self.max_index, index))

        return max_temp

    def construct(self, potentials: Tensor, biases: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""merge the potential energies and bias potential energies.

        Args:
            potentials (Tensor):    Tensor of shape `(B, U)`. Data type is float.
                                    Potential energies.
            biases (Tensor):        The shape of tensor is `(B, V)`. The data type is float.
                                    Bias potential energies. Default: None

        Return:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total energy (potential energy and bias energy).
            bias (Tensor):      Tensor of shape `(B, 1)`. Data type is float.
                                Total bias potential used for reweighting calculation.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            U:  Dimension of potential energy.
            V:  Dimension of bias potential.
        """

        # (B, 1) <- (B, U)
        potential = func.keepdims_sum(potentials, -1)

        bias = 0
        if biases is not None:
            # (B, 1) <- (B, V)
            bias = func.keepdims_sum(biases, -1)

        # (B, 1):
        # E(R) = U_{select}(R) + V_{select}(R)
        energy = potential + bias

        min_energy = F.reduce_min(F.stop_gradient(energy))
        min_energy = F.select(min_energy < self.minimum_energy, min_energy, self.minimum_energy)

        energy += self.energy_shift
        energy = F.depend(energy, F.assign(self.energy, energy))
        energy = F.depend(energy, F.assign(self.minimum_energy, min_energy))

        # (B, T) - (B, T) * (B, 1)
        # \log {\left [ n_k e ^ {-\beta_k U(R)} \right ] }
        gf = self.weight_factors - self.get_beta_array() * energy
        gf = msnp.where(self.temp_mask, gf, -5e4)

        # (B, 1) <- (B, T)
        # \log{\sum_k^N {n_k e ^ {-\beta_k U(R)}}}
        gfsum = F.logsumexp(gf, -1, True)
        # (B, 1) * (B, 1)
        # U_{eff}(R) = -\frac{1}{\beta_0} \log{\sum_k^N {n_k e ^ {-\beta_k U(R)}}}
        eff_energy = -self.get_sim_kbt() * gfsum

        eff_energy = F.depend(eff_energy, F.assign(self.eff_energy, eff_energy))

        # (B, 1) - (B, 1) - (B, 1)
        bias = eff_energy - potential - self.energy_shift

        return eff_energy, bias

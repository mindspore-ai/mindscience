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
"""Integrated tempering sampling (ITS) for atomic force"""

from typing import Tuple
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .modifier import ForceModifier
from ... import function as func
from ...function import any_none, get_ms_array, get_integer


class ModiferITS(ForceModifier):
    r"""Force modifier for integrated tempering sampling (ITS).
        WARNING: CANNOT be used with the ITS energy wrapper (mindsponge.sampling.wrapper.ITS)!

    References:

        Gao, Y. Q.
        An Integrate-over-Temperature Approach for Enhanced Sampling [J].
        The Journal of Chemical Physics, 2008, 128(6): 064105.

    Math:

    .. math::

        U_{eff}(R) = -\frac{1}{\beta_0} \log{\sum_k ^ N {n_k e ^ {-\beta_k U(R)}}}

    Args:
        sim_temp (float):       Simulation temperature.

        temp_min (float):       Minimum temperature for integration.
                                Only used when `temperature` is None.
                                Default: ``None``.

        temp_max (float):       Minimum temperature for integration.
                                Only used when `temperature` is None.
                                Default: ``None``.

        temp_bin (int):         Number of temperatures for integration.
                                Only used when `temperature` is None.
                                Default: ``None``.

        unlinear_temp (bool)    Whether to generate unlinear integration temperatures
                                Default: ``False``.

        temperatures (Tensor):  Temperatures for integration.
                                The shape of tensor is `(B, T)`, the data type is float.
                                Default: ``None``.

        update_pace (int):      Freuency for updating ITS. Default: 100

        multi_walkers (bool):   Whether to use multiple-walkers ITS. Default: ``True``.

        num_walker (int):       Number of simulation walkers.
                                Cannot be None when `multi_walkers` is False. Default: ``None``.

        energy_shift (float):   Initial shift value for potential energy. Default: 0

        ratio_exponent (float): Exponent for calculating the iteration weights of
                                the neighbouring temperatures. Default: 0.5

        step_weight (float):    Weight for iteration step in calculating the iteration weights.
                                Default: 0

        weight_bias (float):    Bias value for the iteration of weighting factors.
                                Defatul: 0

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: ``None``.

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:

        B:  Batchsize, i.e. number of walkers in simulation.

        T:  Number of integration temperatures.

        U:  Dimension of potential energy.

        V:  Dimension of bias potential.

    """

    def __init__(self,
                 sim_temp: float,
                 temp_min: float = None,
                 temp_max: float = None,
                 temp_bin: int = None,
                 unlinear_temp: bool = False,
                 temperatures: Tensor = None,
                 update_pace: int = 100,
                 multi_walkers: bool = True,
                 num_walker: int = None,
                 energy_shift: float = 0,
                 ratio_exponent: float = 0.5,
                 step_weight: float = 0,
                 weight_bias: float = 0,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):

        super().__init__(
            update_pace=update_pace,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )

        self.multi_walkers = multi_walkers
        if self.multi_walkers:
            self.num_walker = 1
        else:
            if num_walker is None:
                raise ValueError(
                    'num_walkers must be given when multi_walkers is False!')
            self.num_walker = get_integer(num_walker)

        # (B)
        self.sim_temp = self._check_temp(sim_temp, 'sim_temp')

        if temperatures is None:
            if any_none([temp_min, temp_max, temp_bin]):
                raise ValueError('When temperatures is None, '
                                 'temp_min, temp_max, temp_bin cannot be None.')

            # T'
            temp_bin = get_integer(temp_bin)
            # T = T' + 1
            self.temp_bin = temp_bin + 1
            # (B)
            self.temp_min = self._check_temp(temp_min, 'temp_min')
            self.temp_max = self._check_temp(temp_max, 'temp_max')

            # (B, 1) <- (B)
            temp_min = F.expand_dims(self.temp_min, -1)
            if unlinear_temp:
                # (B)
                temp_ratio = msnp.exp(
                    msnp.log(self.temp_max / self.temp_min) / temp_bin)
                # (B, 1) <- (B)
                temp_ratio = F.expand_dims(temp_ratio, -1)

                self.temperatures = temp_min * \
                    msnp.power(temp_ratio, msnp.arange(self.temp_bin))
            else:
                # (B)
                temp_space = (self.temp_max - self.temp_min) / temp_bin
                # (B, 1) <- (B)
                temp_space = F.expand_dims(temp_space, -1)

                # (B, T) = (B, 1) + (T) * (B, 1)
                self.temperatures = temp_min + \
                    msnp.arange(0, self.temp_bin) * temp_space
        else:
            self.temperatures = get_ms_array(temperatures, ms.float32)
            if self.temperatures.ndim > 2:
                raise ValueError(f'The rank(ndim) of temperatures cannot be larger than 2 '
                                 f'but got: {self.temperatures.ndim}')
            if self.temperatures.ndim < 2:
                self.temperatures = F.reshape(self.temperatures, (1, -1))
            if self.temperatures.shape[0] != self.num_walker:
                if self.temperatures.shape[0] != 1:
                    raise ValueError(f'The 1st dimension of temperatures ({self.temperatures.shape[0]}) '
                                     f'does match the number of multiple walkers ({self.num_walker})')
                self.temperatures = msnp.broadcast_to(
                    self.temperatures, (self.num_walker, -1))
            # T
            self.temp_bin = self.temperatures.shape[-1]
            # (B)
            self.temp_min = msnp.min(self.temperatures, axis=-1)
            self.temp_max = msnp.max(self.temperatures, axis=-1)

        if (self.temp_max <= self.temp_min).any():
            raise ValueError(f'temp_max ({self.temp_max}) must be larger than '
                             f'temp_min ({self.temp_min})!')
        if (self.sim_temp >= self.temp_max).any():
            raise ValueError(f'temp_max ({self.temp_max}) must be larger than '
                             f'sim_temp ({self.sim_temp})!')
        if (self.sim_temp <= self.temp_min).any():
            raise ValueError(f'temp_min ({self.temp_min}) must be less than '
                             f'sim_temp ({self.sim_temp})!')

        self.log_bins = msnp.log(Tensor(self.temp_bin, ms.float32))

        # (B, 1) <- (B)
        sim_temp = F.expand_dims(self.sim_temp, -1)
        # (B, 1)
        self.temp0_index, self.temp0_ratio = self.find_temp_index(sim_temp)

        self.boltzmann = self.units.boltzmann
        # (B, 1)
        self.kbt_sim = self.boltzmann * F.expand_dims(self.sim_temp, -1)
        self.beta_sim = msnp.reciprocal(self.kbt_sim)

        # (B, T)
        self.kbtk = self.boltzmann * self.temperatures
        self.betak = msnp.reciprocal(self.kbtk)

        # (B, 1)
        self.kbt_min = self.boltzmann * F.expand_dims(self.temp_min, -1)
        self.beta_min = msnp.reciprocal(self.kbt_min)

        # (B, T)
        # self.weighting_factors: \log{n_k}
        self.weighting_factors = Parameter(F.zeros_like(self.betak),
                                           name='weighting_factors', requires_grad=False)

        # \log{0} = -\infty
        neg_inf = Tensor(float('-inf'), ms.float32)

        # (B, T)
        self.zero_rbfb = msnp.broadcast_to(
            neg_inf, (self.num_walker, self.temp_bin))

        # (B, T)
        self.partitions = Parameter(
            self.zero_rbfb, name='partitions', requires_grad=False)

        # (B, T-1)
        self.normal = Parameter(msnp.broadcast_to(neg_inf, (self.num_walker, self.temp_bin - 1)),
                                name='partition_normalization', requires_grad=False)

        self.ratio_exponent = get_ms_array(ratio_exponent, ms.float32)
        self.step_weight = get_ms_array(step_weight, ms.float32)
        self.weight_bias = get_ms_array(weight_bias, ms.float32)

        energy_shift = get_ms_array(energy_shift, ms.float32)
        if energy_shift.ndim > 2:
            raise ValueError(f'The rank(ndim) of energy_shift cannot be larger than 2 '
                             f'but got: {energy_shift.ndim}')
        if energy_shift.ndim < 2:
            energy_shift = F.reshape(energy_shift, (-1, 1))
        if energy_shift.shape[0] != self.num_walker:
            if energy_shift.shape[0] != 1:
                raise ValueError(f'The 1st dimension of energy_shift does not match '
                                 f'the number of multiple walkers ({self.num_walker})')
            energy_shift = msnp.broadcast_to(
                energy_shift, (self.num_walker, 1))

        # (B, 1)
        self.zeros = msnp.zeros((self.num_walker, 1), ms.float32)

        # (B, 1)
        self.energy_shift = Parameter(energy_shift, name='energy_shift', requires_grad=False)
        self.min_energy = Parameter(energy_shift, name='minimum_energy', requires_grad=False)
        self.reweight_factor = Parameter(self.zeros, name='reweight_factor', requires_grad=False)
        self.bias = Parameter(self.zeros, name='its_bias', requires_grad=False)
        self.step = Parameter(Tensor(0, ms.int32), name='iteration_step', requires_grad=False)

        # (B, T)
        self.log_betak = msnp.log(self.betak)

    def find_temp_index(self, temperature: Tensor) -> Tuple[Tensor, Tensor]:
        r"""find the index of a specify temperatures at the serial of integration temperatues.

        Args:
            temperature (Tensor):   Tensor of shape `(B, ...)`. Data type is float.
                                    Temperature.

        Returns:
            index (Tensor): Tensor of shape `(B, ...)`. Data type is int.
                            Index of the nearest temperature.

            ratio (Tensor): Tensor of shape `(B, ...)`. Data type is float.
                            Ratio to adjust the factor by neighbouring temperature.

        """

        if (temperature >= self.temp_max).any():
            raise ValueError(f'temperature ({temperature}) must be less than '
                             f'temp_max ({self.temp_max})')

        if (temperature <= self.temp_min).any():
            raise ValueError(f'temperature ({temperature}) must be greater than '
                             f'temp_min ({self.temp_min})')

        # (B, ..., T) = (B, 1, T) - (B, ..., 1)
        temp_diff = F.abs(F.expand_dims(self.temperatures, -2) -
                          F.expand_dims(temperature, -1))

        # (B, 1)
        index = msnp.argmin(temp_diff, axis=-1)

        nearest = F.gather_d(self.temperatures, -1, index)

        mask = temperature >= nearest

        index = F.select(mask, index, index - 1)

        low = F.gather_d(self.temperatures, -1, index)
        high = F.gather_d(self.temperatures, -1, index + 1)

        ratio = (temperature - low) / (high - low)

        return index, ratio

    def get_weighting_factor(self, index: Tensor, ratio: Tensor = None) -> Tensor:
        r"""get weighting reweight of specify index.

        Args:
            index (Tensor): Tensor of shape `(B, ...)`. Data type is int.
                            Temperature index.

            ratio (Tensor): Tensor of shape `(B, ...)`. Data type is float.
                            Ratio to adjust the factor by neighbouring temperature.

        Returns:
            fb (Tensor):    Tensor of shape `(B, ...)`. Data type is float.
                            Weighting factor

        """

        if ratio is None:
            return F.gather_d(self.weighting_factors, -1, index)

        fb_low = F.gather_d(self.weighting_factors, -1, index)
        fb_high = F.gather_d(self.weighting_factors, -1, index + 1)
        return fb_low + (fb_high - fb_low) * ratio

    def calc_reweight_factor(self, temperature: Tensor = None) -> Tensor:
        r"""calculate reweight factor :math:`c(t)` for ITS.

        Args:
            temperature (Tensor):   Tensor of shape `(B, ...)`. Data type is float.
                                    Temperature to reweight. If None is given,
                                    the simulation temperature will be used.
                                    Default: ``None``.

        Returns:
            rct (Tensor):   Tensor of shape `(B, ...)`. Data type is float.
                            Reweight factor :math:`c(t)`

        """
        if temperature is None:
            index = self.temp0_index
            ratio = self.temp0_ratio
            kbt = self.kbt_sim
        else:
            index, ratio = self.find_temp_index(temperature)
            kbt = self.boltzmann * temperature

        fb = self.get_weighting_factor(index, ratio)
        # c_{T}(t) = -\frac{1}{\beta_{T}} \log{N n_{T}(t)}
        rct = - kbt * (self.log_bins + fb)
        return rct

    def change_energy_shift(self, peshift: Tensor) -> Tensor:
        r"""change the the shift value for potential energy

        Arg:
            peshift (Tensor):   Tensor of shape `(B, 1)`. Data type is float.
                                Energy shift.

        """
        # (B, 1) = (B, 1) + (B, 1) * (B, 1)
        fb0 = self.weighting_factors[:, [0]] + self.beta_min * peshift
        # (B, T) = (B, T) * (B, 1) - (B, 1)
        fb_add = self.betak * peshift - fb0
        peshift = F.depend(peshift, F.assign_add(self.weighting_factors, fb_add))
        return F.assign(self.energy_shift, peshift)

    def update(self):
        r"""update ITS"""
        if (self.min_energy < self.energy_shift).any():
            self.change_energy_shift(self.min_energy)

        F.assign(self.step, self.step + 1)

        # (B, T-1)
        fb0 = self.weighting_factors[:, :-1]
        fb1 = self.weighting_factors[:, 1:]
        # fb_ratio: \log{m_k(t-1)}, k \in [1, N-1]
        # m_k(t-1) = n_k(t-1) / n_{k+1}(t-1)
        fb_ratio0 = fb0 - fb1

        # (B, T-1)
        rbfb0 = self.partitions[:, :-1]
        rbfb1 = self.partitions[:, 1:]
        # rbfb_ratio: \log{r_k(t)}
        # r_k(t-1) = \frac{P_{k+1}(t-1)}{P_k(t-1)}
        rbfb_ratio0 = rbfb1 - rbfb0

        # rb: \log{w_k(t)}, k \in [1, N-1]
        # w_k(t) = e ^ {c_t t} [P_k(t-1) P_{k+1}(t-1) ] ^ {c_p}
        # Default: c_p = 0.5 and c_t = 0, so w_k(t) = \sqrt{p_k(t-1) p_{k+1}(t-1)}
        rb = (rbfb0 + rbfb1) * self.ratio_exponent + \
            self.step * self.step_weight
        # normal: \log{W_k(t)}
        # W_k(t) = \sum_{\tau}^{t}{w_k(\tau)} = W_k(t-1) + w_k(t)
        normal = msnp.logaddexp(self.normal, rb)

        # (B, T-1)
        # weights: \log{w'_k(t)}, k \in [1, N-1]
        # w'_k(t) = c_w w_k(t)
        weights = self.weight_bias + rb
        # weights_norm: \log{W'_k(t)}
        # W'_k(t) = W_k(t-1) + w'_k(t)
        weights_norm = msnp.logaddexp(self.normal, weights)

        # (B, T-1)
        # fb_ratio1: \log{m_k(t+1)}
        # m'_k(t) = m_k(t-1) \frac{P_{k+1}(t-1)}{P_k(t-1)}
        #         = m_k(t-1) r_k(t-1)
        # m_k(t) = n_k(t) / n_{k+1}(t)
        #        = \frac{\sum_{\tau}^{t}{w'_k(\tau) m'_k(\tau)}}{\sum_{\tau}^{t}{w'_k(\tau)}}
        #        = \frac{W'_k(t-1) m_k(t-1) + w'_k(t) m'_k(t)}{W'_k(t-1) + w'_k(t)}
        #        = m_k(t-1) \frac{W'_k(t-1) + w'_k(t) r_k(t-1)}{W'_k(t-1) + w'_k(t)}
        #        # At the first iteration step, W_k(0) = 0, then m_k(1) = m'_k(1) = m_k(0) r_k(0)
        fb_ratio1 = fb_ratio0 + \
            msnp.logaddexp(self.normal, rbfb_ratio0 + weights) - weights_norm

        # (B, T-1)
        # fb_new: \log{n'_k(t)}, k \in [1, N-1]
        # n'_k(t) = \prod_{i=1}^{k}{\frac{1}{m_i(t)}}
        #         = \prod_{i=1}^k{\frac{n_{i+1}(t)}{n_i(t)}}
        #         = \frac{n_{k+1}(t)}{n_1(t)}
        fb_new = F.cumsum(-fb_ratio1, -1)

        # (B, T) <- (B, 1) & (B, T-1)
        # Let n_1(t) = 1, then fb_new: \log{n_k(t)}, k \in [1, N]
        fb_new = func.concat_last_dim((self.zeros, fb_new))

        # (B, 1)
        rct = self.calc_reweight_factor()

        # (B, T)
        F.assign(self.weighting_factors, fb_new)
        F.assign(self.partitions, self.zero_rbfb)
        # (B, T-1)
        F.assign(self.normal, normal)
        # (B, 1)
        F.assign(self.reweight_factor, rct)
        return self

    def construct(self,
                  energy: Tensor = 0,
                  energy_ad: Tensor = 0,
                  force: Tensor = 0,
                  force_ad: Tensor = 0,
                  virial: Tensor = None,
                  virial_ad: Tensor = None,
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        """modify atomice force by ITS.

        Args:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Potential energy from ForceCell.
            energy_ad (Tensor): Tensor of shape (B, 1). Data type is float.
                                Potential energy from EnergyCell.
            force (Tensor):     Tensor of shape (B, A, D). Data type is float.
                                Atomic forces from ForceCell.
            force_ad (Tensor):  Tensor of shape (B, A, D). Data type is float.
                                Atomic forces calculated by automatic differentiation.
            virial (Tensor):    Tensor of shape (B, D). Data type is float.
                                Virial calculated from ForceCell.
            virial_ad (Tensor): Tensor of shape (B, D). Data type is float.
                                Virial calculated calculated by automatic differentiation.

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Totoal potential energy for simulation.
            force (Tensor):     Tensor of shape (B, A, D). Data type is float.
                                Total atomic force for simulation.
            virial (Tensor):    Tensor of shape (B, D). Data type is float.
                                Total virial for simulation.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.
        """

        force = force + force_ad
        energy = energy + energy_ad

        if virial is not None or virial_ad is not None:
            if virial is None:
                virial = 0
            if virial_ad is None:
                virial_ad = 0
            virial = virial + virial_ad

        if self.update_pace > 0:
            # (B, 1)
            min_energy = F.select(energy < self.min_energy,
                                  F.stop_gradient(energy), self.min_energy)
            energy = F.depend(energy, F.assign(self.min_energy, min_energy))

        # (B, 1)
        energy += self.energy_shift

        # (B, T) - (B, T) * (B, 1)
        # log {n_k e ^ {-\beta_k * U(R)}}
        gf = self.weighting_factors - self.betak * energy
        # log {n_k \beta_k e ^ {-\beta_k * U(R)}}
        bgf = gf + self.log_betak

        if self.update_pace > 0:
            # (B, T)
            gf_ = F.stop_gradient(gf)
            if self.multi_walkers and energy.shape[0] > 1:
                # (1, T) <- (B, T)
                gf_ = F.logsumexp(gf_, 0, True)

            # rbfb: \log{P_k}
            # P_k(t) = \sum_{\tau}^t{p_k(\tau)} = \sum_{\tau}{n_k e ^ {-\beta_k E[R(\tau)]}}
            rbfb = msnp.logaddexp(self.partitions, gf_)
            gf = F.depend(gf, F.assign(self.partitions, rbfb))

        # (B, 1) <- (B, T)
        # \sum_k^N {n_k e ^ {-\beta_k U(R)}}
        gfsum = F.logsumexp(gf, -1, True)
        # (B, 1) * (B, 1)
        # U_{eff}(R) = -\frac{1}{\beta_0} \log{\sum_k^N {n_k e ^ {-\beta_k U(R)}}}
        eff_energy = -self.kbt_sim * gfsum

        # (B, 1) <- (B, T)
        # \sum_k^N {n_k \beta_k e ^ {-\beta_k U(R)}}
        bgfsum = F.logsumexp(bgf, -1, True)
        # F_{eff}(R) = \frac{\sum_k^N {n_k \beta_k e ^ {-\beta_k U(R)}}}
        #              {\beta_0 \sum_k^N {n_k e ^ {-\beta_k U(R)}}} F(R)
        force *= self.kbt_sim * F.exp(bgfsum - gfsum)

        # (B, 1) - (B, 1) - (B, 1)
        bias = eff_energy - energy - self.reweight_factor
        energy = F.depend(energy, F.assign(self.bias, bias))
        # (B, 1)
        energy = eff_energy

        return energy, force, virial

    def _check_temp(self, temp, name: str) -> Tensor:
        """check the shape of temperature related variables"""
        temp = get_ms_array(temp, ms.float32)
        if temp.ndim > 1:
            raise ValueError(f'The rank(ndim) of {name} cannot be larger than 1 '
                             f'but got: {temp.ndim}')
        if temp.ndim == 0:
            temp = F.reshape(temp, (-1,))

        if temp.size != self.num_walker:
            if temp.size != 1:
                raise ValueError(f'The size of {name} ({temp.size}) cannot match '
                                 f'the number of multiple walkers ({self.num_walker})')
            temp = msnp.broadcast_to(temp.reshape((1,)), (self.num_walker,))
        return temp

    def _check_ratio(self, ratio, name: str) -> Tensor:
        """check the shape of ratio related variables"""
        ratio = get_ms_array(ratio, ms.float32)
        if ratio.ndim > 2:
            raise ValueError(f'The rank(ndim) of {name} cannot be larger than 2 '
                             f'but got: {ratio.ndim}')
        if ratio.ndim < 2:
            ratio = F.reshape(ratio, (1, -1))
        if ratio.shape[0] != self.num_walker:
            if ratio.shape[0] != 1:
                raise ValueError(f'The 1st dimension of {name} ({ratio.shape[0]}) does not match '
                                 f'the number of multiple walkers ({self.num_walker})')
            ratio = msnp.broadcast_to(ratio, (self.num_walker, -1))
        return ratio

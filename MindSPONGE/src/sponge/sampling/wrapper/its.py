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
"""Integrated tempering sampling (ITS)"""

from typing import Tuple
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .wrapper import EnergyWrapper
from ... import function as func
from ...function import any_none, get_ms_array, get_integer


class ITS(EnergyWrapper):
    r"""Energy wrapper for (selected) integrated tempering sampling (ITS/SITS).

    References:

        Gao, Y. Q.
        An Integrate-over-Temperature Approach for Enhanced Sampling [J].
        The Journal of Chemical Physics, 2008, 128(6): 064105.

        Yang, L.; Gao, Y. Q.
        A Selective Integrated Tempering Method [J].
        The Journal of Chemical Physics, 2009, 131(21): 214109.

        Yang, Y. I.; Niu, H.; Parrinello, M.
        Combining Metadynamics and Integrated Tempering Sampling [J].
        The Journal of Physical Chemistry Letters, 2018, 9(22): 6426-6430.

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

        temp_bin (int):         Number of temperatures for integrationintergration.
                                Only used when `temperature` is None.
                                Default: ``None``.

        unlinear_temp (bool)    Whether to generate unlinear integration temperatures
                                Default: ``False``.

        temperatures (Tensor):  Temperatures for integration.
                                The shape of tensor is `(B, T)`, the data type is float.
                                Default: ``None``.

        update_pace (int):      Freuency for updating ITS. Default: 100

        num_walker (int):       Number of multiple walkers.
                                Cannot be None when `share_parameter` is False. Default: ``None``.

        share_parameter (bool): Whether to share ITS parameters for all walkers. Default: ``True``.

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
                 num_walker: int = None,
                 share_parameter: bool = True,
                 energy_shift: float = 0,
                 energy_ratio: float = 1,
                 bias_ratio: float = 1,
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

        self.share_parameter = share_parameter
        self.num_walker = get_integer(num_walker)
        self.num_parameter = self.num_walker
        if self.share_parameter:
            self.num_parameter = 1
        if self.num_walker is None:
            if self.share_parameter:
                self.num_walker = 1
            else:
                raise ValueError('num_walkers must be given when share_parameter is False!')

        # (B)
        sim_temp = self._check_temp(sim_temp, 'sim_temp')
        self.sim_temp = Parameter(sim_temp, name='sim_temp', requires_grad=False)

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
                temp_ratio = msnp.exp(msnp.log(self.temp_max / self.temp_min) / temp_bin)
                # (B, 1) <- (B)
                temp_ratio = F.expand_dims(temp_ratio, -1)
                # (B, T)
                self.temperatures = temp_min * msnp.power(temp_ratio, msnp.arange(self.temp_bin))
            else:
                # (B)
                temp_space = (self.temp_max - self.temp_min) / temp_bin
                # (B, 1) <- (B)
                temp_space = F.expand_dims(temp_space, -1)

                # (B, T) = (B, 1) + (T) * (B, 1)
                self.temperatures = temp_min + msnp.arange(0, self.temp_bin) * temp_space
        else:
            self.temperatures = get_ms_array(temperatures, ms.float32)
            if self.temperatures.ndim > 2:
                raise ValueError(f'The rank(ndim) of temperatures cannot be larger than 2 '
                                 f'but got: {self.temperatures.ndim}')
            if self.temperatures.ndim < 2:
                self.temperatures = F.reshape(self.temperatures, (1, -1))
            if self.temperatures.shape[0] != self.num_parameter:
                if self.temperatures.shape[0] != 1:
                    raise ValueError(f'The 1st dimension of temperatures ({self.temperatures.shape[0]}) '
                                     f'does match the number of multiple walkers ({self.num_parameter})')
                self.temperatures = msnp.broadcast_to(self.temperatures, (self.num_parameter, -1))
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

        # (1, 1) or (B, 1)
        sim_temp = F.expand_dims(self.sim_temp, -1)
        self.temp0_index, self.temp0_ratio = self.find_temp_index(sim_temp)

        self.boltzmann = self.units.boltzmann

        # (1, 1) or (B, 1)
        self.kbt_array = self.boltzmann * self.temperatures
        self.beta_array = msnp.reciprocal(self.kbt_array)

        # (1, 1) or (B, 1)
        self.kbt_min = self.boltzmann * F.expand_dims(self.temp_min, -1)
        self.beta_min = msnp.reciprocal(self.kbt_min)

        # (1, U) or (B, U)
        self.energy_ratio = self._check_ratio(energy_ratio, 'energy_ratio')
        self.pot_rest_ratio = 1.0 - self.energy_ratio

        # (1, V) or (B, V)
        self.bias_ratio = self._check_ratio(bias_ratio, 'bias_ratio')
        self.bias_rest_ratio = 1.0 - self.bias_ratio

        # \log{0} = -\infty
        # neg_inf = Tensor(float(-5e4), ms.float32)
        neg_inf = Tensor(float('-inf'), ms.float32)

        # (1, T-1) or (B, T-1)
        self.normal = Parameter(msnp.broadcast_to(neg_inf, (self.num_parameter, self.temp_bin - 1)),
                                name='partition_normalization', requires_grad=False)

        # (1, T) or (B, T)
        # self.weight_factors: \log{n_k}
        self.weight_factors = Parameter(F.zeros_like(self.beta_array),
                                        name='weighting_factors', requires_grad=False)

        # (1, 1) or (B, 1)
        self.zeros = msnp.zeros((self.num_parameter, 1), ms.float32)
        # c(t)
        self.reweight_factor = Parameter(self.zeros, name='reweight_factor', requires_grad=False)

        energy_shift = get_ms_array(energy_shift, ms.float32)
        if energy_shift.ndim > 2:
            raise ValueError(f'The rank(ndim) of energy_shift cannot be larger than 2 '
                             f'but got: {energy_shift.ndim}')
        if energy_shift.ndim < 2:
            energy_shift = F.reshape(energy_shift, (-1, 1))
        if energy_shift.shape[0] != self.num_parameter:
            if energy_shift.shape[0] != 1:
                raise ValueError(f'The 1st dimension of energy_shift does not match '
                                 f'the number of multiple num_parameter(s) ({self.num_parameter})')
            energy_shift = msnp.broadcast_to(energy_shift, (self.num_parameter, 1))

        self.energy_shift = Parameter(energy_shift, name='energy_shift', requires_grad=False)

        self.ratio_exponent = get_ms_array(ratio_exponent, ms.float32)
        self.step_weight = get_ms_array(step_weight, ms.float32)
        self.weight_bias = get_ms_array(weight_bias, ms.float32)

        # (B, T)
        self.zero_rbfb = msnp.broadcast_to(neg_inf, (self.num_walker, self.temp_bin))
        self.partitions = Parameter(self.zero_rbfb, name='partitions', requires_grad=False)

        # (B, 1)
        self.min_energy = Parameter(msnp.zeros((self.num_walker, 1), ms.float32),
                                    name='minimum_energy', requires_grad=False)

        self.step = Parameter(Tensor(0, ms.int32), name='iteration_step', requires_grad=False)

    @property
    def sim_kbt(self):
        """:math:`k_b T` at simlation temperature"""
        # (1, 1) or (B, 1)
        return self.get_sim_kbt()

    @property
    def sim_beta(self):
        """:math:`\\beta` at simlation temperature"""
        # (1, 1) or (B, 1)
        return self.get_sim_beta()

    def get_sim_kbt(self) -> Tensor:
        """get :math:`k_b T` at simlation temperature"""
        return self.boltzmann * F.expand_dims(self.sim_temp, -1)

    def get_sim_beta(self) -> Tensor:
        """get :math:`\\beta` at simlation temperature"""
        return msnp.reciprocal(self.boltzmann * F.expand_dims(self.sim_temp, -1))

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
            return F.gather_d(self.weight_factors, -1, index)

        fb_low = F.gather_d(self.weight_factors, -1, index)
        fb_high = F.gather_d(self.weight_factors, -1, index + 1)
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
            kbt = self.get_sim_kbt()
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
        # (1, 1) or (B, 1)
        fb0 = self.weight_factors[:, [0]] + self.beta_min * peshift
        # (1, T) = (1, T) * (1, 1) - (1, 1) or (B, T) = (B, T) * (B, 1) - (B, 1)
        fb_add = self.beta_array * peshift - fb0
        peshift = F.depend(peshift, F.assign_add(self.weight_factors, fb_add))
        return F.assign(self.energy_shift, peshift)

    def update(self):
        """update ITS"""

        # (B, 1)
        min_energy = self.min_energy
        # (B, T)
        partitions = self.partitions
        if self.share_parameter and self.num_walker > 1:
            # (1, 1) <- (B, 1)
            min_energy = msnp.amin(min_energy, axis=0, keepdims=True)
            # (1, T) <- (B, T)
            partitions = F.logsumexp(partitions, 0, True)

        if (min_energy + self.energy_shift < 0).any():
            self.change_energy_shift(-min_energy)

        F.assign(self.step, self.step + 1)

        # (1, T-1) or (B, T-1)
        fb0 = self.weight_factors[:, :-1]
        fb1 = self.weight_factors[:, 1:]
        # fb_ratio: \log{m_k(t-1)}, k \in [1, N-1]
        # m_k(t-1) = n_k(t-1) / n_{k+1}(t-1)
        fb_ratio0 = fb0 - fb1

        # (1, T-1) or (B, T-1)
        rbfb0 = partitions[:, :-1]
        rbfb1 = partitions[:, 1:]
        # rbfb_ratio: \log{r_k(t)}
        # r_k(t-1) = \frac{P_{k+1}(t-1)}{P_k(t-1)}
        rbfb_ratio0 = rbfb1 - rbfb0

        # (1, T-1) or (B, T-1)
        # rb: \log{w_k(t)}, k \in [1, N-1]
        # w_k(t) = e ^ {c_t t} [P_k(t-1) P_{k+1}(t-1) ] ^ {c_p}
        # Default: c_p = 0.5 and c_t = 0, so w_k(t) = \sqrt{p_k(t-1) p_{k+1}(t-1)}
        rb = (rbfb0 + rbfb1) * self.ratio_exponent + self.step * self.step_weight
        # normal: \log{W_k(t)}
        # W_k(t) = \sum_{\tau}^{t}{w_k(\tau)} = W_k(t-1) + w_k(t)
        normal = msnp.logaddexp(self.normal, rb)

        # (1, T-1) or (B, T-1)
        # weights: \log{w'_k(t)}, k \in [1, N-1]
        # w'_k(t) = c_w w_k(t)
        weights = self.weight_bias + rb
        # weights_norm: \log{W'_k(t)}
        # W'_k(t) = W_k(t-1) + w'_k(t)
        weights_norm = msnp.logaddexp(self.normal, weights)

        # (1, T-1) or (B, T-1)
        # fb_ratio1: \log{m_k(t+1)}
        #
        # m'_k(t) = m_k(t-1) \frac{P_{k+1}(t-1)}{P_k(t-1)}
        #         = m_k(t-1) r_k(t-1)
        #
        # m_k(t) = n_k(t) / n_{k+1}(t)
        #        = \frac{\sum_{\tau}^{t}{w'_k(\tau) m'_k(\tau)}}{\sum_{\tau}^{t}{w'_k(\tau)}}
        #        = \frac{\sum_{\tau}^{t-1}{w'_k(\tau) m'_k(\tau)} +
        #          w'_k(t) m'_k(t)}{\sum_{\tau}^{t-1}{w'_k(\tau)}+w'_k(t)}
        #
        # \because
        # m_k(t-1) = \frac{\sum_{\tau}^{t-1}{w'_k(\tau) m'_k(\tau)}}{W'_k(t-1)}
        #
        # \therefore
        # m_k(t) = \frac{W'_k(t-1) m_k(t-1) + w'_k(t) m'_k(t)}{W'_k(t-1) + w'_k(t)}
        #        = m_k(t-1) \frac{W'_k(t-1) + w'_k(t) r_k(t-1)}{W'_k(t-1) + w'_k(t)}
        #
        # At the first iteration step, W_k(0) = 0, then m_k(1) = m'_k(1) = m_k(0) r_k(0)
        fb_ratio1 = fb_ratio0 + msnp.logaddexp(self.normal, rbfb_ratio0 + weights) - weights_norm

        # (1, T-1) or (B, T-1)
        # fb_new: \log{n'_k(t)}, k \in [1, N-1]
        # n'_k(t) = \prod_{i=1}^{k}{\frac{1}{m_i(t)}}
        #         = \prod_{i=1}^k{\frac{n_{i+1}(t)}{n_i(t)}}
        #         = \frac{n_{k+1}(t)}{n_1(t)}
        fb_new = F.cumsum(-fb_ratio1, -1)

        # (1, T) <- (1, 1) & (1, T-1) OR (B, T) <- (B, 1) & (B, T-1)
        # Let n_1(t) = 1, then fb_new: \log{n_k(t)}, k \in [1, N]
        fb_new = func.concat_last_dim((self.zeros, fb_new))

        # (1, 1) or (B, 1)
        rct = self.calc_reweight_factor()

        # (1, T) or (B, T)
        F.assign(self.weight_factors, fb_new)
        F.assign(self.partitions, self.zero_rbfb)
        # (1, T-1) or (B, T-1)
        F.assign(self.normal, normal)
        # (1, 1) or (B, 1)
        F.assign(self.reweight_factor, rct)
        return self

    def construct(self, potentials: Tensor, biases: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""merge the potential energies and bias potential energies.

        Args:
            potentials (Tensor):    Tensor of shape `(B, U)`. Data type is float.
                                    Potential energies.
            biases (Tensor):        The shape of tensor is `(B, V)`. The data type is float.
                                    Bias potential energies. Default: ``None``.

        Returns:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total energy (potential energy and bias energy).
            bias (Tensor):      Tensor of shape `(B, 1)`. Data type is float.
                                Total bias potential used for reweighting calculation.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation.
            U:  Dimension of potential energy.
            V:  Dimension of bias potential.
        """

        # (B, U) * (B, U)
        # U_{select}(R)
        enhanced_pot = potentials * self.energy_ratio
        # U_{rest}(R) = U(R) - U_{select}(R)
        rest_potential = potentials * self.pot_rest_ratio

        # (B, 1) <- (B, U)
        # V_{select}(R)
        enhanced_pot = func.keepdims_sum(enhanced_pot, -1)
        # V_{rest}(R) = V(R) - V_{select}(R)
        rest_potential = func.keepdims_sum(rest_potential, -1)

        enhanced_bias = 0
        rest_bias = 0
        if biases is not None:
            # (B, V)
            enhanced_bias = biases * self.bias_ratio
            rest_bias = biases * self.bias_rest_ratio
            # (B, 1) <- (B, V)
            enhanced_bias = func.keepdims_sum(enhanced_bias, -1)
            rest_bias = func.keepdims_sum(rest_bias, -1)

        # (B, 1):
        # E(R) = U_{select}(R) + V_{select}(R)
        enhanced_energy = enhanced_pot + enhanced_bias

        if self.update_pace > 0:
            # (B, 1)
            min_energy = F.stop_gradient(enhanced_energy)
            min_energy = F.select(min_energy < self.min_energy, min_energy, self.min_energy)
            enhanced_energy = F.depend(enhanced_energy, F.assign(self.min_energy, min_energy))

        # (B, 1) + (1, 1) OR (B, 1) + (B, 1)
        enhanced_energy += self.energy_shift

        # (B, T) - (B, T) * (B, 1)
        # \log {\left [ n_k e ^ {-\beta_k U(R)} \right ] }
        # gf = self.weight_factors - self.beta_array * enhanced_energy
        gf1 = - self.beta_array * enhanced_energy
        gf = self.weight_factors + gf1

        if self.update_pace > 0:
            # (B, T)
            # rbfb: \log{P_k}
            # P_k(t) = \sum_{\tau}^{t}{p_k(\tau)}
            #        = \sum_{\tau}^{t}{n_k e ^ {-\beta_k E[R(\tau)]}}
            rbfb = msnp.logaddexp(self.partitions, F.stop_gradient(gf))
            gf = F.depend(gf, F.assign(self.partitions, rbfb))

        # (B, 1) <- (B, T)
        # \log{\sum_k^N {n_k e ^ {-\beta_k U(R)}}}
        gfsum = F.logsumexp(gf, -1, True)
        # (B, 1) * (B, 1)
        # U_{eff}(R) = -\frac{1}{\beta_0} \log{\sum_k^N {n_k e ^ {-\beta_k U(R)}}}
        eff_energy = -self.get_sim_kbt() * gfsum

        # (B, 1)
        energy = eff_energy + rest_potential + rest_bias
        # (B, 1) <- (B, U)
        potential = func.keepdims_sum(potentials, -1) + self.energy_shift
        # (B, 1) - (B, 1) - (B, 1)
        bias = energy - potential - self.reweight_factor

        return energy, bias

    def _check_temp(self, temp, name: str) -> Tensor:
        """check the shape of temperature related variables"""
        temp = get_ms_array(temp, ms.float32)
        if temp.ndim > 1:
            raise ValueError(f'The rank(ndim) of {name} cannot be larger than 1 '
                             f'but got: {temp.ndim}')
        if temp.ndim == 0:
            temp = F.reshape(temp, (-1,))

        if temp.size != self.num_parameter:
            if temp.size != 1:
                raise ValueError(f'The size of {name} ({temp.size}) cannot match '
                                 f'the number of parameter(s) ({self.num_parameter})')
            temp = msnp.broadcast_to(temp.reshape((1,)), (self.num_parameter,))
        return temp

    def _check_ratio(self, ratio, name: str) -> Tensor:
        """check the shape of ratio related variables"""
        ratio = get_ms_array(ratio, ms.float32)
        if ratio.ndim > 2:
            raise ValueError(f'The rank(ndim) of {name} cannot be larger than 2 '
                             f'but got: {ratio.ndim}')
        if ratio.ndim < 2:
            ratio = F.reshape(ratio, (1, -1))
        if ratio.shape[0] != self.num_parameter:
            if ratio.shape[0] != 1:
                raise ValueError(f'The 1st dimension of {name} ({ratio.shape[0]}) does not match '
                                 f'the number of parameter(s) ({self.num_parameter})')
            ratio = msnp.broadcast_to(ratio, (self.num_parameter, ratio.shape[-1]))
        return ratio

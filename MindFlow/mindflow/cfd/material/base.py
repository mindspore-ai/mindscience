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
# ==============================================================================
"""abstract base class for material."""
from abc import abstractmethod


class Material:
    """Abstract base class for material."""
    def __init__(self, config):
        self.gamma = None
        self.gas_constant = None
        self.cp = None

        self.given_dynamic_viscosity = config.get('dynamic_viscosity', 0.0)
        self.sutherland = config.get('sutherland', False)
        self.sutherland_parameters = config.get('sutherland_parameters', None)

        self.bulk_viscosity = config.get('bulk_viscosity', 0.0)

        self.given_thermal_conductivity = config.get('thermal_conductivity', 0.0)
        self.prandtl_conductivity = config.get('prandtl_conductivity', False)
        self.prandtl_number = config.get('prandtl_number', 0.0)

    def thermal_conductivity(self, pri_var):
        res = None
        if self.prandtl_conductivity:
            res = self.cp * self.dynamic_viscosity(pri_var) / self.prandtl_number
        else:
            res = self.given_thermal_conductivity
        return res

    def dynamic_viscosity(self, pri_var):
        res = None
        if self.sutherland:
            temp = self.temperature(pri_var)
            mu_0, temp_0, coe = self.sutherland_parameters
            res = mu_0 * ((temp_0 + coe) / (temp + coe)) * (temp / temp_0) ** 1.5
        else:
            res = self.given_dynamic_viscosity
        return res

    @abstractmethod
    def sound_speed(self, pri_var):
        raise NotImplementedError()

    @abstractmethod
    def temperature(self, pri_var):
        raise NotImplementedError()

    @abstractmethod
    def energy(self, pri_var):
        raise NotImplementedError()

    @abstractmethod
    def total_energy(self, pri_var):
        raise NotImplementedError()

    @abstractmethod
    def total_enthalpy(self, pri_var):
        raise NotImplementedError()

    @abstractmethod
    def psi(self, pri_var):
        raise NotImplementedError()

    @abstractmethod
    def grueneisen(self, pri_var):
        raise NotImplementedError()

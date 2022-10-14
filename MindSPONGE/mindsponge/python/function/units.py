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
Units
"""

import math

__all__ = [
    'AVOGADRO_NUMBER',
    'BOLTZMANN_CONSTANT',
    'GAS_CONSTANT',
    'ELEMENTARY_CHARGE',
    'VACCUM_PERMITTIVITY',
    'COULOMB_CONSTANT',
    'STANDARD_ATMOSPHERE',
    'Length',
    'Energy',
    'get_length_ref',
    'get_length_unit',
    'get_length_unit_name',
    'get_energy_ref',
    'get_energy_unit',
    'get_energy_unit_name',
    'length_convert',
    'energy_convert',
    'Units',
    'global_units',
    'set_global_length_unit',
    'set_global_energy_unit',
    'set_global_units',
]

# origin constant
AVOGADRO_NUMBER = 6.02214076e23           # N_A
BOLTZMANN_CONSTANT = 1.380649e-23         # kB
GAS_CONSTANT = 8.31446261815324           # R unit=1/mol
ELEMENTARY_CHARGE = 1.602176634e-19       # e unit=C
VACCUM_PERMITTIVITY = 8.854187812813e-12  # \epsilon_0
COULOMB_CONSTANT = 8.9875517923e9         # k = 1/(4*pi*\epsilon_0) unit=N*m^2/C^2
STANDARD_ATMOSPHERE = 101325              # atm

_LENGTH_UNITS = (
    'nm',
    'um',
    'a',
    'angstrom',
    'bohr',
    'user',
    'none',
)

_LENGTH_REF = {
    'nm': 1.0,
    'um': 1e3,
    'a': 0.1,
    'angstrom': 0.1,
    'bohr': 0.052917721067,
    'user': None,
    'none': None,
}

_LENGTH_NAME = {
    'nm': 'nm',
    'um': 'um',
    'a': 'Angstrom',
    'bohr': 'Bohr',
    'user': 'User_Length',
    'none': "None"
}

_ENERGY_UNITS = (
    'kj/mol',
    'j/mol',
    'kcal/mol',
    'cal/mol',
    'ha',
    'ev',
    'mev'
    'kbt0',
    'kbt300',
    'user',
    'none',
)

_ENERGY_REF = {
    'kj/mol': 1.0,
    'j/mol': 1e-3,
    'kcal/mol': 4.184,
    'cal/mol': 4.184e-3,
    'ha': 2625.5002,
    'ev': 96.48530749925793,
    'mev': 0.09648530749925793,
    'kbt0': 2.271095464,
    'kbt300': 2.494338785,
    'user': None,
    'none': None,
}

_ENERGY_NAME = {
    'kj/mol': 'kJ mol-1',
    'j/mol': 'J mol-1',
    'kcal/mol': 'kcal mol-1',
    'cal/mol': 'cal mol-1',
    'ha': 'Hartree',
    'ev': 'eV',
    'mev': 'meV',
    'kbt0': 'kBT(273.15K)',
    'kbt300': 'kBT(300K)',
    'user': 'User_Energy',
    'none': 'None',
}

# Boltzmann constant for simulation (kJ/mol)
_BOLTZMANN_DEFAULT_REF = 8.31446261815324e-3
# Coulomb constant for simulation (e^2*kJ/mol*nm)
# N_A*e^2/(4*pi*\epsilon_0)*1e9nm[1m]*1e-3kJ[1J]
_COULOMB_DEFAULT_REF = 138.93545764498226165718756672623
# Pressure 1 Bar in kJ mol-1 nm^3
_BAR_DEFAULT_REF = 16.6053906717384685


class Length:
    """
    Length.

    Args:
        value (float):   length value.
        unit (str):      length value unit. Default: 'nm'

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, value: float, unit: str = 'nm'):
        if isinstance(value, Length):
            self.__value = value.value
            self.__unit = value.unit
            self.__ref = value.ref
            self.__abs_size = value.abs_size
            self.__unit_name = value.unit_name
        elif isinstance(value, (float, int)):
            self.__value = float(value)
            if isinstance(unit, (str, Units)):
                self.__unit = get_length_unit(unit)
                self.__ref = get_length_ref(unit)
            elif isinstance(unit, (float, int)):
                self.__unit = 'user'
                self.__ref = float(unit)
            else:
                raise TypeError(
                    'Unsupported length unit type: ' + str(type(unit)))
            self.__abs_size = self.__value * self.__ref
            self.__unit_name = get_length_unit_name(self.__unit)
        else:
            raise TypeError(
                'Unsupported length value type: ' + str(type(value)))

    def change_unit(self, unit):
        """
        change unit.

        Args:
            unit (str): Energy unit.
        """
        if isinstance(unit, (str, Units)):
            self.__unit = get_length_unit(unit)
            self.__ref = get_length_ref(unit)
        elif isinstance(unit, (float, int)):
            self.__unit = 'user'
            self.__ref = unit
        else:
            raise TypeError('Unsupported length unit type: ' + str(type(unit)))
        self.__value = length_convert('nm', unit) * self.__abs_size
        self.__unit_name = get_length_unit_name(self.__unit)
        return self

    @property
    def abs_size(self):
        """
        absolute size of length.

        Returns:
            float, the absolute size of length.
        """
        return self.__abs_size

    @property
    def value(self):
        """
        value of length.

        Returns:
            float, the value of length.
        """
        return self.__value

    @property
    def ref(self):
        """
        reference value.

        Returns:
            float, a reference value.
        """
        return self.__ref

    @property
    def unit(self):
        """
        length unit.

        Returns:
            str, the length unit.
        """
        return self.__unit

    @property
    def unit_name(self):
        """
        name of length unit.

        Returns:
            str, the name of length unit.
        """
        return self.__unit_name

    def __call__(self, unit=None):
        return self.__value * length_convert(self.__unit, unit)

    def __str__(self):
        return str(self.__value) + ' ' + self.__unit_name

    def __lt__(self, other):
        if isinstance(other, Length):
            return self.__abs_size < other.abs_size
        return self.__value < other

    def __gt__(self, other):
        if isinstance(other, Length):
            return self.__abs_size > other.abs_size
        return self.__value > other

    def __eq__(self, other):
        if isinstance(other, Length):
            return self.__abs_size == other.abs_size
        return self.__value == other

    def __le__(self, other):
        if isinstance(other, Length):
            return self.__abs_size <= other.abs_size
        return self.__value <= other

    def __ge__(self, other):
        if isinstance(other, Length):
            return self.__abs_size >= other.abs_size
        return self.__value >= other


class Energy:
    """
    Energy.

    Args:
        value (float):   energy value.
        unit (str):      energy value unit. Default: 'kl/mol'

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, value: float, unit: str = 'kj/mol'):
        if isinstance(value, Energy):
            self.__value = value.value
            self.__unit = value.unit
            self.__ref = value.ref
            self.__abs_size = value.abs_size
            self.__unit_name = value.unit_name
        elif isinstance(value, (float, int)):
            self.__value = float(value)
            if isinstance(unit, (str, Units)):
                self.__unit = get_energy_unit(unit)
                self.__ref = get_energy_ref(unit)
            elif isinstance(unit, (float, int)):
                self.__unit = 'user'
                self.__ref = float(unit)
            else:
                raise TypeError('Unsupported energy unit type: ' + str(type(unit)))
            self.__abs_size = self.__value * self.__ref
            self.__unit_name = get_energy_unit_name(self.__unit)
        else:
            raise TypeError(
                'Unsupported energy value type: ' + str(type(value)))

    def change_unit(self, unit):
        """
        change unit.

        Args:
            unit (str): Energy unit.
        """
        if isinstance(unit, (str, Units)):
            self.__unit = get_energy_unit(unit)
            self.__ref = get_energy_ref(unit)
        elif isinstance(unit, (float, int)):
            self.__unit = 'user'
            self.__ref = unit
        else:
            raise TypeError('Unsupported energy unit type: ' + str(type(unit)))
        self.__value = energy_convert('kj/mol', unit) * self.__abs_size
        self.__unit_name = get_energy_unit_name(self.__unit)
        return self

    def __call__(self, unit=None):
        return self.__value * energy_convert(self.__unit, unit)

    def __str__(self):
        return str(self.__value) + ' ' + self.__unit_name

    def __lt__(self, other):
        if isinstance(other, Energy):
            return self.__abs_size < other.abs_size
        return self.__value < other

    def __gt__(self, other):
        if isinstance(other, Energy):
            return self.__abs_size > other.abs_size
        return self.__value > other

    def __eq__(self, other):
        if isinstance(other, Energy):
            return self.__abs_size == other.abs_size
        return self.__value == other

    def __le__(self, other):
        if isinstance(other, Energy):
            return self.__abs_size <= other.abs_size
        return self.__value <= other

    def __ge__(self, other):
        if isinstance(other, Energy):
            return self.__abs_size >= other.abs_size
        return self.__value >= other

    @property
    def abs_size(self):
        """
        absolute size of energy.

        Returns:
            float, the absolute size of energy.
        """
        return self.__abs_size

    @property
    def value(self):
        """
        value of energy.

        Returns:
            float, the value of energy.
        """
        return self.__value

    @property
    def ref(self):
        """
        reference value.

        Returns:
            float, the reference value of energy.
        """
        return self.__ref

    @property
    def unit(self):
        """
        energy unit.

        Returns:
            str, the unit of energy value.
        """
        return self.__unit

    @property
    def unit_name(self):
        """
        name of energy unit.

        Returns:
            str, the name of energy unit.
        """
        return self.__unit_name


def get_length_ref(unit):
    """
    get length reference.

    Args:
        unit (Union[str, Units, Length, float, int]):   Length unit.

    Returns:
        length reference(Union[str, float, int]).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if unit is None:
        return None
    if isinstance(unit, str):
        if unit.lower() not in _LENGTH_REF.keys():
            raise KeyError('length unit "' + unit + '" is not recorded!')
        return _LENGTH_REF.get(unit.lower())
    if isinstance(unit, Units):
        return unit.length_ref
    if isinstance(unit, Length):
        return unit.ref
    if isinstance(unit, (float, int)):
        return unit
    raise TypeError('Unsupported length reference type: ' + str(type(unit)))


def get_length_unit(unit):
    """
    get length unit.

    Args:
        unit (Union[str, Units, Length, float, int]):   Length unit.

    Returns:
        length unit(Union[str, float, int]).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if unit is None:
        return 'none'
    if isinstance(unit, str):
        if unit.lower() not in _LENGTH_UNITS:
            raise KeyError('length unit "' + unit + '" is not recorded!')
        return unit.lower()
    if isinstance(unit, Units):
        return unit.length_unit
    if isinstance(unit, Length):
        return unit.unit
    if isinstance(unit, (float, int)):
        return 'user'
    raise TypeError('Unsupported length unit type: ' + str(type(unit)))


def get_length_unit_name(unit):
    """
    get name of length unit.

    Args:
        unit (Union[str, Units, Length, float, int]):   Length unit.

    Returns:
        length unit(str).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if unit is None:
        return 'None'
    if isinstance(unit, str):
        if unit.lower() not in _LENGTH_NAME.keys():
            raise KeyError('length unit "' + unit + '" is not recorded!')
        return _LENGTH_NAME.get(unit.lower())
    if isinstance(unit, Units):
        return unit.length_unit_name
    if isinstance(unit, Length):
        return unit.unit_name
    if isinstance(unit, (float, int)):
        return 'User_Length'
    raise TypeError('Unsupported length unit name type: ' + str(type(unit)))


def get_energy_ref(unit):
    """
    get energy reference.

    Args:
        unit (Union[str, Units, Length, float, int]):   Energy unit.

    Returns:
        energy reference(Union[str, float, int]).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if unit is None:
        return None
    if isinstance(unit, str):
        if unit.lower() not in _ENERGY_REF.keys():
            raise KeyError('energy unit "' + unit + '" is not recorded!')
        return _ENERGY_REF.get(unit.lower())
    if isinstance(unit, Units):
        return unit.energy_ref
    if isinstance(unit, Energy):
        return unit.ref
    if isinstance(unit, (float, int)):
        return unit
    raise TypeError('Unsupported energy reference type: ' + str(type(unit)))


def get_energy_unit(unit):
    """
    get energy unit.

    Args:
        unit (Union[str, Units, Length, float, int]):   Energy unit.

    Returns:
        energy unit(str).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if unit is None:
        return 'none'
    if isinstance(unit, str):
        if unit.lower() not in _ENERGY_UNITS:
            raise KeyError('energy unit "' + unit + '" is not recorded!')
        return unit.lower()
    if isinstance(unit, Units):
        return unit.energy_unit
    if isinstance(unit, Energy):
        return unit.unit
    if isinstance(unit, (float, int)):
        return 'user'
    raise TypeError('Unsupported energy unit type: ' + str(type(unit)))


def get_energy_unit_name(unit):
    """
    get the name of energy unit.

    Args:
        unit (Union[str, Units, Length, float, int]):   Energy unit.

    Returns:
        name of energy unit(str).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if unit is None:
        return 'None'
    if isinstance(unit, str):
        if unit.lower() not in _ENERGY_NAME.keys():
            raise KeyError('energy unit "' + unit + '" is not recorded!')
        return _ENERGY_NAME.get(unit.lower())
    if isinstance(unit, Units):
        return unit.energy_unit_name
    if isinstance(unit, Energy):
        return unit.unit_name
    if isinstance(unit, (float, int)):
        return 'User_Energy'
    raise TypeError('Unsupported energy unit name type: ' + str(type(unit)))


def length_convert(unit_in, unit_out):
    """
    convert length according to different units.

    Args:
        unit_in (Union[str, Units, Length, float, int]):    input unit of length.
        unit_out (Union[str, Units, Length, float, int]):   output unit of length.

    Returns:
        float, length according to different units.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    length_in = get_length_ref(unit_in)
    length_out = get_length_ref(unit_out)
    if length_in is None or length_out is None:
        return 1
    return length_in / length_out


def energy_convert(unit_in, unit_out):
    """
    convert energy according to difference units.

    Args:
        unit_in (Union[str, Units, Length, float, int]):    input unit of length.
        unit_out (Union[str, Units, Length, float, int]):   output unit of length.

    Returns:
        float, energy according to different units.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    energy_in = get_energy_ref(unit_in)
    energy_out = get_energy_ref(unit_out)
    if energy_in is None or energy_out is None:
        return 1
    return energy_in / energy_out


class Units:
    r"""
    Unit class to record and convert the length and energy units.

    Args:
        length_unit (str):  Length unit. Default: None
        energy_unit (str):  Energy unit. Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):

        self.__length_unit = get_length_unit(length_unit)
        self.__length_unit_name = get_length_unit_name(length_unit)
        self.__length_ref = get_length_ref(length_unit)

        self.__energy_unit = get_energy_unit(energy_unit)
        self.__energy_unit_name = get_energy_unit_name(energy_unit)
        self.__energy_ref = get_energy_ref(energy_unit)

        self.__boltzmann = _BOLTZMANN_DEFAULT_REF
        if self.__energy_ref is not None:
            self.__boltzmann /= self.__energy_ref
        self.__coulomb = _COULOMB_DEFAULT_REF
        if self.__length_ref is not None and self.__energy_ref is not None:
            self.__coulomb /= self.__energy_ref * self.__length_ref

        self.time_unit = 'ps'

    def set_length_unit(self, unit=None):
        """
        set length unit.

        Args:
            unit (str): Length unit.
        """
        if unit is not None:
            self.__length_unit = get_length_unit(unit)
            self.__length_unit_name = get_length_unit_name(unit)
            self.__length_ref = get_length_ref(unit)
            self._set_constants()
        return self

    def set_energy_unit(self, unit=None):
        """
        set energy unit.

        Args:
            unit (str): Energy unit.
        """
        if unit is not None:
            self.__energy_unit = get_energy_unit(unit)
            self.__energy_unit_name = get_energy_unit_name(unit)
            self.__energy_ref = get_energy_ref(unit)
            self._set_constants()
        return self

    def set_units(self, length_unit, energy_unit, units=None):
        """
        set unit.

        Args:
            length_unit (str):  Length unit.
            energy_unit (str):  Energy unit.
            units (str):        Units.
        """
        if units is None:
            if length_unit is not None:
                self.__length_unit = get_length_unit(length_unit)
                self.__length_unit_name = get_length_unit_name(length_unit)
                self.__length_ref = get_length_ref(length_unit)
            if energy_unit is not None:
                self.__energy_unit = get_energy_unit(energy_unit)
                self.__energy_unit_name = get_energy_unit_name(energy_unit)
                self.__energy_ref = get_energy_ref(energy_unit)
        else:
            if not isinstance(units, Units):
                raise TypeError('The type of units must be "Units"')
            self.__length_unit = get_length_unit(units)
            self.__length_unit_name = get_length_unit_name(units)
            self.__length_ref = get_length_ref(units)
            self.__energy_unit = get_energy_unit(units)
            self.__energy_unit_name = get_energy_unit_name(units)
            self.__energy_ref = get_energy_ref(units)
        return self._set_constants()

    def _set_constants(self):
        """set constant values"""
        self.__boltzmann = _BOLTZMANN_DEFAULT_REF
        if self.__energy_ref is not None:
            self.__boltzmann /= self.__energy_ref
        self.__coulomb = _COULOMB_DEFAULT_REF
        if self.__length_ref is not None and self.__energy_ref is not None:
            self.__coulomb /= self.__energy_ref * self.__length_ref
        return self

    def length(self, value, unit=None):
        """
        return the length value of the specified unit.

        Args:
            value (float):  Length value.
            unit (str):     Length unit.

        Returns:
            float, the length value.
        """
        return value * self.convert_length_from(unit)

    def energy(self, value, unit=None):
        """
        return the energy value of the specified unit.

        Args:
            value (float):  Energy value.
            unit (str):     Energy unit.

        Returns:
            float, the energy value.
        """
        return value * self.convert_energy_from(unit)

    def convert_length_to(self, unit):
        """
        convert length to a specified units.

        Args:
            unit (str): Length unit.

        Returns:
            float, length according to a specified units.
        """
        return length_convert(self.__length_unit, unit)

    def convert_energy_to(self, unit):
        """
        convert energy to a specified units.

        Args:
            unit (str): Energy unit.

        Returns:
            float, energy according to a specified units.
        """
        return energy_convert(self.__energy_unit, unit)

    def convert_length_from(self, unit):
        """convert length from a specified units.

        Args:
            unit (str): Length unit.

        Returns:
            float, length according from a specified units.
        """
        return length_convert(unit, self.__length_unit)

    def convert_energy_from(self, unit):
        """
        convert energy from a specified units.

        Args:
            unit (str): Energy unit.

        Returns:
            float, energy according from a specified units.
        """
        return energy_convert(unit, self.__energy_unit)

    @property
    def boltzmann_def(self):
        """
        Boltzmann constant in kJ/mol.

        Returns:
            float, Boltzmann constant in kJ/mol.
        """
        return _BOLTZMANN_DEFAULT_REF

    @property
    def boltzmann(self):
        """
        Boltzmann constant in current unit.

        Returns:
            float, Boltzmann constant in current unit.
        """
        return self.__boltzmann

    @property
    def coulomb(self):
        """
        Coulomb constant in current unit.

        Returns:
            float, Coulomb constant in current unit.
        """
        return self.__coulomb

    @property
    def avogadro(self):
        """
        Avogadro number.

        Returns:
            float, Avogadro number.
        """
        return AVOGADRO_NUMBER

    @property
    def gas_constant(self):
        """
        gas constant.

        Returns:
            float, gas constant.
        """
        return GAS_CONSTANT

    @property
    def length_unit(self):
        """
        length unit.

        Returns:
            length unit (Union[str, float, int]).
        """
        return self.__length_unit

    @property
    def energy_unit(self):
        """
        energy unit.

        Returns:
            energy unit (Union[str, float, int]).
        """
        return self.__energy_unit

    @property
    def length_unit_name(self):
        """
        name of length unit.

        Returns:
            str, name of length unit.
        """
        return self.__length_unit_name

    @property
    def energy_unit_name(self):
        """
        name of energy unit.

        Returns:
            str, name of energy unit.
        """
        return self.__energy_unit_name

    @property
    def volume_unit(self):
        """
        velocity unit.

        Returns:
            str, velocity unit.
        """
        return self.__length_unit + "^3"

    @property
    def volume_unit_name(self):
        """
        velocity unit name.

        Returns:
            str, velocity unit name.
        """
        return self.__length_unit + "+3"

    @property
    def force_unit(self):
        """
        force unit.

        Returns:
            str, force unit.
        """
        return self.__energy_unit + '/' + self.__length_unit

    @property
    def force_unit_name(self):
        """
        name of force unit.

        Returns:
            str, name of force unit.
        """
        return self.__energy_unit_name + ' ' + self.__length_unit_name + '-1'

    @property
    def velocity_unit(self):
        """
        velocity unit.

        Returns:
            str, velocity unit.
        """
        return self.__length_unit + "/" + self.time_unit

    @property
    def velocity_unit_name(self):
        """
        name of velocity unit.

        Returns:
            str, name of velocity unit.
        """
        return self.__length_unit_name + ' ' + self.time_unit + '-1'

    @property
    def length_ref(self):
        """
        reference value of length.

        Returns:
            float, reference value of length.
        """
        return self.__length_ref

    @property
    def energy_ref(self):
        """
        reference value of energy.

        Returns:
            float, reference value of energy.
        """
        return self.__energy_ref

    @property
    def force_ref(self):
        """
        reference value of force.

        Returns:
            float, reference value of force.
        """
        if self.__energy_ref is None:
            return None
        return self.__energy_ref / self.__length_ref

    @property
    def acceleration_ref(self):
        """
        reference value of acceleration.

        Returns:
            float, reference value of acceleration.
        """
        if self.__energy_ref is None or self.__length_ref is None:
            return None
        return self.__energy_ref / self.__length_ref / self.__length_ref

    @property
    def kinetic_ref(self):
        """
        reference value of kinetic.

        Returns:
            float, reference value of kinetic.
        """
        if self.__energy_ref is None or self.__length_ref is None:
            return None
        return self.__length_ref * self.__length_ref / self.__energy_ref

    @property
    def pressure_ref(self):
        if self.__energy_ref is None or self.__length_ref is None:
            return None
        return _BAR_DEFAULT_REF * self.__energy_ref / math.pow(self.__length_ref, 3)


global_units = Units('nm', 'kj/mol')


def set_global_length_unit(unit):
    """
    set length unit for global_units.

    Args:
        unit (str):  Length unit. Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    global_units.set_length_unit(unit)


def set_global_energy_unit(unit):
    """
    set energy unit for global_units.

    Args:
        unit (str):  Energy unit. Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    global_units.set_energy_unit(unit)


def set_global_units(length_unit, energy_unit, units=None):
    """
    set units for global_units.

    Args:
        length_unit (str):  Length unit. Default: None
        energy_unit (str):  Energy unit. Default: None
        units (str):        units. Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    global_units.set_units(length_unit, energy_unit, units)

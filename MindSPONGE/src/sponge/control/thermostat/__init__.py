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
"""Thermostat"""

from typing import Union
from ...system import Molecule

from .thermostat import Thermostat, _THERMOSTAT_BY_KEY
from .berendsen import BerendsenThermostat
from .langevin import Langevin

__all__ = ['Thermostat', 'BerendsenThermostat', 'Langevin', 'get_thermostat']

_THERMOSTAT_BY_NAME = {cell.__name__: cell for cell in _THERMOSTAT_BY_KEY.values()}


def get_thermostat(cls_name: Union[str, dict, Thermostat],
                   system: Molecule,
                   temperature: float = None,
                   **kwargs) -> Thermostat:
    r"""get object of thermostat.

    Args:
        cls_name (Union[str, dict, :class:`sponge.control.Thermostat`]): Class name,
          arguments or object of a thermostat.
        system (:class:`sponge.system.Molecule`): Simulation system.
        temperature (float, optional): Reference temperature for temperature coupling.
          If `None` is given and the type of `cls_name` is `str`, `None` will be returned.
          Default: ``None``.
        **kwargs: Other arguments.

    Returns:
        :class:`sponge.control.Thermostat`, object of thermostat.
    """

    if cls_name is None:
        return None

    if isinstance(cls_name, Thermostat):
        return cls_name

    if isinstance(cls_name, dict):
        return get_thermostat(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None

        if temperature is None:
            return None

        if cls_name.lower() in _THERMOSTAT_BY_KEY.keys():
            return _THERMOSTAT_BY_KEY.get(cls_name.lower())(system=system,
                                                            temperature=temperature,
                                                            **kwargs)
        if cls_name in _THERMOSTAT_BY_NAME.keys():
            return _THERMOSTAT_BY_NAME.get(cls_name.lower())(system=system,
                                                             temperature=temperature,
                                                             **kwargs)

        raise ValueError("The thermostat corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported thermostat type '{}'.".format(type(cls_name)))

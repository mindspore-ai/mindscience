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
"""Barostat"""

from typing import Union
from ...system import Molecule

from .barostat import Barostat, _BAROSTAT_BY_KEY
from .berendsen import BerendsenBarostat
from .andersen import AndersenBarostat

__all__ = ['Barostat', 'BerendsenBarostat', 'AndersenBarostat', 'get_barostat']

_BAROSTAT_BY_NAME = {cell.__name__: cell for cell in _BAROSTAT_BY_KEY.values()}


def get_barostat(cls_name: Union[str, dict, Barostat],
                 system: Molecule,
                 pressure: float = None,
                 **kwargs) -> Barostat:
    r"""
    Get object of barostat.

    Args:
        cls_name (Union[str, dict, :class:`sponge.control.Barostat`]): Class name,
          arguments or object of a barostat.
        system ( :class:`sponge.system.Molecule`): Simulation system.
        pressure (float): Reference pressure for pressure coupling. If `None` is given and
          the type of `cls_name` is `str`, `None` will be returned. Default: ``None``.
        **kwargs: Other arguments.

    Returns:
        :class:`sponge.control.Barostat`, object of barostat.
    """

    if cls_name is None:
        return None

    if isinstance(cls_name, Barostat):
        return cls_name

    if isinstance(cls_name, dict):
        return get_barostat(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None

        if pressure is None:
            return None

        if cls_name.lower() in _BAROSTAT_BY_KEY.keys():
            return _BAROSTAT_BY_KEY.get(cls_name.lower())(system=system,
                                                          pressure=pressure,
                                                          **kwargs)
        if cls_name in _BAROSTAT_BY_NAME.keys():
            return _BAROSTAT_BY_NAME.get(cls_name.lower())(system=system,
                                                           pressure=pressure,
                                                           **kwargs)

        raise ValueError("The barostat corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported barostat type '{}'.".format(type(cls_name)))

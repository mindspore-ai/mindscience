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
"""Simulation integrator"""

from typing import Union
from ...system import Molecule

from .integrator import Integrator, _INTEGRATOR_BY_KEY
from .leapfrog import LeapFrog
from .velocityverlet import VelocityVerlet
from .brownian import Brownian

__all__ = ['Integrator', 'LeapFrog', 'VelocityVerlet', 'Brownian', 'get_integrator']

_INTEGRATOR_BY_NAME = {cell.__name__: cell for cell in _INTEGRATOR_BY_KEY.values()}


def get_integrator(cls_name: Union[str, dict, Integrator],
                   system: Molecule,
                   **kwargs) -> Integrator:
    r"""
    Get an object of an integrator.

    Args:
        cls_name (Union[str, dict, :class:`sponge.control.Integrator`]): Class name,
          arguments or object of a integrator
        system (:class:`sponge.system.Molecule`): Simulation system.
        **kwargs: Other arguments.

    Returns:
        :class:`sponge.control.Integrator`, object of integrator.
    """

    if cls_name is None:
        return None

    if isinstance(cls_name, Integrator):
        return cls_name

    if isinstance(cls_name, dict):
        return get_integrator(**cls_name)

    if isinstance(cls_name, str):
        if cls_name.lower() == 'none':
            return None

        #pylint: disable=invalid-name
        if cls_name.lower() in _INTEGRATOR_BY_KEY.keys():
            return _INTEGRATOR_BY_KEY.get(cls_name.lower())(system=system, **kwargs)
        if cls_name in _INTEGRATOR_BY_NAME.keys():
            return _INTEGRATOR_BY_NAME.get(cls_name.lower())(system=system, **kwargs)

        raise ValueError("The integrator corresponding to '{}' was not found.".format(cls_name))

    raise TypeError("Unsupported integrator type '{}'.".format(type(cls_name)))

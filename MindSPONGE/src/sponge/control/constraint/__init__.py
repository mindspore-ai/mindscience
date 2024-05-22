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
"""constraint"""

from typing import Union, List
from ...system import Molecule

from .constraint import Constraint
from .lincs import Lincs
from .settle import SETTLE

__all__ = ['Constraint', 'Lincs', 'SETTLE', 'get_constraint']


def get_constraint(constraint: Union[str, Constraint, List[Constraint]], system: Molecule):
    r"""
    Get constraint object.

    Args:
        constraint (Union[str, :class:`sponge.control.Constraint`, List[:class:`sponge.control.Constraint`]]):
            constraint name, `Constraint` object or list of `Constraint` objects.
        system (:class:`sponge.system.Molecule`): Simulation system.

    Returns:
        List[:class:`sponge.control.Constraint`], constraint object list.
    """

    if constraint is None:
        return None

    if isinstance(constraint, Constraint):
        return [constraint]

    if isinstance(constraint, list):
        for c in constraint:
            if not isinstance(c, Constraint):
                raise TypeError(f'The elements in list must be Constraint but got: '
                                f'{type((c))}')
        return constraint

    settle = None
    if system.force_settle:
        settle = SETTLE(system)

    if isinstance(constraint, str):
        if constraint.lower() in ['all-bonds', 'h-bonds']:
            if system.settle_index is not None and settle is None:
                settle = SETTLE(system)
            if system.remaining_index is None:
                constraint = []
            else:
                constraint = [Lincs(system, bonds=constraint.lower())]

        elif constraint.lower() == 'none':
            constraint = []
        else:
            raise ValueError(f'Inputs of type `str` as `constraint` can only be "none", "all-bonds" or "h-bonds", '
                             f'but got: {constraint}')

        if settle is not None:
            constraint.append(settle)

    if constraint:
        return constraint

    return None

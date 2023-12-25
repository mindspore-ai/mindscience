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
Atoms and virtual atoms
"""

from typing import Union, List, Tuple
import numpy as np
from numpy import ndarray
from mindspore import Tensor, Parameter

from .atoms import AtomsBase, Atoms
from .group import Group


def get_atoms(atoms: Union[AtomsBase, List[AtomsBase], Tuple[AtomsBase], Tensor, Parameter, ndarray],
              batched: bool = False,
              keep_in_box: bool = False,
              ) -> AtomsBase:
    r"""
    get atom(s) or group.

    Args:
        atoms (Union[list, tuple, AtomsBase, Tensor, Parameter, ndarray]):
                            List of atoms.

        batched (bool):     Whether the first dimension of index is the batch size.
                            Default: ``False``.

        keep_in_box (bool): Whether to displace the coordinate in PBC box.
                            Default: ``False``.

    Returns:
        atoms (Union[Atoms, Group]), Atoms or group.

    """
    #pylint: disable=bare-except
    if atoms is None:
        return None
    if isinstance(atoms, AtomsBase):
        return atoms
    if isinstance(atoms, (Tensor, Parameter, ndarray)):
        return Atoms(atoms, batched, keep_in_box)
    if isinstance(atoms, (list, tuple)):
        def _convert_array(atoms):
            if not isinstance(atoms, (list, tuple)):
                return atoms
            try:
                atoms = np.array(atoms, np.int32)
            except ValueError:
                if set(map(type, atoms)) in ({list}, {tuple}):
                    atoms = [_convert_array(a) for a in atoms]
            return atoms

        atoms = _convert_array(atoms)
        if isinstance(atoms, ndarray):
            return Atoms(atoms, batched, keep_in_box)

        if isinstance(atoms, list):
            return Group(atoms, batched, keep_in_box)

    raise TypeError(f'The type of "atoms" must be list, tuple or AtomsBase, '
                    f'but got: {type(atoms)}')

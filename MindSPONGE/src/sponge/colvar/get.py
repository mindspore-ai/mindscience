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
Atoms and virtual colvar
"""

from typing import Union, List, Tuple

from .colvar import Colvar
from .group import ColvarGroup


def get_colvar(colvar_: Union[Colvar, List[Colvar], Tuple[Colvar]],
               axis: int = -1,
               use_pbc: bool = None,
               name: str = None,
               ) -> Colvar:
    r"""
    get group of collective variables.

    Args:
        colvar (Union[Colvar, List[Colvar], Tuple[Colvar]]): Colvar or array of colvars.

        axis (int): Axis to be concatenated. Default: -1.

        use_pbc (bool): Whether to use periodic boundary condition. Default: ``None``.

        name (str): Name. Default: ``None``.

    Returns:
        colvar (Union[Atoms, Group]), Atoms or group.

    """
    if colvar_ is None:
        return None

    if isinstance(colvar_, (list, tuple)):
        colvar_ = ColvarGroup(colvar_, axis=axis, use_pbc=use_pbc)

    if not isinstance(colvar_, Colvar):
        raise TypeError(f'The type of "colvar" must be list, tuple or Colvar but got: {type(colvar_)}')

    if name is not None:
        colvar_.set_name(name)

    return colvar_

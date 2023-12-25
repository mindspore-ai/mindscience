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
Sine of Colvar
"""

from mindspore.ops.operations import Sin

from .transform import TransformCV
from ..colvar import Colvar


class SinCV(TransformCV):
    r"""
    Sine of collective variables (CVs) :math:`s(R)`. The return value has the same shape as the input CVs.

    .. math::

        s' = \sin{s(R)}

    Args:
        colvar (Colvar): Collective variables (CVs) :math:`s(R)`.

        name (str): Name of the collective variables. Default: 'cosine'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Colvar,
                 name: str = 'sine',
                 ):

        super().__init__(
            colvar=colvar,
            function=Sin(),
            periodic=False,
            unit=None,
            name=name,
        )

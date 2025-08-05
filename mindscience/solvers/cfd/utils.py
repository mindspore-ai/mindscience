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
"""utils of MindFlow CFD."""
from mindspore import numpy as mnp
from mindspore import ops

__all__ = ['cal_con_var', 'cal_pri_var', 'cal_flux']


def cal_con_var(pri_var, material):
    """
    Calculate conservative variables from primitive variables and material.

    Args:
        pri_var (Tensor): The primitive variables.
        material (mindflow.cfd.Material): Material of the fluid.

    Returns:
        Tensor, with the same shape as `pri_var`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindflow.utils import cal_con_var
        >>> from mindflow import material
        >>> config =  {'type': 'IdealGas', 'heat_ratio': 1.4, 'specific_heat_ratio': 1.4, 'specific_gas_constant': 1.0}
        >>> m = material.select_material(config)
        >>> x1 = Tensor(np.random.randn(5, 32, 32, 1))
        >>> x2 = cal_con_var(x, m)
        >>> x2.shape
        (5, 32, 32, 1)
    """
    rho = pri_var[0, ...]
    rho_ux = pri_var[0, ...] * pri_var[1, ...]
    rho_uy = pri_var[0, ...] * pri_var[2, ...]
    rho_uz = pri_var[0, ...] * pri_var[3, ...]
    total_energy = material.total_energy(pri_var)
    con_var = mnp.stack([rho, rho_ux, rho_uy, rho_uz, total_energy], axis=0)
    return con_var


def cal_pri_var(con_var, material):
    """
    Calculate primitive variables from conservative variables and material.

    Args:
        con_var (Tensor): The conservative variables.
        material (mindflow.cfd.Material): Material of the fluid.

    Returns:
        Tensor, with the same shape as `con_var`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindflow.utils import cal_pri_var
        >>> from mindflow import material
        >>> config =  {'type': 'IdealGas', 'heat_ratio': 1.4, 'specific_heat_ratio': 1.4, 'specific_gas_constant': 1.0}
        >>> m = material.select_material(config)
        >>> x1 = Tensor(np.random.randn(5, 32, 32, 1))
        >>> x2 = cal_pri_var(x, m)
        >>> x2.shape
        (5, 32, 32, 1)
    """
    rho = con_var[0, ...]
    eps = ops.Eps()
    u_x = con_var[1, ...] / (con_var[0, ...] + eps(rho))
    u_y = con_var[2, ...] / (con_var[0, ...] + eps(rho))
    u_z = con_var[3, ...] / (con_var[0, ...] + eps(rho))
    internal_energy = con_var[4] / (con_var[0, ...] + eps(rho)) - 0.5 * (u_x ** 2 + u_y ** 2 + u_z ** 2)
    pressure = (material.gamma - 1) * internal_energy * rho
    pri_var = mnp.stack([rho, u_x, u_y, u_z, pressure], axis=0)
    return pri_var


def cal_flux(con_var, pri_var, axis):
    """
    Calculate flux from primitive variables and conservative variables.

    Args:
        con_var (Tensor): The conservative variables.
        pri_var (Tensor): The primitive variables.
        axis (int): Axis of the flux.

    Supported Platforms:
        ``GPU``

    Returns:
        Tensor, with the same shape as `pri_var`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindflow.utils import cal_pri_var, cal_flux
        >>> from mindflow import material
        >>> config =  {'type': 'IdealGas', 'heat_ratio': 1.4, 'specific_heat_ratio': 1.4, 'specific_gas_constant': 1.0}
        >>> m = material.select_material(config)
        >>> x1 = Tensor(np.random.randn(5, 32, 32, 1))
        >>> x2 = cal_pri_var(x, m)
        >>> y = cal_flux(x1, x2, 0)
        >>> y.shape
        (5, 32, 32, 1)
    """
    rho_ui = con_var[axis + 1, ...]
    rho_ui_ux = con_var[axis + 1, ...] * pri_var[1, ...]
    rho_ui_uy = con_var[axis + 1, ...] * pri_var[2, ...]
    rho_ui_uz = con_var[axis + 1, ...] * pri_var[3, ...]
    ui_ep = pri_var[axis + 1, ...] * (con_var[4, ...] + pri_var[4, ...])
    if axis == 0:
        rho_ui_ux += pri_var[4, ...]
    elif axis == 1:
        rho_ui_uy += pri_var[4, ...]
    elif axis == 2:
        rho_ui_uz += pri_var[4, ...]
    flux = mnp.stack([rho_ui, rho_ui_ux, rho_ui_uy, rho_ui_uz, ui_ep], axis=0)
    return flux

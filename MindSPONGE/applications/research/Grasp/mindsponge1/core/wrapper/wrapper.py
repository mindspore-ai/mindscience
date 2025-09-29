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
"""Energy wrapper"""

from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell

from ...function import get_integer

_ENERGY_WRAPPER_BY_KEY = dict()


def _energy_wrapper_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _ENERGY_WRAPPER_BY_KEY:
            _ENERGY_WRAPPER_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _ENERGY_WRAPPER_BY_KEY:
                _ENERGY_WRAPPER_BY_KEY[alias] = cls

        return cls

    return alias_reg


class EnergyWrapper(Cell):
    r"""A network to process and merge the potential and bias during the simulation.

    Args:

        num_walker (int):       Number of multiple walker (B). Default: 1

        dim_potential (int):    Dimension of potential energy (U). Default: 1

        dim_bias (int):         Dimension of bias potential (V). Default: 1

    """
    def __init__(self,
                 num_walker: int = 1,
                 dim_potential: int = 1,
                 dim_bias: int = 1,
                 ):

        super().__init__(auto_prefix=False)

        self.num_walker = get_integer(num_walker)
        self.dim_potential = get_integer(dim_potential)
        self.dim_bias = get_integer(dim_bias)

        self.concat_last_dim = ops.Concat(-1)
        self.sum_last_dim = ops.ReduceSum(keep_dims=True)

    def construct(self, potential: Tensor, bias: Tensor = None):
        """merge the potential and bias.

        Args:
            potential (Tensor): Tensor of shape (B, U). Data type is float.
                                Potential energy.
            bias (Tensor):      Tensor of shape (B, V). Data type is float.
                                Bias potential. Default: None

        Return:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Total energy.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation.
            U:  Dimension of potential energy.
            V:  Dimension of bias potential.

        """
        raise NotImplementedError


def get_energy_wrapper(wrapper: str,
                       num_walker: int,
                       dim_potential: int,
                       dim_bias: int,
                       ) -> EnergyWrapper:
    """get energy wrapper by name"""
    if wrapper is None or isinstance(wrapper, EnergyWrapper):
        return wrapper
    if isinstance(wrapper, str):
        if wrapper.lower() == 'none':
            return None
        if wrapper.lower() in _ENERGY_WRAPPER_BY_KEY.keys():
            return _ENERGY_WRAPPER_BY_KEY.get(wrapper.lower())(
                num_walker=num_walker,
                dim_potential=dim_potential,
                dim_bias=dim_bias,
                )
        raise ValueError(
            "The energy wrapper corresponding to '{}' was not found.".format(wrapper))
    raise TypeError(
        "Unsupported energy wrapper type '{}'.".format(type(wrapper)))

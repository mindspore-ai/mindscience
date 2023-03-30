# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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
Radical basis functions (RBF)
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from mindsponge.data import get_hyper_string, get_hyper_parameter
from mindsponge.data import set_hyper_parameter
from mindsponge.data import set_class_into_hyper_param
from mindsponge.function import get_integer
from mindsponge.function import Units, Length

__all__ = [
    "GaussianBasis",
    "LogGaussianBasis",
    "get_rbf",
]

_RBF_BY_KEY = dict()


def _rbf_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _RBF_BY_KEY:
            _RBF_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _RBF_BY_KEY:
                _RBF_BY_KEY[alias] = cls
        return cls
    return alias_reg


class RadicalBasisFunctions(Cell):
    r"""Network of radical basis functions.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0 nm

        sigma (float):          Simga. Default: 0

        delta (float):          Space interval. Default: None

        num_basis (int):        Number of basis functions. Defatul: None

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: False

        clip_distance (bool):   Whether to clip the value of distance. Default: False

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: None

    """
    def __init__(self,
                 r_max: Length = 1,
                 r_min: Length = 0,
                 sigma: float = 0,
                 delta: float = None,
                 num_basis: int = None,
                 rescale: bool = False,
                 clip_distance: bool = False,
                 length_unit: str = 'nm',
                 hyper_param: dict = None,
                 ):

        super().__init__()

        if hyper_param is not None:
            r_max = get_hyper_parameter(hyper_param, 'r_max')
            r_min = get_hyper_parameter(hyper_param, 'r_min')
            sigma = get_hyper_parameter(hyper_param, 'sigma')
            delta = get_hyper_parameter(hyper_param, 'delta')
            num_basis = get_hyper_parameter(hyper_param, 'num_basis')
            clip_distance = get_hyper_parameter(hyper_param, 'clip_distance')
            rescale = get_hyper_parameter(hyper_param, 'rescale')
            length_unit = get_hyper_string(hyper_param, 'length_unit')

        self.units = Units(length_unit)
        self.r_max = self.get_length(r_max)
        self.r_min = self.get_length(r_min)
        self.sigma = sigma
        self.delta = delta
        self.num_basis = get_integer(num_basis)
        self.clip_distance = Tensor(clip_distance, ms.bool_)
        self.rescale = Tensor(rescale, ms.bool_)

        self.length_unit = self.units.length_unit

        self.check_range()
        self.check_basis(num_basis, delta)
        self.r_range = self.r_max - self.r_min
        self.offsets = None

        self.hyper_param = dict()
        self.hyper_types = {
            'r_max': 'float',
            'r_min': 'float',
            'sigma': 'float',
            'delta': 'float',
            'num_basis': 'int',
            'clip_distance': 'bool',
            'rescale': 'bool',
            'length_unit': 'str',
        }

    def set_hyper_param(self):
        """set hyperparameter"""
        set_hyper_parameter(self.hyper_param, 'name', self.cls_name)
        set_class_into_hyper_param(self.hyper_param, self.hyper_types, self)
        return self

    def check_basis(self, num_basis, delta):
        """check basis functions"""
        if num_basis is None and delta is None:
            raise TypeError('"num_basis" and "delta" cannot both be "None".')
        if num_basis is not None and num_basis <= 0:
            raise ValueError('"num_basis" must be larger than 0.')
        if delta is not None and delta <= 0:
            raise ValueError('"delta" must be larger than 0.')
        return self

    def check_range(self):
        """check range of distance"""
        if self.r_max <= self.r_min:
            raise ValueError('The argument "r_max" must be larger ' +
                             'than the argument "r_min" in RBF!')
        return self

    def set_rmax(self, r_max):
        """set minimum distance"""
        self.r_max = self.get_length(r_max)
        self.check_range()
        self.r_range = self.r_max - self.r_min
        set_hyper_parameter(self.hyper_param, 'r_max', self.r_max)
        return self

    def set_rmin(self, r_min):
        """set minimum distance"""
        self.r_min = self.get_length(r_min)
        self.check_range()
        self.r_range = self.r_max - self.r_min
        set_hyper_parameter(self.hyper_param, 'r_min', self.r_min)
        return self

    def set_range(self, r_min, r_max):
        """set range of distance"""
        self.r_max = self.get_length(r_max)
        self.r_min = self.get_length(r_min)
        self.check_range()
        self.r_range = self.r_max - self.r_min
        set_hyper_parameter(self.hyper_param, 'r_max', self.r_max)
        set_hyper_parameter(self.hyper_param, 'r_min', self.r_min)
        return self

    def set_sigma(self, sigma):
        """set sigma"""
        self.sigma = sigma
        set_hyper_parameter(self.hyper_param, 'sigma', self.sigma)
        return self

    def set_basis(self, num_basis=None, delta=None):
        """set number of basis function"""
        if num_basis is None and delta is None:
            raise TypeError('"num_basis" and "delta" cannot both be "None".')
        return self

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        """print the information of RBF"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Minimum distance: ' +
              str(self.r_min)+' '+self.units.length_unit)
        print(ret+gap+' Maximum distance: ' +
              str(self.r_max)+' '+self.units.length_unit)
        print(ret+gap+' Number of basis functions: ' + str(self.num_basis))
        print(ret+gap+' Interval: ' + str(self.delta))
        print(ret+gap+' Sigma: ' + str(self.sigma))
        if self.clip_distance:
            print(ret+gap+' Clip the range of distance to (r_min,r_max).')
        if self.rescale:
            print(ret+gap+' Rescale the range of RBF to (-1,1).')
        return self

    def get_length(self, length, unit=None):
        """get length value"""
        if isinstance(length, Length):
            if unit is None:
                unit = self.units
            return length(unit)

        return Tensor(length, ms.float32)

    def change_unit(self, unit):
        """change unit"""
        scale = self.units.convert_length_to(unit)
        self.r_min *= scale
        self.r_max *= scale
        self.r_range *= scale
        self.units.set_length_unit(unit)
        return scale

    def construct(self, distance: Tensor):
        raise NotImplementedError


@_rbf_register('gaussian')
class GaussianBasis(RadicalBasisFunctions):
    r"""Gaussian type RBF.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0 nm

        sigma (float):          Simga. Default: 0.03 nm

        delta (float):          Space interval. Default: 0.016 nm

        num_basis (int):        Number of basis functions. Defatul: None

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: False

        clip_distance (bool):   Whether to clip the value of distance. Default: False

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: None

    """

    def __init__(self,
                 r_max: Length = Length(1, 'nm'),
                 r_min: Length = Length(0, 'nm'),
                 sigma: Length = Length(0.03, 'nm'),
                 delta: Length = Length(0.016, 'nm'),
                 num_basis: int = None,
                 rescale: bool = False,
                 clip_distance: bool = False,
                 length_unit: str = 'nm',
                 hyper_param: dict = None,
                 ):

        super().__init__(
            r_min=r_min,
            r_max=r_max,
            sigma=sigma,
            delta=delta,
            num_basis=num_basis,
            rescale=rescale,
            clip_distance=clip_distance,
            length_unit=length_unit,
            hyper_param=hyper_param,
        )

        self.reg_key = 'gaussian'

        self.sigma = self.get_length(self.sigma)
        self.delta = self.get_length(self.delta)

        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))

        if self.delta is None:
            self.offsets = msnp.linspace(
                self.r_min, self.r_max, self.num_basis, dtype=ms.float32)
            self.delta = Tensor(self.r_range/(self.num_basis-1), ms.float32)
        else:
            if self.num_basis is None:
                num_basis = msnp.ceil(self.r_range/self.delta, ms.int32) + 1
                self.num_basis = get_integer(num_basis)
            self.offsets = self.r_min + \
                msnp.arange(0, self.num_basis) * self.delta

        self.set_hyper_param()

        self.exp = P.Exp()

    def set_sigma(self, sigma):
        self.sigma = self.get_length(sigma)
        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))
        set_hyper_parameter(self.hyper_param, 'sigma', self.sigma)
        return self

    def set_basis(self, num_basis=None, delta=None):
        self.check_basis(num_basis, delta)
        if delta is None:
            self.num_basis = get_integer(num_basis)
            self.offsets = msnp.linspace(
                self.r_min, self.r_max, self.num_basis, dtype=ms.float32)
            self.delta = Tensor(self.r_range/(self.num_basis-1), ms.float32)
        else:
            self.delta = self.get_length(delta)
            if num_basis is None:
                num_basis = msnp.ceil(self.r_range/self.delta, ms.int32) + 1
                self.num_basis = get_integer(num_basis)
            self.offsets = self.r_min + \
                msnp.arange(0, self.num_basis) * self.delta
        set_hyper_parameter(self.hyper_param, 'delta', self.delta)
        set_hyper_parameter(self.hyper_param, 'num_basis', self.num_basis)
        return self

    def change_unit(self, unit):
        scale = super().change_unit(unit)
        self.sigma *= scale
        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))
        self.delta *= scale
        self.offsets = self.r_min + msnp.arange(0, self.num_basis) * self.delta
        self.set_hyper_param()
        return self

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Minimum distance: ' +
              str(self.r_min)+' '+self.units.length_unit)
        print(ret+gap+' Maximum distance: ' +
              str(self.r_max)+' '+self.units.length_unit)
        print(ret+gap+' Sigma for Gaussian: ' +
              str(self.sigma)+' '+self.units.length_unit)
        print(ret+gap+' Interval for Gaussian: ' +
              str(self.delta)+' '+self.units.length_unit)
        print(ret+gap+' Number of basis functions: ' + str(self.num_basis))
        if self.clip_distance:
            print(ret+gap+' Clip the range of distance to (r_min,r_max).')
        if self.rescale:
            print(ret+gap+' Rescale the range of RBF to (-1,1).')
        return self

    def construct(self, distance: Tensor):
        """Compute gaussian type RBF.

        Args:
            distance (ms.Tensor[float], [B,A,N]): distances between atoms

        Returns:
            RBF (ms.Tensor[float], [B,A,N,F]): radical basis functions

        """
        if self.clip_distance:
            distance = C.clip_by_value(distance, self.r_min, self.r_max)

        ex_dis = F.expand_dims(distance, -1)
        diff = ex_dis - self.offsets
        rbf = self.exp(self.coeff * F.square(diff))

        if self.rescale:
            rbf = rbf * 2 - 1.0

        return rbf


@_rbf_register('log_gaussian')
class LogGaussianBasis(RadicalBasisFunctions):
    r"""Log Gaussian type RBF.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0.04 nm

        sigma (float):          Simga. Default: 0.3

        delta (float):          Space interval. Default: 0.0512

        num_basis (int):        Number of basis functions. Defatul: None

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: True

        clip_distance (bool):   Whether to clip the value of distance. Default: False

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: None

        r_ref (Length):         Reference distance. Default: 1 nm

    """
    def __init__(self,
                 r_max: Length = Length(1, 'nm'),
                 r_min: Length = Length(0.04, 'nm'),
                 sigma: float = 0.3,
                 delta: float = 0.0512,
                 num_basis: int = None,
                 rescale: bool = True,
                 clip_distance: bool = False,
                 length_unit: str = 'nm',
                 hyper_param: dict = None,
                 r_ref: Length = Length(1, 'nm'),
                 ):

        super().__init__(
            r_min=r_min,
            r_max=r_max,
            sigma=sigma,
            delta=delta,
            num_basis=num_basis,
            rescale=rescale,
            clip_distance=clip_distance,
            length_unit=length_unit,
            hyper_param=hyper_param,
        )
        if hyper_param is not None:
            r_ref = get_hyper_parameter(hyper_param, 'r_ref')

        self.reg_key = 'log_gaussian'

        if isinstance(self.sigma, Length):
            raise TypeError(
                '"sigma" in Log Gaussian RBF is dimensionless, so its type should not be "Length"')
        if isinstance(self.delta, Length):
            raise TypeError(
                '"delta" in Log Gaussian RBF is dimensionless, so its type should not be "Length"')

        self.delta = Tensor(self.delta, ms.float32)
        self.r_ref = self.get_length(r_ref)

        log_rmin = msnp.log(self.r_min/self.r_ref, dtype=ms.float32)
        log_rmax = msnp.log(self.r_max/self.r_ref, dtype=ms.float32)
        log_range = log_rmax-log_rmin
        if self.delta is None:
            self.offsets = msnp.linspace(
                log_rmin, log_rmax, self.num_basis, dtype=ms.float32)
            self.delta = Tensor(log_range/(self.num_basis-1), ms.float32)
        else:
            if self.num_basis is None:
                num_basis = msnp.ceil(log_range/self.delta, ms.int32) + 1
                self.num_basis = get_integer(num_basis)
            self.offsets = msnp.log(self.r_min/self.r_ref) + \
                msnp.arange(0, self.num_basis) * self.delta

        # self.sigma = Tensor(sigma,ms.float32)
        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))
        self.inv_ref = msnp.reciprocal(self.r_ref)

        self.set_hyper_param()

        self.log = P.Log()
        self.exp = P.Exp()
        self.max = P.Maximum()
        self.min = P.Minimum()

        self.exp = P.Exp()

    def set_basis(self, num_basis=None, delta=None):
        self.check_basis(num_basis, delta)

        log_rmin = msnp.log(self.r_min/self.r_ref, dtype=ms.float32)
        log_rmax = msnp.log(self.r_max/self.r_ref, dtype=ms.float32)
        log_range = log_rmax-log_rmin

        if delta is None:
            log_rmax = msnp.log(self.r_max/self.r_ref, dtype=ms.float32)
            self.num_basis = get_integer(num_basis)
            self.offsets = msnp.linspace(
                log_rmin, log_rmax, num_basis, dtype=ms.float32)
            self.delta = Tensor(log_range/(num_basis-1), ms.float32)
        else:
            self.delta = Tensor(delta, ms.float32)
            if num_basis is None:
                num_basis = msnp.ceil(log_range/self.delta, ms.int32) + 1
                self.num_basis = get_integer(num_basis)
            self.offsets = msnp.log(
                self.r_min/self.r_ref) + msnp.arange(0, self.num_basis.asnumpy()) * self.delta
        set_hyper_parameter(self.hyper_param, 'delta', self.delta)
        set_hyper_parameter(self.hyper_param, 'num_basis', self.num_basis)
        return self

    def set_sigma(self, sigma):
        self.sigma = Tensor(sigma, ms.float32)
        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))
        set_hyper_parameter(self.hyper_param, 'sigma', self.sigma)
        return self

    def set_ref(self, r_ref):
        self.r_ref = self.get_length(r_ref)
        self.inv_ref = msnp.reciprocal(self.r_ref)
        set_hyper_parameter(self.hyper_param, 'r_ref', self.r_ref)
        return self

    def set_hyper_param(self):
        super().set_hyper_param()
        set_hyper_parameter(self.hyper_param, 'r_ref', self.r_ref)
        return self

    def change_unit(self, unit):
        scale = super().change_unit(unit)
        self.r_ref *= scale
        self.inv_ref = msnp.reciprocal(self.r_ref)
        self.set_hyper_param()
        return scale

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = '-'):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+' Minimum distance: ' +
              str(self.r_min)+' '+self.units.length_unit)
        print(ret+gap+' Maximum distance: ' +
              str(self.r_max)+' '+self.units.length_unit)
        print(ret+gap+' Reference distance: ' +
              str(self.r_ref)+' '+self.units.length_unit)
        print(ret+gap+' Log Gaussian begin: ' + str(self.offsets[0]))
        print(ret+gap+' Log Gaussian end: ' + str(self.offsets[-1]))
        print(ret+gap+' Interval for log Gaussian: '+str(self.delta))
        print(ret+gap+' Sigma for log gaussian: ' + str(self.sigma))
        print(ret+gap+' Number of basis functions: ' + str(self.num_basis))
        if self.clip_distance:
            print(ret+gap+' Clip the range of distance to (r_min,r_max).')
        if self.rescale:
            print(ret+gap+' Rescale the range of RBF to (-1,1).')
        return self

    def construct(self, distance: Tensor):
        """Compute log gaussian type RBF.

        Args:
            distance (ms.Tensor[float], [B,A,N]): distances between atoms

        Returns:
            RBF (ms.Tensor[float], [B,A,N,F]): radical basis functions

        """
        if self.clip_distance:
            distance = C.clip_by_value(distance, self.r_min, self.r_max)

        # (B,A,N)
        log_r = self.log(distance * self.inv_ref)
        # (B,A,N,1)
        log_r = F.expand_dims(log_r, -1)

        # (B,A,N,K) = (B,A,N,1) - (K)
        log_diff = log_r - self.offsets
        rbf = self.exp(self.coeff*F.square(log_diff))

        if self.rescale:
            rbf = rbf * 2 - 1.0

        return rbf


_RBF_BY_NAME = {rbf.__name__: rbf for rbf in _RBF_BY_KEY.values()}


def get_rbf(rbf: str = None,
            r_max=Length(1, 'nm'),
            length_unit='nm'
            ) -> RadicalBasisFunctions:
    """get RBF by name"""

    if isinstance(rbf, RadicalBasisFunctions):
        return rbf
    if rbf is None:
        return None

    hyper_param = None
    if isinstance(rbf, dict):
        if 'name' not in rbf.keys():
            raise KeyError('Cannot find the key "name" in rbf dict!')
        hyper_param = rbf
        rbf = get_hyper_string(hyper_param, 'name')

    if isinstance(rbf, str):
        if rbf.lower() == 'none':
            return None
        if rbf.lower() in _RBF_BY_KEY.keys():
            return _RBF_BY_KEY[rbf.lower()](r_max=r_max, length_unit=length_unit, hyper_param=hyper_param)
        if rbf in _RBF_BY_NAME.keys():
            return _RBF_BY_NAME[rbf](r_max=r_max, length_unit=length_unit, hyper_param=hyper_param)

        raise ValueError(
            "The RBF corresponding to '{}' was not found.".format(rbf))

    raise TypeError("Unsupported RBF type '{}'.".format(type(rbf)))

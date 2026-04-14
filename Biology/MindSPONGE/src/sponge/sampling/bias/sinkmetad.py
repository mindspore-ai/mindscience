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
"""Metadynamics"""

from typing import Tuple, List, Union
import itertools
import numpy as np
from numpy import ndarray
import mindspore as ms
try:
    # MindSpore 2.X
    from mindspore import jit
except ImportError:
    # MindSpore 1.X
    from mindspore import ms_function as jit
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from ...potential.bias import Bias
from ...colvar import Colvar
from ...function import get_arguments
from ...function import get_ms_array, get_integer, get_tensor, periodic_difference, keepdims_sum


class SinkMetadynamics(Bias):
    r"""Bais potential of (well-tempered) metadynamics (MetaD/WT-MetaD)

    References:

        Laio, A.; Parrinello, M.
        Escaping Free-Energy Minima [J].
        Proceedings of the National Academy of Sciences, 2002, 99(20): 12562-12566.

        Barducci, A.; Bussi, G.; Parrinello, M.
        Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method [J].
        Physical Review Letters, 2008, 100(2): 020603.

    Math:

    .. math::

        V[s(R)] = \sum_t {\omega(t) e ^ {-\frac{[s(R) - s(t)] ^ 2}{2 \sigma ^ 2}}}

        \omega (t) = w e ^ {-\frac{1}{\gamma - 1} \beta V[R(t)]}

    Args:

        colvar (Colvar):        Collective variables (CVs) :math:`s(R)`.

        update_pace (int):      Frequency for hill addition.

        grid_min (float):       Lower bounds for the grids of CVs.

        grid_max (float):       Upper bounds for the grids of CVs.

        grid_bin (int):         Number of bins for the grids of CVs.

        height (float):         Heights of the Gaussian hills :math:`w`.

        sigma (float):          Widths of the Gaussian hills :math:`\sigma`.

        bias_factor (float):    Well-tempered bias factor :math:`\gamma`.
                                When None is given, WT-MetaD is not used. Default: None

        share_parameter (bool): Whether to share Metadynamics parameter for all walkers.
                                If False is given, then num_walker must be given.
                                Default: True

        num_walker (int):       Number of multiple walkers. Default: None

        use_cutoff (bool):      Whether to use cutoff when calculating gaussian from grids.
                                Default: True

        grid_cutoff (float):    Cutoff for grids. Default: 3.6

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: None

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: None

    Supported Platforms:

        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Colvar,
                 update_pace: int,
                 height: float,
                 sigma: float,
                 grid_min: float = None,
                 grid_max: float = None,
                 grid_bin: int = None,
                 grid_mask: Union[Tensor, ndarray, List, Tuple] = None,
                 custom_grid: Union[Tensor, ndarray, List, Tuple] = None,
                 bias_factor: float = None,
                 temperature: float = 300,
                 share_parameter: bool = True,
                 num_walker: int = None,
                 use_cutoff: bool = True,
                 grid_cutoff: float = 2.5,
                 project_to_grids: bool = False,
                 integral_dimension: int = None,
                 integral_sigma: float = None,
                 integral_spacing: float = None,
                 sinking: bool = False,
                 sink_depth: float = 0,
                 length_unit: str = None,
                 energy_unit: str = None,
                 scale_sink_factor: bool = True,
                 **kwargs,
                 ):

        super().__init__(
            name='metadynamics',
            colvar=colvar,
            update_pace=update_pace,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if self.colvar.ndim != 1:
            raise ValueError(f'The rank (ndim) of the colvar used in Metdyanmics must be 1 '
                             f'but got: {self.colvar.ndim}')

        # S: dimension of the collective variables
        self.dim_colvar = self.colvar.shape[-1]

        self.share_parameter = share_parameter
        self.num_walker = get_integer(num_walker)
        self.integral_dimension = get_integer(integral_dimension)
        if integral_dimension is None:
            self.integral_dimension = self.dim_colvar
        self.num_parameter = self.num_walker
        if self.share_parameter:
            self.num_parameter = 1
        if self.num_walker is None:
            if self.share_parameter:
                self.num_walker = 1
            else:
                raise ValueError('num_walkers must be given when share_parameter is False!')

        self.any_periodic = self.colvar.any_periodic
        self.all_periodic = self.colvar.all_periodic

        self.sinking = sinking

        self.periodic_mask = None if self.all_periodic else self.colvar.periodic

        self.use_extend_grids = True
        if self.sinking or self.all_periodic:
            self.use_extend_grids = False

        def _check_dimension(inputs: Tensor, name: str, dtype: type = None) -> Tensor:
            """check dimension of variables"""
            inputs = get_ms_array(inputs, dtype)
            if inputs.ndim > 1:
                raise ValueError(
                    f'The ndim of {name} cannot be larger than 1 but got: {inputs.ndim}')
            if inputs.ndim == 0:
                inputs = F.reshape(inputs, (1,))
            if inputs.size != self.dim_colvar:
                if inputs.size != 1:
                    raise ValueError(f'The dimension of {name} ({inputs.size}) does not match '
                                     f'the dimension of colvar {self.dim_colvar}')
                inputs = msnp.broadcast_to(inputs, (self.dim_colvar,))
            return inputs

        def _check_int_dim(inputs: Tensor, name: str, dtype: type = None) -> Tensor:
            """check dimension of integral"""
            inputs = get_ms_array(inputs, dtype)
            if inputs.ndim > 1:
                raise ValueError(
                    f'The ndim of {name} cannot be larger than 1 but got: {inputs.ndim}')
            if inputs.ndim == 0:
                inputs = F.reshape(inputs, (1,))
            if inputs.size != self.integral_dimension:
                if inputs.size != 1:
                    raise ValueError(f'The dimension of {name} ({inputs.size}) does not match '
                                     f'the `integral_dimension` {self.integral_dimension}')
                inputs = msnp.broadcast_to(inputs, (self.dim_colvar,))
            return inputs

        # \sqrt(2)
        self.sqrt2 = msnp.sqrt(2.0, ms.float32)
        # \sqrt(2)^D
        self.sqrt2d = msnp.power(self.sqrt2, self.integral_dimension)
        # 1 / {\sqrt(2)^D}
        self.inv_sqrt2d = msnp.reciprocal(self.sqrt2d)
        # 2^D
        self.power2d = msnp.power(2.0, self.integral_dimension, ms.float32)
        # (1/2)^D
        self.inv_power2d = msnp.reciprocal(self.power2d)

        # (S)
        self.sigma0 = _check_dimension(sigma, 'sigma', ms.float32)
        self.height = get_ms_array(height, ms.float32)
        if self.height.size != 1:
            raise ValueError(f'The size of height must be 1 but got: {self.height.size}')
        self.sigma = self.sigma0 / self.sqrt2
        self.coeff = -0.5 / F.square(self.sigma)

        if integral_sigma is None:
            if self.integral_dimension == self.dim_colvar:
                self.integral_sigma = self.sigma
            elif get_ms_array(sigma).size == 1:
                integral_sigma = _check_int_dim(sigma, 'integral_sigma', ms.float32)
                self.integral_sigma = integral_sigma / self.sqrt2
            else:
                raise ValueError('`integral_sigma` cannot be `None` when `integral_dimension` is not equal to '
                                 'the dimension of colvar and the input number of `sigma` is greater than 1.')
        else:
            self.integral_sigma = _check_int_dim(integral_sigma, 'integral_sigma', ms.float32)

        self.neigh_grids_shift: Tensor = None
        self.num_neigh: int = None
        self.grid_mask: Tensor = None
        self.index_convert_factor: Tensor = None
        self.grids0_neigh_index: Tensor = None
        self.neigh_gaussian: Tensor = None
        self.grid_cutoff = Tensor(grid_cutoff, ms.float32)
        if custom_grid is None:
            self.use_custom_grid = False
            self.custom_grid = None
            self.use_cutoff = use_cutoff

            # (S)
            self.grid0_min = _check_dimension(grid_min, 'grid_min', ms.float32)
            self.grid0_max = _check_dimension(grid_max, 'grid_max', ms.float32)
            self.grid0_range: Tensor = self.grid0_max - self.grid0_min

            grid0_bin = _check_dimension(grid_bin, 'grid_bin', ms.int32)

            self.grid_spacing: Tensor = self.grid0_range / grid0_bin
            self.grid0_bin = msnp.where(self.colvar.periodic, grid0_bin, grid0_bin + 1)

            grid0_min = self.grid0_min.asnumpy()
            grid0_max = self.grid0_max.asnumpy()
            grid0_bin = self.grid0_bin.asnumpy()

            grid_spacing = self.grid_spacing.asnumpy()
            grids0 = []
            for i in range(self.dim_colvar):
                if self.colvar.periodic[i]:
                    grids0.append(np.arange(0, grid0_bin[i]) * grid_spacing[i] + grid0_min[i])
                else:
                    grids0.append(np.linspace(grid0_min[i], grid0_max[i], grid0_bin[i]))

            grids0 = tuple(itertools.product(*grids0))
            # (G_0, S)
            self.grids0 = Tensor(grids0, ms.float32)
            # G_0
            self.num_grids0 = self.grids0.shape[0]
            # (G_0)
            self.grid0_factor = msnp.cumprod(self.grid0_bin[::-1], axis=-1)
            self.grid0_factor = msnp.concatenate((self.grid0_factor[1::-1], Tensor([1], ms.int32)), axis=-1)

            # (S)
            cutoff_bin = msnp.ceil(self.grid_cutoff * self.sigma0 / self.grid_spacing)
            cutoff_bin = F.cast(cutoff_bin, ms.int32)
            cutoff = self.grid_spacing * cutoff_bin

            self.grid_bin: Tensor = self.grid0_bin
            self.grid_min: Tensor = self.grid0_min
            self.grid_max: Tensor = self.grid0_max
            self.grid_range: Tensor = self.grid0_range
            self.grids: Tensor = self.grids0
            if self.use_extend_grids:
                ex_range = msnp.where(self.colvar.periodic, 0, cutoff)
                self.grid_bin = self.grid0_bin + msnp.where(self.colvar.periodic, 0, 2 * cutoff_bin)
                self.grid_min: Tensor = self.grid0_min - ex_range
                self.grid_max: Tensor = self.grid0_max + ex_range
                self.grid_range: Tensor = self.grid0_range + 2 * ex_range

                grid_min = self.grid_min.asnumpy()
                grid_max = self.grid_max.asnumpy()
                grid_bin = self.grid_bin.asnumpy()

                grids = []
                for i in range(self.dim_colvar):
                    if self.colvar.periodic[i]:
                        grids.append(np.arange(0, grid_bin[i]) * grid_spacing[i] + grid_min[i])
                    else:
                        grids.append(np.linspace(grid_min[i], grid_max[i], grid_bin[i]))

                # (G, S)
                grids = tuple(itertools.product(*grids))
                self.grids = Tensor(grids, ms.float32)

            self.num_grids = self.grids.shape[0]

            if integral_spacing is None:
                if self.integral_dimension != self.dim_colvar:
                    raise ValueError('`integral_spacing` cannot be `None` when `integral_dimension` '
                                     'is not equal to the dimension of colvar.')
                # (1,) <- (S)
                self.integral_spacing = F.prod(self.grid_spacing, -1)
            else:
                integral_spacing = get_ms_array(integral_spacing, ms.float32)
                if integral_spacing.ndim > 1:
                    raise ValueError(
                        f'The ndim of integral_spacing must be 0 or 1 but got: {integral_spacing.ndim}')
                if integral_spacing.size not in (1, self.num_grids):
                    raise ValueError(f'The size of integral_spacing must be 1 or '
                                    f'equal to the number of grids ({self.num_grids}), '
                                    f'but got: {integral_spacing}')
                self.integral_spacing = integral_spacing

            # (G)
            self.index_convert_factor = msnp.cumprod(self.grid_bin[::-1], axis=-1)
            self.index_convert_factor = msnp.concatenate((self.index_convert_factor[:-1][::-1],
                                                          Tensor([1], ms.int32)), axis=-1)

            if self.use_cutoff:
                cutoff_bin_ = cutoff_bin.asnumpy()
                neigh_grids_shift = []
                for i in range(self.dim_colvar):
                    neigh_grids_shift.append(np.arange(-cutoff_bin_[i], cutoff_bin_[i] + 1, dtype=np.int32))
                # (N, S)
                neigh_grids_shift = tuple(itertools.product(*neigh_grids_shift))
                self.neigh_grids_shift = Tensor(neigh_grids_shift, ms.int32)
                self.num_neigh = self.neigh_grids_shift.shape[0]

                # (N, S)
                neigh_diff = self.neigh_grids_shift * self.grid_spacing
                # (N) <- (N, S)
                self.neigh_gaussian = F.exp(F.reduce_sum(
                    self.coeff * F.square(neigh_diff), -1)) * self.integral_spacing

                # (G_0, S)
                grids0_index = self.get_nearest_grid(self.grids0)
                # (G_0, N, S)
                _, grids0_neigh_index = self.get_neighbours(grids0_index)
                # (G_0, N)
                self.grids0_neigh_index = self.get_hills_index(grids0_neigh_index)

        else:
            if not self.sinking:
                raise ValueError('Cannot use `custom_grid` without SinkMeta!')
            self.use_custom_grid = True
            self.use_cutoff = False

            # (G, S)
            self.grids0 = get_tensor(custom_grid, ms.float32)
            if self.grids0.shape[-1] != self.dim_colvar:
                raise ValueError(f'The last dimension of `custom_grid` must be equal to '
                                 f'the dimension of `colvar` ({self.dim_colvar}), '
                                 f'but got: {self.grids0.shape[-1]}.')

            self.num_grids0 = self.grids0.shape[0]
            self.num_grids = self.num_grids0

            if integral_spacing is None:
                raise ValueError(
                    '`integral_spacing` cannot be `None` when using custom grids.')
            integral_spacing = get_ms_array(integral_spacing, ms.float32)
            if integral_spacing.ndim > 1:
                raise ValueError(
                    f'The ndim of integral_spacing must be 0 or 1 but got: {integral_spacing.ndim}')
            if integral_spacing.size not in (1, self.num_grids):
                raise ValueError(f'The size of integral_spacing must be 1 or '
                                 f'equal to the number of grids ({self.num_grids}), '
                                 f'but got: {integral_spacing}')
            self.integral_spacing = integral_spacing

            if self.any_periodic:
                if grid_min is None:
                    raise ValueError('`grid_min` cannot be `None` when `colvar` has a periodic component.')
                if grid_max is None:
                    raise ValueError('`grid_max` cannot be `None` when `colvar` has a periodic component.')
                self.grid0_min = _check_dimension(grid_min, 'grid_min', ms.float32)
                self.grid0_max = _check_dimension(grid_max, 'grid_max', ms.float32)
            else:
                self.grid0_min = msnp.amin(self.grids, 0)
                self.grid0_max = msnp.amax(self.grids, 0)
            self.grid0_range = self.grid0_max - self.grid0_min

            self.grids = self.grids0
            self.grid_min = self.grid0_min
            self.grid_max = self.grid0_max
            self.grid_range = self.grid0_range

        self.unequally_spacing = False
        self.avg_spacing = self.integral_spacing
        if self.integral_spacing.size > 1:
            self.unequally_spacing = True
            self.avg_spacing = msnp.mean(self.integral_spacing, -1, True)

        if self.use_cutoff:
            self.grids0_guassian: Tensor = None
        else:
            # (G_0, G, S) = (G_0, 1, S) - (1, G, S)
            grids0_diff = F.expand_dims(self.grids0, -2) - F.expand_dims(self.grids, -3)
            if self.any_periodic:
                grids0_diff = periodic_difference(
                    grids0_diff, self.grid0_range, self.periodic_mask)
            # (G_0, G, S) = (G_0, G, S) * (S)
            grids0_exp = self.coeff * F.square(grids0_diff)
            # (G_0, G) <- (G_0, G, S)
            self.grids0_guassian = F.exp(F.reduce_sum(grids0_exp, -1)) * self.integral_spacing

        # G
        self.num_grids = self.grids.shape[0]

        # (B, G)
        if grid_mask is not None:
            if not self.sinking:
                raise ValueError('Cannot use `grid_mask` without SinkMeta!')
            self.grid_mask = get_tensor(grid_mask, ms.bool_)
            if self.grid_mask.shape[-1] != self.num_grids:
                raise ValueError(f'The last dimension of `grid_mask` must be equal to '
                                 f'the number of grids ({self.num_grids}), '
                                 f'but got: {self.grid_mask.shape[-1]}.')
            if self.grid_mask.ndim != 2:
                if self.grid_mask.ndim == 1:
                    self.grid_mask = F.expand_dims(self.grid_mask, 0)
                else:
                    raise ValueError(f'The rank(ndim) of `grid_mask` must be 1 or 2, '
                                        f'but got: {self.grid_mask.ndim}.')
            if self.grid_mask.shape[0] != self.num_parameter:
                if self.grid_mask.shape[0] == 1:
                    self.grid_mask = msnp.broadcast_to(self.grid_mask, (self.num_parameter, self.num_grids))
                else:
                    raise ValueError(f'The first dimension of `grid_mask` must be 1 or equal to '
                                        f'`num_walker`, but got: {self.grid_mask.shape[0]}.')

        # :math:`\sqrt(2) \sigma^'`
        factor_norm = msnp.sqrt(msnp.pi, ms.float32) * self.integral_sigma
        self.factor_norm = F.prod(factor_norm, -1, True)
        self.inv_factor_norm = msnp.reciprocal(self.factor_norm)
        self.max_factor = self.inv_factor_norm

        self.project_to_grids = project_to_grids
        # C' = \sqrt(\pi) \sigma'
        conv_norm = msnp.sqrt(msnp.pi) * self.integral_sigma
        # () <- (S')
        self.conv_norm = F.prod(conv_norm, -1, True)
        # (B, 1) or (1, 1)
        self.inv_conv_norm = msnp.reciprocal(self.conv_norm)

        self.guassian_norm = self.conv_norm * self.sqrt2d
        self.guassian2_norm = self.conv_norm
        self.guassian_sqrt_norm = self.conv_norm * self.power2d

        self.num_max = self.guassian_norm / self.avg_spacing
        self.num_max2 = self.guassian2_norm / self.avg_spacing
        self.num_max_sqrt = self.guassian_sqrt_norm / self.avg_spacing

        self.sink_depth = 0
        self.sink_norm = 1
        self.inv_sink_norm = 1
        self.scale_sink_factor = False
        if self.project_to_grids:
            self.scale_sink_factor = scale_sink_factor
        if self.sinking:
            self.sink_depth = get_tensor(sink_depth, ms.float32)

            # C' = \sqrt(2\pi) \sigma'
            sink_norm = self.sqrt2 * conv_norm
            # () <- (S')
            self.sink_norm = F.prod(sink_norm, -1, True)
            # (B, 1) or (1, 1)
            self.inv_sink_norm = msnp.reciprocal(self.sink_norm)

        self.temperature = temperature
        self.kbt = self.units.boltzmann * temperature
        self.beta0 = 1.0 / self.kbt

        self.large_neg = Tensor(-65504, ms.float32)

        #  (1, G) or (B, G)
        self.hills = Parameter(msnp.zeros((self.num_parameter, self.num_grids), ms.float32),
                               name="hills", requires_grad=False)

        # \gamma
        self.bias_factor = get_ms_array(bias_factor, ms.float32)
        if self.bias_factor is None:
            self.well_temped = False
            self.wt_factor = 0
            self.wt_factor0 = 0
            self.reweighting_factor = None
        else:
            if self.bias_factor.size != 1:
                raise ValueError(f'The size of bias_factor must be 1 but got: {self.bias_factor.size}')
            self.well_temped = True
            if self.bias_factor <= 1:
                raise ValueError('bias_factor must be larger than 1')
            # 1 / (\gamma - 1) * \beta
            self.wt_factor = self.beta0 / (self.bias_factor - 1.0)
            # \gamma / (\gamma - 1) * \beta
            self.wt_factor0 = self.beta0 * self.bias_factor / (self.bias_factor - 1.0)
            self.reweighting_factor = Parameter(msnp.zeros((self.num_parameter, 1), ms.float32),
                                                name="reweighting_factor", requires_grad=False)

        if self.sinking:
            #  (1, G) or (B, G)
            self.max_bias = Parameter(msnp.zeros((self.num_parameter, 1), ms.float32),
                                      name="max_bias", requires_grad=False)
            #  (1, G) or (B, G)
            self.bias_depth = Parameter(msnp.zeros((self.num_parameter, 1), ms.float32),
                                        name="bias_depth", requires_grad=False)

            self.shift_factor = Parameter(msnp.zeros((self.num_parameter, 1), ms.float32),
                                          name="shift_factor", requires_grad=False)
        else:
            self.max_bias: Parameter = None
            self.bias_depth: Parameter = None
            self.shift_factor: Parameter = None

    @property
    def boltzmann(self) -> float:
        """Boltzmann constant"""
        return self.units.boltzmann

    @property
    def periodic(self) -> Tensor:
        """periodic of collectiva variables"""
        return self.colvar.periodic

    def gather_value(self, value: Tensor, index: Tensor):
        if self.share_parameter:
            # (B, N) = (B, N) * (1, N)
            return value[0][index]
        # (B, N) <- (B, G) | (B, N)
        return F.gather_d(value, -1, index)

    def get_weights(self,
                    index: Tensor = None,
                    shift: Tensor = 0,
                    mask: Tensor = None) -> Tensor:
        r"""get weights by index of hills.

        Args:
            index (Tensor):     Tensor of shape (B, N). Data type is int.
                                Index hills.
                                If None is given, weights of the full hills will be return.
                                Default: None

        Returns:
            weight (Tensor):    Tensor of shape (B, N) or (B, G). Data type is float.
                                Value of neighbouring grids.

        """
        if index is None:
            if mask is not None:
                # (B, G) <- [(1, 1) or (B, 1)]
                shift = shift * mask
            # (B, G)
            return self.hills + shift

        if self.share_parameter:
            if mask is not None:
                # (1, N) <- (1, 1)
                shift = shift * mask[0][index]
            # (1, N)
            return self.hills[0][index] + shift

        if mask is not None:
            # (B, N) <- (B, G)
            mask = F.gather_d(mask, -1, index)
            # (B, N) <- (B, 1)
            shift = shift * mask

        # (B, N) <- (B, G)
        return F.gather_d(self.hills, -1, index) + shift

    def get_gaussians(self):
        """return gaussian grids"""
        return self.grids0

    def get_neighbours(self, center: Tensor) -> Tuple[Tensor, Tensor]:
        r"""get neighbouring grids of a cetner grid.

        Args:
            center (Tensor):    Tensor of shape `(..., S)`. Data type is int.
                                Index of center grid.

        Returns:
            grids (Tensor):     Tensor of shape `(..., N, S)`. Data type is float.
                                Value of neighbouring grids.
            index (Tensor):     Tensor of shape `(..., N, S)`. Data type is int.
                                Index of neighbouring grids.

        """

        # (..., N, S) = (..., 1, S) + (N, S)
        index = F.expand_dims(center, -2) + self.neigh_grids_shift
        # (..., N, S) = (S) + (..., N, S) * (S)
        grids = self.grid_min + index * self.grid_spacing

        if self.any_periodic:
            period_idx = F.select(index < 0, index + self.grid_bin, index)
            min_index = period_idx - self.grid_bin
            period_idx = F.select(period_idx >= self.grid_bin, min_index, period_idx)

            if self.all_periodic:
                index = period_idx
            else:
                index = msnp.where(self.periodic, period_idx, index)

        return grids, index

    def get_nearest_grid(self, colvar: Tensor) -> Tensor:
        r"""get the nearest grid of a set of collectivate variables (CVs).

        Args:
            colvar (Tensor):    Tensor of shape `(..., S)`. Data type is float.
                                Collective variabless

        Returns:
            index (Tensor):     Tensor of shape `(..., S)`. Data type is int.
                                Index of the nearest grids.

        """
        # (..., S) = ((..., S) - (S)) / (S)
        nearest_grid = F.floor((colvar - self.grid_min) / self.grid_spacing + 0.5)
        return F.cast(nearest_grid, ms.int32)

    def get_hills_index(self, grid: Tensor) -> Tensor:
        r"""convert the index of grid to the index of the hills of metadynamics.

        Args:
            grids (Tensor): Tensor of shape (..., S). Data type is int.
                            Index of grids

        Returns:
            index (Tensor): Tensor of shape (...). Data type is int.
                            Index of hills.

        """
        # (...) <- (..., S) * (S)
        grid *= self.index_convert_factor
        return grid.sum(-1)

    def calc_basis_exp(self, colvar: Tensor) -> Tuple[Tensor, Tensor]:
        r"""calculate the value and (hills) indices of the gaussians of neighbouring grids
            of the collective variables (CVs).

        Args:
            colvar (Tensor):    Tensor of shape `(B, S)`. Data type is float.
                                Collective variables (CVs).

        Returns:
            gaussian (Tensor):  Tensor of shape `(B, N)`. Data type is float.
                                Value of the neighbouring Gaussian.
            indices (Tensor):   Tensor of shape `(B, N)`. Data type is int.
                                Hills indice of the neighbouring Gaussians.

        """

        indices = None
        if self.use_cutoff:
            # (B, S)
            nearest_grid = self.get_nearest_grid(colvar)
            # (B, N, S)
            neigh_grids, neigh_index = self.get_neighbours(nearest_grid)
            # (B, N, S) = (B, N, S) - (B, 1, S)
            diff = F.expand_dims(colvar, -2) - neigh_grids

            # (B, N) <- (B, G, S) = (S) * (B, G, S)
            gaussian_exp = F.reduce_sum(self.coeff * F.square(diff), -1)
            indices = self.get_hills_index(neigh_index)

        # (B, G, S) = (B, 1, S) - (G, S)
        diff = F.expand_dims(colvar, -2) - self.grids
        if self.any_periodic:
            diff = periodic_difference(diff, self.grid0_range, self.periodic_mask)

        # (B, G) <- (B, G, S) = (S) * (B, G, S)
        gaussian_exp = F.reduce_sum(self.coeff * F.square(diff), -1)

        return gaussian_exp, indices

    def calc_gaussian_basis(self, colvar: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        # (B, G) or (B, N)
        gaussian_exp, indices = self.calc_basis_exp(colvar)
        gaussian = F.exp(gaussian_exp)

        if mask is not None:
            mask = self.get_spacing(indices)
            return gaussian * mask

        return gaussian, indices

    def calc_neighbour_gaussian(self, colvar: Tensor,
                                mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""calculate the value and (hills) indices of the gaussians of neighbouring grids
            of the collective variables (CVs).

        Args:
            colvar (Tensor):    Tensor of shape `(B, S)`. Data type is float.
                                Collective variables (CVs).

        Returns:
            gaussian (Tensor):  Tensor of shape `(B, N)`. Data type is float.
                                Value of the neighbouring Gaussian.
            indices (Tensor):   Tensor of shape `(B, N)`. Data type is int.
                                Hills indice of the neighbouring Gaussians.

        """
        # (B, S)
        nearest_grid = self.get_nearest_grid(colvar)
        # (B, N, S)
        neigh_grids, neigh_index = self.get_neighbours(nearest_grid)

        # (B, N, S) = (B, N, S) - (B, 1, S)
        diff = F.expand_dims(colvar, -2) - neigh_grids

        # (B, N) <- (B, G, S) = (S) * (B, G, S)
        gaussian = F.exp(F.reduce_sum(self.coeff * F.square(diff), -1))

        # (B, N) <- (B, N, S)
        indices = self.get_hills_index(neigh_index)

        if mask is not None:
            if self.share_parameter:
                # (B, N) = (B, N) * (1, N)
                gaussian = gaussian * mask[0][indices]
            else:
                # (B, N) <- (B, G) | (B, N)
                mask_ = F.gather_d(mask, -1, indices)
                # (B, N) = (B, N) * (B, N)
                gaussian = gaussian * mask_

        return gaussian, indices

    def calc_grids_gaussian(self, colvar: Tensor, mask: Tensor = None) -> Tensor:
        r"""calculate the gaussians of grids of the collective variables (CVs).

        Args:
            colvar (Tensor):    Tensor of shape `(B, S)`. Data type is float.
                                Collective variables (CVs).

        Returns:
            gaussian (Tensor):  Tensor of shape `(B, G)`. Data type is int.
                                Gaussian of grids.

        """
        # (B, G, S) = (B, 1, S) - (G, S)
        diff = F.expand_dims(colvar, -2) - self.grids

        if self.any_periodic:
            diff = periodic_difference(diff, self.grid0_range, self.periodic_mask)

        # (B, G) <- (B, G, S) = (S) * (B, G, S)
        gaussian = F.exp(F.reduce_sum(self.coeff * F.square(diff), -1))

        if mask is not None:
            # (B, G) * (B, G)
            return gaussian * mask

        return gaussian

    def calc_reweight_factor(self) -> Tensor:
        r"""calculate the reweighting factor :math:`c(t)` of metadynamics

        Return:
            rct (Tensor):   Tensor of shape `(B, 1)`. Data type is float.
                            Reweighting factor :math:`c(t)`.
        """
        if self.reweighting_factor is None:
            return None

        if self.use_cutoff:
            # (B, G_0, N) <- (B, G) | (G_0, N)
            weights = F.gather(self.hills, self.grids0_neigh_index, -1)
            # (B, G_0, N) = (B, G_0, N) * (N)
            biases = F.stop_gradient(weights) * self.neigh_gaussian
        else:
            # (B, 1, G) <- (B, G)
            weights = F.expand_dims(self.hills, -2)
            # (B, G_0, G) <- (B, 1, G) * (G_0, G)
            biases = F.stop_gradient(weights) * self.grids0_guassian

        # (B, G_0) <- [(B, G_0, N) or (B, G_0, G)]
        biases = F.reduce_sum(biases, -1)

        if self.sinking:
            # (B, 1) <- (B, G_0)
            max_bias = msnp.amax(biases, -1, keepdims=True)
            max_bias = msnp.where(max_bias > self.max_bias, max_bias, self.max_bias)
            biases = F.depend(biases, F.assign(self.max_bias, max_bias))

        # \gamma / (\gamma - 1) * \beta * V(t)
        rct0 = self.wt_factor0 * biases
        # 1 / (\gamma - 1) * \beta * V(t)
        rct1 = self.wt_factor * biases

        # (B, 1) <- (B, G_0)
        rct0 = F.logsumexp(rct0, -1, True)
        rct1 = F.logsumexp(rct1, -1, True)
        rct = (rct0 - rct1) * self.kbt

        return rct

    def update_reweight_factor(self, rct: Tensor = None) -> Tensor:
        """update the value of reweighting factor :math:`c(t)`"""
        if self.reweighting_factor is None:
            return None
        if rct is None:
            rct = self.calc_reweight_factor()
        return F.assign(self.reweighting_factor, rct)

    def get_spacing(self, index: Tensor = None):
        if index is not None and self.unequally_spacing:
            return self.gather_value(self.integral_spacing, index)
        return self.integral_spacing

    def get_mask(self, index: Tensor = None):
        if index is not None and self.grid_mask is not None:
            if self.share_parameter:
                # (B, N) = (B, N) * (1, N)
                return self.grid_mask[0][index]

            # (B, N) <- (B, G) | (B, N)
            return F.gather_d(self.grid_mask, -1, index)

        return self.grid_mask

    def calc_ds_gaussian(self, gaussian: Tensor, index: Tensor = None):
        """calculate :math:`\\Delta S_i \\cdot g(s)`"""
        ds = self.get_spacing(index)
        # (B, G) * (G) OR (B, N) * (N)
        return gaussian * ds

    def add_gaussian(self, colvar: Tensor) -> Tensor:
        """add gaussian to hills"""
        # (B, G) or (B, N)
        gaussian_exp, index = self.calc_basis_exp(colvar)
        gaussian = F.exp(gaussian_exp)
        mask = self.get_mask(index)
        if mask is not None:
            gaussian_exp = msnp.where(mask, gaussian_exp, self.large_neg)
            gaussian = F.select(mask, gaussian, F.zeros_like(gaussian))
        # :math:`\Delta S_i g_i(s)`
        ds = self.get_spacing(index)
        ds_gaussian = gaussian * ds
        log_ds = F.log(ds)
        # log(\Delta S_i f_i)
        log_ds_gauss = log_ds + gaussian_exp
        # log(\Delta S_i \sqrt(f_i))
        log_ds_sqrt_gauss = log_ds + gaussian_exp * 0.5

        # (B, 1) <- {(B, G) or (B, N)}
        # :math:`\log(\sum_i {\Delta S_i f_i})`
        log_sum_gauss = F.logsumexp(log_ds_gauss, -1)
        # :math:`\log(\sum_i {\Delta S_i \sqrt(f_i)})`
        log_sum_sqrt_gauss = F.logsumexp(log_ds_sqrt_gauss, -1)

        # (B, 1)
        inv_ratio = F.exp(log_sum_sqrt_gauss - log_sum_gauss)
        sum_gaussian = msnp.sum(gaussian, -1, keepdims=True)

        non_zero_mask = sum_gaussian > 0
        inv_ratio = msnp.where(non_zero_mask, inv_ratio, 0)

        if self.project_to_grids:
            # (B, 1)
            max_gaussian = msnp.amax(gaussian, -1, True)
            max_gaussian2 = F.square(max_gaussian)
            max_gaussian_sqrt = F.sqrt(max_gaussian)

            sum_gaussian2 = msnp.sum(F.square(gaussian), -1, keepdims=True)
            sum_gaussian_sqrt = msnp.sum(F.sqrt(gaussian), -1, keepdims=True)

            max_ratio = (sum_gaussian / max_gaussian - 1) / (self.num_max - 1)
            max2_ratio = (sum_gaussian2 / max_gaussian2 - 1) / (self.num_max2 - 1)
            max_sqrt_ratio = (sum_gaussian_sqrt / max_gaussian_sqrt - 1) / (self.num_max_sqrt - 1)

            max_ratio = F.select(max_ratio > 1, F.ones_like(max_ratio), max_ratio)
            max2_ratio = F.select(max_ratio > 1, F.ones_like(max2_ratio), max2_ratio)
            max_sqrt_ratio = F.select(max_ratio > 1, F.ones_like(max_sqrt_ratio), max_sqrt_ratio)

            ratio = msnp.where(max_ratio > 0.5,
                               max_ratio / max_sqrt_ratio,
                               max2_ratio / max_sqrt_ratio)

            inv_norm = self.inv_power2d * self.inv_conv_norm * F.square(inv_ratio) * ratio

            factors = gaussian * inv_norm

            # (B, 1) < {(B, G) or (B, N)}
            max_factor = msnp.amax(factors, -1, True)
            if(max_factor > self.max_factor)[0]:
                print(max_factor)
            scale = msnp.where(max_factor > self.max_factor, self.max_factor / max_factor, 1)
            factors = factors * scale
            inv_norm = inv_norm * scale

            ds_gaussian = ds_gaussian * inv_norm * self.conv_norm
        else:
            inv_norm = self.inv_sqrt2d * self.inv_conv_norm * inv_ratio
            factors = gaussian * inv_norm

        # (B, G) or (B, N)
        fit_factors = self.height * factors

        old_weights = 0
        if self.well_temped or self.sinking:
            # (B, G) or (B, N)
            old_weights = self.get_weights(index)

        if self.well_temped:
            # (B, 1) <- (B, G) or (B, N)
            old_bias = keepdims_sum(old_weights * ds_gaussian, -1)
            # -1 / (\gamma - 1) * \beta * V(s)
            fit_factors *= F.exp(-self.wt_factor * old_bias)

        if self.use_cutoff:
            empty_hills = F.zeros((colvar.shape[0], self.num_grids), ms.float32)
            # (B, G) <- (B, N)
            fit_factors = F.tensor_scatter_elements(empty_hills, index, fit_factors, -1)

        if self.share_parameter and colvar.shape[0] > 1:
            # (1, G) <- (B, G) OR (1, N) <- (B, N)
            fit_factors = keepdims_sum(fit_factors, 0)

        new_weights = F.assign_add(self.hills, fit_factors)

        if self.reweighting_factor is not None:
            rct = self.calc_reweight_factor()

        bias_depth = 0
        if self.sinking:
            # (B, G) or (B, N)
            # (B, 1) <- [(B, G) or (B, N)]
            bias = keepdims_sum(new_weights * ds_gaussian, -1)
            if self.share_parameter and bias.shape[0] > 1:
                # (1, 1) <- (B, 1)
                bias = msnp.max(bias, 0, True)

            # (B, 1) or (1, 1)
            max_bias_cond = bias > self.max_bias
            max_bias = msnp.where(max_bias_cond, bias, self.max_bias)
            bias_depth = msnp.where(max_bias_cond, max_bias + self.sink_depth, self.bias_depth)
            shift_factor = msnp.where(max_bias_cond, bias_depth * self.inv_sink_norm, self.shift_factor)

            if self.scale_sink_factor:
                # (B, 1) or (1, 1)
                max_weight = msnp.amax(self.hills, -1, True)
                scale_cond = max_weight > shift_factor
                shift_factor = msnp.where(scale_cond, max_weight, shift_factor)
                bias_depth = msnp.where(scale_cond, shift_factor * self.sink_norm, bias_depth)
                max_bias = msnp.where(scale_cond, bias_depth - self.sink_depth, max_bias)

            fit_factors = F.depend(fit_factors, F.assign(self.max_bias, max_bias))
            fit_factors = F.depend(fit_factors, F.assign(self.bias_depth, bias_depth))
            fit_factors = F.depend(fit_factors, F.assign(self.shift_factor, shift_factor))

        if self.reweighting_factor is not None:
            fit_factors = F.depend(fit_factors, F.assign(self.reweighting_factor, rct - bias_depth))

        return fit_factors

    def update(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tensor:
        """update parameter of bias potential"""
        coordinate = F.stop_gradient(coordinate)
        if pbc_box is not None:
            pbc_box = F.stop_gradient(pbc_box)
        colvar = self.colvar(coordinate, pbc_box)
        return self.add_gaussian(colvar)

    @jit
    def calc_metad_bias(self, colvar: Tensor, shift: Tensor = 0) -> Tensor:
        """calculate bias potential of MetaD"""
        if self.use_cutoff:
            # (B, N) <- (B, S)
            gaussian, index = self.calc_neighbour_gaussian(colvar)
        else:
            # (B, G) <- (B, S)
            gaussian = self.calc_grids_gaussian(colvar)
            index = None

        # (B, G) or (B, N)
        weights = self.get_weights(index, shift, self.grid_mask)
        ds_gaussian = self.calc_ds_gaussian(gaussian, index)

        # (B, G) * (B, G) OR (B, N) * (B, N)
        bias = weights * ds_gaussian
        # (B, 1) <- (B, G) or (B, N)
        return keepdims_sum(bias, -1)

    @jit
    def calc_bias(self, colvar: Tensor) -> Tensor:
        """calculate bias potential by colvar"""
        shift = 0
        if self.sinking:
            # pylint:disable=invalid-unary-operand-type
            shift = -self.shift_factor

        bias = self.calc_metad_bias(colvar, shift)

        if self.reweighting_factor is None:
            return bias

        # (B, 1) - (B, 1)
        return bias - self.reweighting_factor

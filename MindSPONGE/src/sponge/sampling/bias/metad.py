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

from typing import Tuple
import itertools
import numpy as np
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
from ...function import get_ms_array, get_integer, periodic_difference, keepdims_sum


class Metadynamics(Bias):
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
                                When None is given, WT-MetaD is not used. Default: ``None``.

        share_parameter (bool): Whether to share Metadynamics parameter for all walkers.
                                If False is given, then num_walker must be given.
                                Default: ``True``.

        num_walker (int):       Number of multiple walkers. Default: ``None``.

        use_cutoff (bool):      Whether to use cutoff when calculating gaussian from grids.
                                Default: ``True``.

        dp2cutoff (float):      Cutoff for grids. Default: 6.25

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: ``None``.

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Colvar,
                 update_pace: int,
                 grid_min: float,
                 grid_max: float,
                 grid_bin: int,
                 height: float,
                 sigma: float,
                 bias_factor: float = None,
                 temperature: float = 300,
                 share_parameter: bool = True,
                 num_walker: int = None,
                 use_cutoff: bool = True,
                 dp2cutoff: float = 6.25,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):

        super().__init__(
            name='metadynamics',
            colvar=colvar,
            update_pace=update_pace,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )

        if self.colvar.ndim != 1:
            raise ValueError(f'The rank (ndim) of the colvar used in Metdyanmics must be 1 '
                             f'but got: {self.colvar.ndim}')

        self.dp2cutoff = Tensor(dp2cutoff, ms.float32)
        self.use_cutoff = use_cutoff

        # S: dimension of the collective variables
        self.dim_colvar = self.colvar.shape[-1]

        self.share_parameter = share_parameter
        self.num_walker = get_integer(num_walker)
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

        self.periodic_mask = None if self.all_periodic else self.colvar.periodic

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

        # (S)
        self.sigma0 = _check_dimension(sigma, 'sigma', ms.float32)
        self.height = get_ms_array(height, ms.float32)
        if self.height.size != 1:
            raise ValueError(f'The size of height must be 1 but got: {self.height.size}')

        self.grid0_min = _check_dimension(grid_min, 'grid_min', ms.float32)
        self.grid0_max = _check_dimension(grid_max, 'grid_max', ms.float32)

        self.grid0_range = self.grid0_max - self.grid0_min
        grid0_bin = _check_dimension(grid_bin, 'grid_bin', ms.int32)
        self.grid_space = self.grid0_range / grid0_bin

        self.grid0_bin = msnp.where(self.colvar.periodic, grid0_bin, grid0_bin + 1)

        self.sigma = self.sigma0 / msnp.sqrt(2.0, ms.float32)
        self.coeff = -0.5 / F.square(self.sigma)
        cutoff_bins = msnp.ceil(msnp.sqrt(2 * self.dp2cutoff) * self.sigma / self.grid_space)
        cutoff_bins = F.cast(cutoff_bins, ms.int32)
        cutoff = self.grid_space * cutoff_bins

        ex_range = msnp.where(self.colvar.periodic, 0, cutoff)
        self.grid_min = self.grid0_min - ex_range
        self.grid_max = self.grid0_max + ex_range
        self.grid_range = self.grid0_range + 2 * ex_range

        self.grid_bin = self.grid0_bin + msnp.where(self.colvar.periodic, 0, 2 * cutoff_bins)

        grid0_min = self.grid0_min.asnumpy()
        grid0_max = self.grid0_max.asnumpy()
        grid0_bin = self.grid0_bin.asnumpy()
        grid_min = self.grid_min.asnumpy()
        grid_max = self.grid_max.asnumpy()
        grid_bin = self.grid_bin.asnumpy()

        grid_space = self.grid_space.asnumpy()
        cutoff_bins = cutoff_bins.asnumpy()

        grids0 = []
        grids = []
        neigh_grids_shift = []

        for i in range(self.dim_colvar):

            neigh_grids_shift.append(np.arange(-cutoff_bins[i], cutoff_bins[i] + 1, dtype=np.int32))
            if self.colvar.periodic[i]:
                grids0.append(np.arange(0, grid0_bin[i]) * grid_space[i] + grid0_min[i])
                grids.append(np.arange(0, grid_bin[i]) * grid_space[i] + grid_min[i])
            else:
                grids0.append(np.linspace(grid0_min[i], grid0_max[i], grid0_bin[i]))
                grids.append(np.linspace(grid_min[i], grid_max[i], grid_bin[i]))

        grids0 = tuple(itertools.product(*grids0))
        # (G_0, S)
        self.grids0 = Tensor(grids0, ms.float32)
        # G_0
        self.num_grids0 = self.grids0.shape[0]
        # (G_0)
        self.grid0_factor = msnp.cumprod(self.grid0_bin[::-1], axis=-1)
        self.grid0_factor = msnp.concatenate((self.grid0_factor[1::-1], Tensor([1], ms.int32)), axis=-1)

        # (G, S)
        grids = tuple(itertools.product(*grids))
        self.grids = Tensor(grids, ms.float32)
        # G
        self.num_grids = self.grids.shape[0]
        # (G)
        self.grid_factor = msnp.cumprod(self.grid_bin[::-1], axis=-1)
        self.grid_factor = msnp.concatenate((self.grid_factor[:-1][::-1], Tensor([1], ms.int32)), axis=-1)

        # (1, G)
        self.full_index = msnp.arange(self.num_grids, dtype=ms.int32).reshape(1, -1)

        # (N, S)
        neigh_grids_shift = tuple(itertools.product(*neigh_grids_shift))
        self.neigh_grids_shift = Tensor(neigh_grids_shift, ms.int32)
        self.num_neigh = self.neigh_grids_shift.shape[0]

        #  (1, G) or (B, G)
        self.hills = Parameter(msnp.zeros((self.num_parameter, self.num_grids), ms.float32),
                               name="hills", requires_grad=False)

        # (G_0, S)
        grids0_index = self.get_nearest_grid(self.grids0)
        # (G_0, N, S)
        _, grids0_neigh_index = self.get_neighbours(grids0_index)
        # (G_0, N)
        self.grids0_neigh_index = self.get_hills_index(grids0_neigh_index)

        # (N, S)
        neigh_diff = self.neigh_grids_shift * self.grid_space
        # (N) <- (N, S)
        self.neigh_gaussian = self.height * F.exp(F.reduce_sum(self.coeff * F.square(neigh_diff), -1))

        self.temperature = temperature
        self.kbt = self.units.boltzmann * temperature
        self.beta0 = 1.0 / self.kbt

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

    @property
    def boltzmann(self) -> float:
        """Boltzmann constant"""
        return self.units.boltzmann

    @property
    def periodic(self) -> Tensor:
        """periodic of collectiva variables"""
        return self.colvar.periodic

    @jit
    def calc_bias(self, colvar: Tensor) -> Tensor:
        """calculate bias potential by colvar"""
        if self.use_cutoff:
            # (B, N) <- (B, S)
            gaussian, index = self.calc_neighbour_gaussian(colvar)
        else:
            # (B, G) <- (B, S)
            gaussian = self.calc_grids_gaussian(colvar)
            index = None

        # (B, G) or (B, N)
        weights = self.get_weights(index)

        # (B, G) * (B, G) OR (B, N) * (B, N)
        bias = weights * self.height * gaussian
        # (B, 1) <- (B, G) or (B, N)
        bias = keepdims_sum(bias, -1)

        if self.reweighting_factor is None:
            return bias

        # (B, 1) - (B, 1)
        return bias - self.reweighting_factor

    def get_weights(self, index: Tensor = None) -> Tensor:
        r"""get weights by index of hills.

        Args:
            index (Tensor):     Tensor of shape (B, N). Data type is int.
                                Index hills.
                                If None is given, weights of the full hills will be return.
                                Default: ``None``.

        Returns:
            weight (Tensor):    Tensor of shape (B, N) or (B, G). Data type is float.
                                Value of neighbouring grids.

        """
        if index is None:
            # (B, G)
            return self.hills
        if self.share_parameter:
            # (1, N)
            return self.hills[0][index]
        # (B, N) <- (B, G) | (B, N)
        return F.gather_d(self.hills, -1, index)

    def get_gaussians(self):
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
        grids = self.grid_min + index * self.grid_space

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
        nearest_grid = F.floor((colvar - self.grid_min) / self.grid_space + 0.5)
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
        grid *= self.grid_factor
        return grid.sum(-1)

    def calc_neighbour_gaussian(self, colvar: Tensor) -> Tuple[Tensor, Tensor]:
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

        return gaussian, indices

    def calc_grids_gaussian(self, colvar: Tensor) -> Tensor:
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
            diff = periodic_difference(diff, self.grid_range, self.periodic_mask)

        # (B, G) <- (B, G, S) = (S) * (B, G, S)
        gaussian = F.exp(F.reduce_sum(self.coeff * F.square(diff), -1))

        return gaussian

    def calc_reweight_factor(self) -> Tensor:
        r"""calculate the reweighting factor :math:`c(t)` of metadynamics

        Returns:
            rct (Tensor):   Tensor of shape `(B, 1)`. Data type is float.
                            Reweighting factor :math:`c(t)`.
        """
        if self.reweighting_factor is None:
            return None
        # (B, G_0, N) <- (B, G) | (G_0, N)
        weights = F.gather(self.hills, self.grids0_neigh_index, -1)
        # (B, G_0, N) = (B, G_0, N) * (N)
        biases = F.stop_gradient(weights) * self.neigh_gaussian
        # (B, G_0) <- (B, G_0, N)
        biases = F.reduce_sum(biases, -1)

        # \gamma / (\gamma - 1) * \beta * V(t)
        rct0 = self.wt_factor0 * biases
        # 1 / (\gamma - 1) * \beta * V(t)
        rct1 = self.wt_factor * biases

        # (B, 1) <- (B, G_0)
        rct0 = F.logsumexp(rct0, -1, True)
        rct1 = F.logsumexp(rct1, -1, True)
        rct = (rct0 - rct1) * self.kbt
        return rct

    def add_gaussian(self, colvar: Tensor) -> Tensor:
        """add gaussian to hills"""
        if self.use_cutoff:
            # (B, N) <- (B, S)
            gaussian, index = self.calc_neighbour_gaussian(colvar)
        else:
            # (B, G) <- (B, S)
            gaussian = self.calc_grids_gaussian(colvar)
            index = None

        # (B, 1) <- (B, G) or (B, N)
        gnorm = msnp.reciprocal(keepdims_sum(F.square(gaussian), -1))

        # (B, G) or (B, N)
        new_hills = gaussian * gnorm

        if self.well_temped:
            # (B, G) or (B, N)
            hills = self.get_weights(index)
            # (B, 1) <- (B, G) or (B, N)
            bias = keepdims_sum(hills * self.height * gaussian, -1)
            # -1 / (\gamma - 1) * \beta * V(s)
            new_hills *= F.exp(-self.wt_factor * bias)

        if self.use_cutoff:
            empty_hills = F.zeros((colvar.shape[0], self.num_grids), ms.float32)
            # (B, G) <- (B, N)
            new_hills = F.tensor_scatter_elements(empty_hills, index, new_hills, -1)

        if self.share_parameter and colvar.shape[0] > 1:
            # (1, G) <- (B, G) OR (1, N) <- (B, N)
            new_hills = keepdims_sum(new_hills, 0)

        new_hills = F.assign_add(self.hills, new_hills)

        if self.reweighting_factor is not None:
            new_hills = F.depend(new_hills, self.update_reweight_factor())

        return new_hills

    def update_reweight_factor(self) -> Tensor:
        if self.reweighting_factor is None:
            return None
        return F.assign(self.reweighting_factor, self.calc_reweight_factor())

    def update(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tensor:
        """update parameter of bias potential"""
        coordinate = F.stop_gradient(coordinate)
        if pbc_box is not None:
            pbc_box = F.stop_gradient(pbc_box)
        colvar = self.colvar(coordinate, pbc_box)
        return self.add_gaussian(colvar)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate bias potential.

        Args:
            coordinate (Tensor):           Tensor of shape `(B, A, D)`. Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape `(B, A, N)`. Data type is int.
                                            Index of neighbour atoms. Default: ``None``.
            neighbour_mask (Tensor):        Tensor of shape `(B, A, N)`. Data type is bool.
                                            Mask for neighbour atoms. Default: ``None``.
            neighbour_vector (Tensor):       Tensor of shape `(B, A, N)`. Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distance (Tensor):   Tensor of shape `(B, A, N)`. Data type is float.
                                            Distance between neigh_shift atoms. Default: ``None``.
            pbc_box (Tensor):               Tensor of shape `(B, D)`. Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            potential (Tensor): Tensor of shape `(B, 1)`. Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        colvar = self.colvar(coordinate, pbc_box)

        return self.calc_bias(colvar)

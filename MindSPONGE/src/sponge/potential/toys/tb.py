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
"""Potential"""

from typing import Union, List, Tuple
import numpy as np
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from .. import PotentialCell
from ...function import get_ms_array, keepdims_sum


class TiwaryBerne(PotentialCell):
    r"""
    Potential energy of a toy model developed by Tiwary and Berne.

    Reference:
        Tiwary, P.; Berne, B. J.
        Predicting Reaction Coordinates in Energy Landscapes with Diffusion Anisotropy [J].
        The Journal of Chemical Physics, 2017, 147(15): 152701.

    Args:
        location: Union[Tensor, ndarray, List[float], Tuple[float]]:
            Array of location(s) of metastable state(s) on the potential energy surface (PES).
            The shape of the array is `(S, D)`, and the data type is float.
            Default: ((-0.5, 0.5), (0.8, 1.2), (0.5, -0.3))

        depth: Union[Tensor, ndarray, List[float], Tuple[float]]:
            Array of depth of metastable state(s) on the potential energy surface (PES).
            The shape of the array is `(S)`, and the data type is float.
            Default: (16, 18, 16)

        name (str): Name of the energy. Default: 'tiwary_berne'

    Returns:
        energy (Tensor), Tensor of shape `(B, 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        S:  Number of metastable state(s).
        D:  Spatial dimension of the toy model. Usually is 2.

    """
    def __init__(self,
                 location: Union[Tensor, ndarray, List[float], Tuple[float]] = ((-0.5, 0.5),
                                                                                (0.8, 1.2),
                                                                                (0.5, -0.3)),
                 depth: Union[Tensor, ndarray, List[float], Tuple[float]] = (16, 18, 16),
                 base_energy: float = 20,
                 name: str = 'tiwary_berne',
                 ):

        super().__init__(
            num_energies=1,
            name=name,
        )

        # (S, D)
        self.location: Tensor = get_ms_array(location, ms.float32)
        self.dimension = self.location.shape[-1]

        if self.location.ndim == 1:
            # (1, D) <- (D)
            self.location = F.expand_dims(self.location, 0)

        # S
        num_states = self.location.shape[-2]

        # (S)
        self.depth: Tensor = get_ms_array(depth, ms.float32)
        if self.depth.shape[-1] != num_states and self.depth.shape[-1] != 1:
            raise ValueError(f'The number of depth {self.depth.shape[-1]} does not match '
                             f'the number of states {num_states}')

        self.base_energy = get_ms_array(base_energy, ms.float32)

        self.split = ops.Split(-1, 2)

    def get_contour_2d(self,
                       vmin: float = -1,
                       vmax: float = 1.5,
                       num_grids: int = 50,
                       ) -> Tuple[ndarray, ndarray, ndarray]:
        """get the data to plot the counter of PES for 2-D system"""
        if self.dimension != 2:
            raise ValueError(f'The function `get_contour_2d` can only be used in a 2-D system, '
                             f'but the dimension of the potential energy is {self.dimension}.')

        grids = np.linspace(vmin, vmax, num_grids)
        x, y = np.meshgrid(grids, grids)
        coordinate = np.stack((x.ravel(), y.ravel()), 1)
        coordinate = np.expand_dims(coordinate, -2)
        z = self.construct(Tensor.from_numpy(coordinate)).reshape(num_grids, num_grids)
        z = F.reshape(z, (num_grids, num_grids)).asnumpy()
        z -= np.min(z)

        return x, y, z

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ) -> Tensor:
        r"""Calculate potential energy.

        Args:
            coordinates (Tensor):           Tensor of shape (B, A, 2). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: ``None``.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: ``None``.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N, 2). Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distances (Tensor):   Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: ``None``.
            pbc_box (Tensor):               Tensor of shape (B, 2). Data type is float.
                                            Tensor of PBC box. Default: ``None``.

        Returns:
            potential (Tensor): Tensor of shape (B, E). Data type is float.

        Note:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            E:  Number of energy terms.

        """
        #pylint: disable=unused-argument

        # (B, A, S, D) = (B, A, 1, D) - (S, D)
        diff = F.expand_dims(coordinate, -2) - self.location

        # (B, A, S) <- (B, A, S, D)
        diff2 = F.reduce_sum(F.square(diff), -1)

        # (B, A, S) = (S) * (B, A, S)
        energy = -1 * self.depth * F.exp(-2 * diff2)
        # (B, A) <- (B, A, S)
        energy = F.reduce_sum(energy, -1)

        # (B, A) <- (B, A, D)
        restraint = 0.5 * F.reduce_sum(F.pow(coordinate, 6), -1)

        # (B, A)
        energy += restraint
        energy += self.base_energy

        # (B, 1) <- (B, A)
        return keepdims_sum(energy, -1)

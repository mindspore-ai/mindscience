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
"""
Collective variables that accept index
"""

import mindspore as ms
from mindspore.ops import functional as F
from mindspore import nn
from mindspore.common import Tensor
from mindspore import numpy as msnp

from ..function import functions as func
from .colvar import Colvar


class IndexColvar(Colvar):
    r"""Collective variables based on index

    Args:
        dim_output (int):   The output dimension, i.e., the last dimension of output Tensor.

        periodic (bool):    Whether the CV is periodic or not. Default: False

        use_pbc (bool):     Whether to calculate the CV at periodic boundary condition (PBC).
                            If "None" is given, it will be determined at runtime based on
                            whether the "pbc_box" is given or not. Default: None

        length_unit (str):  Length unit for position coordinates.
                            If "None" is given, it will use the global units. Default: None

   """

    def __init__(self,
                 dim_output: int,
                 periodic: bool = False,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 ):

        super().__init__(
            dim_output=dim_output,
            periodic=periodic,
            use_pbc=use_pbc,
            length_unit=length_unit,
        )

    def construct(self, coordinate: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        #pylint: disable=arguments-differ
        raise NotImplementedError


class IndexDistances(IndexColvar):
    r"""Calculate distance between atoms by neighbour index

    Args:
        use_pbc (bool):     Whether to use periodic boundary condition. Default: False

        length_unit (str):  Length unit. Default: None

        large_dis (float):  A large value that added to the distance equal to zero to
                            prevent them from becoming zero values after Norm operation,
                            which could lead to auto-differentiation errors.

        keep_dims (bool):   If this is "True", the last axis will be left in the result as
                            dimensions with size one.

    """

    def __init__(self,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 large_dis: float = 100,
                 keep_dims: bool = False,
                 ):

        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            length_unit=length_unit,
        )

        self.norm_last_dim = nn.Norm(-1, keep_dims=keep_dims)
        self.large_dis = Tensor(large_dis, ms.float32)

    def construct(self, coordinate: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        r"""Compute distances between atoms according to index.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Coordinate of system
            index (Tensor):         Tensor of shape (B, A, N). Data type is int.
                                    Neighbour index
            mask (Tensor):          Tensor of shape (B, A, N). Data type is bool.
                                    Mask of neighbour index
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Periodic boundary condition Box.
                                    Default: None

        Returns:
            distances (Tensor):     Tensor of shape (B, A, N). Data type is float.

        Symbols:

            B:  Batchsize, i.e. number of simulation walker.
            A:  Number of atoms.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates.

        """

        # (B,A,1,D) <- (B,A,D)
        atoms = F.expand_dims(coordinate, -2)
        # (B,A,N,D) <- (B,A,D)
        neighbours = func.gather_vectors(coordinate, index)
        vectors = self.get_vector(atoms, neighbours, pbc_box)

        # Add a non-zero value to the vectors whose mask value is False
        # to prevent them from becoming zero values after Norm operation,
        # which could lead to auto-differentiation errors
        if mask is not None:
            # (B,A,N,D) = (B,A,N,D) + (B,A,N,1)
            vectors += F.expand_dims(msnp.where(mask, 0, self.large_dis), -1)

        # (B,A,N) = (B,A,N,D)
        return self.norm_last_dim(vectors)


class IndexVectors(IndexColvar):
    r"""Get vectors by index

    Args:
        use_pbc (bool):     Whether to use periodic boundary condition. Default: False

        length_unit (str):  Length unit. Default: None

    """

    def __init__(self,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 ):

        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            length_unit=length_unit,
        )

    def construct(self, coordinate: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        r"""get vector by index.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Coordinate of system
            index (Tensor):         Tensor of shape (B, A, N). Data type is int.
                                    Neighbour index
            mask (Tensor):          Tensor of shape (B, A, N). Data type is bool.
                                    Mask of neighbour index
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Periodic boundary condition Box.
                                    Default: None

        Returns:
            vector (Tensor):        Tensor of shape (B, A, D). Data type is float.

        Symbols:

            B:  Batchsize, i.e. number of simulation walker.
            A:  Number of atoms.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates.

        """

        # (B,A,1,D) <- (B,A,D)
        atoms = F.expand_dims(coordinate, -2)
        # (B,A,N,D) <- (B,A,D)
        neighbours = func.gather_vectors(coordinate, index)

        return self.get_vector(atoms, neighbours, pbc_box)

# Copyright 2024 Huawei Technologies Co., Ltd
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
"""spherical_basis"""

import mindspore as ms
import mindspore.mint as mint
from .radial_basis import GaussianSmearing


class CircularBasisLayer(ms.nn.Cell):
    r"""
    2D Fourier Bessel Basis

    Args:
        num_spherical (int): Controls maximum frequency.
        radial_basis (int): RadialBasis
        cbf (str): Name of the cosine basis function
        efficient (bool): Whether to use the "efficient" summation order

    Inputs:
        - **d_ca** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **y_l_m** (Tensor) - The shape of tensor is :math:`(num\_spherical, total\_triplets)`.

    Outputs:
        - **rbf** (Tensor) - The shape of tensor is :math:`(total\_edges, num\_radial)`.
        - **cbf** (Tensor) - The shape of tensor is :math:`(total\_triplets, num\_spherical)`.

    Raises:
         ValueError: if cbf is not "gaussian" or "spherical_harmonics".
    """

    def __init__(
            self,
            num_spherical,
            radial_basis,
            cbf_name,
            efficient=False,
    ):
        super().__init__()

        self.radial_basis = radial_basis
        self.efficient = efficient

        self.cbf_name = cbf_name.lower()
        self.num_spherical = num_spherical
        if self.cbf_name == "gaussian":
            self.cosfi_basis = GaussianSmearing(
                start=-1, stop=1, num_gaussians=self.num_spherical
            ).astype(ms.float32)

        elif self.cbf_name == "spherical_harmonics":
            self.cosfi_basis = 0
        else:
            raise ValueError(
                f"Unknown cosine basis function '{self.cbf_name}'.")

    def construct(self, d_ca, y_l_m):
        """circular basis construct"""
        if self.cbf_name == "gaussian":
            cbf = self.cosfi_basis

        elif self.cbf_name == "spherical_harmonics":
            cbf = y_l_m

        rbf = self.radial_basis(d_ca)
        return rbf, mint.permute(cbf, (1, 0))

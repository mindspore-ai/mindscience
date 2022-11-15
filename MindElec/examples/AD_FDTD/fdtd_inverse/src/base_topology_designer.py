# Copyright 2021 Huawei Technologies Co., Ltd
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
#pylint: disable=W0613
"""
Basic class for topology designer.
"""
from mindspore import nn
from .utils import ones, tensor


class BaseTopologyDesigner(nn.Cell):
    """Basic class for Differentiable Topology Designer

    Note:
        Modify the generate_object() and set_source_location() when solving user-defined problems.

    Args:
        cell_numbers (tuple): Number of Yee cells in the (x,y,z) directions.
        cell_lengths (tuple): Lengths of Yee cells.
        background_epsr (float): Relative permittivity of the background. Default: 1.
    """

    def __init__(self, cell_numbers, cell_lengths, background_epsr=1.):
        super(BaseTopologyDesigner, self).__init__(auto_prefix=False)
        self.cell_numbers = cell_numbers
        self.cell_lengths = cell_lengths
        self.background_epsr = tensor(background_epsr)
        self.background_sige = tensor(0.)
        self.grid = ones(self.cell_numbers)

    def construct(self, rho):
        """Generate material tensors.

        Args:
            rho (Parameter): Parameters to be optimized in the inversion domain.

        Returns:
            epsr (Tensor, shape=(self.cell_nunbers)): Relative permittivity in the whole domain.
            sige (Tensor, shape=(self.cell_nunbers)): Conductivity in the whole domain.
        """
        return self.generate_object(rho)

    def generate_object(self, rho):
        """Generate material tensors.

        Args:
            rho (Parameter): Parameters to be optimized in the inversion domain.

        Returns:
            epsr (Tensor, shape=(self.cell_nunbers)): Relative permittivity in the whole domain.
            sige (Tensor, shape=(self.cell_nunbers)): Conductivity in the whole domain.
        """
        # background material tensors
        epsr = self.background_epsr * self.grid
        sige = self.background_sige * self.grid

        # ----------------------------------
        # Customized Differentiable Mapping
        # ----------------------------------
        epsr = rho

        return epsr, sige

    def modify_object(self, *args):
        """
        Generate special objects, such as PEC or lumped elements.

        Args:
            args: material tensors.

        Returns:
            args: material tensors.
        """
        return args

    def update_sources(self, *args):
        """
        Set locations of sources.

        Args:
            args: source tensors.

        Returns:
            args: source tensors.
        """
        return args

    def get_outputs_at_each_step(self, *args):
        """Compute output each step.

        Note:
            Modify this function when solving user-defined problems.

        Args:
            args: field tensors.

        Returns:
            Customized outputs. Default: args.
        """
        return args

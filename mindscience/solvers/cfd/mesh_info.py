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
"""mesh information container."""
from mindspore import numpy as mnp
from mindspore import jit_class


@jit_class
class MeshInfo:
    r"""
    Mesh information container.

    Args:
        config (dict): The dict of parameters.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindflow import cfd
        >>> config = {'dim': 1, 'nx': 100, 'gamma': 1.4, 'x_range': [0, 1], 'pad_size': 3}
        >>> m = cfd.MeshInfo(config)
    """

    def __init__(self, config):
        self.dim = config.get("dim", None)
        self.nx = config.get("nx", 1)
        self.ny = config.get("ny", 1)
        self.nz = config.get("nz", 1)
        self.x_range = config.get("x_range", [0, 0])
        self.y_range = config.get("y_range", [0, 0])
        self.z_range = config.get("z_range", [0, 0])
        self.pad = config.get("pad_size", 0)

        self.active_axis = []
        for i in range(self.dim):
            self.active_axis.append(i)

        self.cell_sizes = [(self.x_range[1] - self.x_range[0]) / self.nx,
                           (self.y_range[1] - self.y_range[0]) / self.ny,
                           (self.z_range[1] - self.z_range[0]) / self.nz]
        self.number_of_cells = [self.nx, self.ny, self.nz]

    def mesh_xyz(self):
        """Compute the mesh center coordinates."""
        dx = (self.x_range[1] - self.x_range[0]) / self.nx
        dy = (self.y_range[1] - self.y_range[0]) / self.ny
        dz = (self.z_range[1] - self.z_range[0]) / self.nz

        x = mnp.linspace(self.x_range[0] + dx / 2, self.x_range[1] - dx / 2, self.nx)
        y = mnp.linspace(self.y_range[0] + dy / 2, self.y_range[1] - dy / 2, self.ny)
        z = mnp.linspace(self.z_range[0] + dz / 2, self.z_range[1] - dz / 2, self.nz)
        mesh_x, mesh_y, mesh_z = mnp.meshgrid(x, y, z, indexing='ij')

        return (mesh_x, mesh_y, mesh_z)

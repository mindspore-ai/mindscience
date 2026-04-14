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
"""Grid Helper for 3D FDTD"""


class GridHelper:
    """
    Helper class for placing entities on the FDTD grid.

    Args:
        cell_numbers (tuple): Number of Yee cells in (x,y,z) directions.
        cell_lengths (tuple): Lengths of Yee cells in (x,y,z) directions.
        origin (tuple): Node index of origin.
    """

    def __init__(self, cell_numbers, cell_lengths, origin) -> None:
        self.cell_numbers = cell_numbers
        self.cell_lengths = cell_lengths
        self.orig = origin

        self.objects_on_edges = []
        self.objects_on_faces = []
        self.objects_in_cells = []
        self.sources_on_edges = []
        self.dynamic_sources_on_edges = []
        self.voltage_monitors = []
        self.current_monitors = []

    def __setitem__(self, key, entity):
        """
        Add an entity in the grid.

        Args:
            key (tuple): location (indices or coordinates)
            entity (Resistor, VoltageSource, VMonitor, IMonitor)
        """
        entity.grid_registry(
            self,
            x=self.get_node_index_range(
                key[0], self.orig[0], self.cell_lengths[0], self.cell_numbers[0]
            ),
            y=self.get_node_index_range(
                key[1], self.orig[1], self.cell_lengths[1], self.cell_numbers[1]
            ),
            z=self.get_node_index_range(
                key[2], self.orig[2], self.cell_lengths[2], self.cell_numbers[2]
            ),
        )

    def get_node_index_range(self, loc, orig, d, n):
        """
        Convert int, float or slice to node index range.

        Args:
            loc (int, float or slice): indices or coordinates
            orig (int): node index of origin
            d (float): cell length
            n (int): cell number

        Returns:
            node index range: [start, stop)
        """
        node_index_range = None

        if isinstance(loc, int):
            # node index
            node_index_range = (loc + orig, loc + orig + 1)

        elif isinstance(loc, float):
            # coordinate
            start = int(loc / d) + orig
            node_index_range = (start, start + 1)

        elif isinstance(loc, slice):
            start = loc.start if loc.start is not None else 0
            start = self.get_node_index_range(start, orig, d, n)[0]
            stop = loc.stop if loc.stop is not None else n
            stop = self.get_node_index_range(stop, orig, d, n)[1]
            node_index_range = (start, stop)

        else:
            raise TypeError(f"key should be int, float or slice.")

        return node_index_range


class UniformBrick:
    """
    Uniform brick in cells.

    Args:
        epsr (float): Relative permittivity.
        sigma (float): Conductivity.

    Attributes:
        indices (list): lists of [start, stop)
    """

    def __init__(self, epsr=1., sigma=0.) -> None:
        self.epsr = epsr
        self.sigma = sigma
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Get indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): node index range ([start, stop)) in x direction
            y (tuple): node index range ([start, stop)) in y direction
            z (tuple): node index range ([start, stop)) in z direction
        """
        grid_helper.objects_in_cells.append(self)
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # convert node index to edge index
        i_e -= 1
        j_e -= 1
        k_e -= 1

        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]


class PECPlate:
    """
    PEC plates.

    Args:
        direction (str): Direction, choose from ['x', 'y', 'z']

    Attributes:
        epsr (float): Relative permittivity. Default: 1.
        sigma (float): Conductivity. Default: 1e10.
        indices (list of lists): [start, stop) of indices
    """

    def __init__(self, direction) -> None:
        self.direction = direction.lower()
        self.epsr = 1.
        self.sigma = 1e10
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Get indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): node index range ([start, stop)) in x direction
            y (tuple): node index range ([start, stop)) in y direction
            z (tuple): node index range ([start, stop)) in z direction
        """
        grid_helper.objects_on_faces.append(self)
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # convert node index to edge index
        i_e -= 1
        j_e -= 1
        k_e -= 1

        # check indices
        if self.direction == 'x' and i_e != i_s:
            raise ValueError(f'Starting node index is not -1 in direction {self.direction}')

        if self.direction == 'y' and j_e != j_s:
            raise ValueError(f'Starting node index is not -1 in direction {self.direction}')

        if self.direction == 'z' and k_e != k_s:
            raise ValueError(f'Starting node index is not -1 in direction {self.direction}')

        if self.direction not in ('x', 'y', 'z'):
            raise ValueError(f'Cannot match direction {self.direction}')

        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]

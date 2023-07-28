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
"""lumped elements for 3D FDTD"""
from .constants import epsilon0


class Resistor:
    """
    Resistor at edges.

    Args:
        r (float): Resistance.
        direction (str): Direction, 'x', 'y' or 'z'.

    Attributes:
        epsr (float): Equivalent epsr. Default: 0.
        sigma (float): Equivalent conductivity. Default: 1/r.
        indices (list): lists of [start, stop)
    """
    def __init__(self, r, direction) -> None:
        self.r = r
        self.direction = direction.lower()
        self.epsr = 0.
        self.sigma = 1. / r
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Set edge indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): nodex index range ([start, stop)) in x direction
            y (tuple): nodex index range ([start, stop)) in y direction
            z (tuple): nodex index range ([start, stop)) in z direction
        """
        grid_helper.objects_on_edges.append(self)
        dx, dy, dz = grid_helper.cell_lengths
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # convert node index to edge index
        if self.direction == 'x':
            i_e -= 1
            if i_e <= i_s:
                raise ValueError('i_e not larger than i_s')
            coef = ((i_e - i_s) * dx) / ((j_e - j_s) * (k_e - k_s) * dy * dz)

        elif self.direction == 'y':
            j_e -= 1
            if j_e <= j_s:
                raise ValueError('j_e not larger than j_s')
            coef = ((j_e - j_s) * dy) / ((i_e - i_s) * (k_e - k_s) * dx * dz)

        elif self.direction == 'z':
            k_e -= 1
            if k_e <= k_s:
                raise ValueError('k_e not larger than k_s')
            coef = ((k_e - k_s) * dz) / ((i_e - i_s) * (j_e - j_s) * dx * dy)

        else:
            raise ValueError(f'Cannot match direction {self.direction}.')

        self.sigma = coef / self.r
        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]


class Capacitor:
    """
    Capacitor at edges.

    Args:
        c (float): Capacitance.
        direction (str): Direction, 'x', 'y' or 'z'.

    Attributes:
        epsr (float): Equivalent epsr. Default: c/epsilon0.
        sigma (float): Equivalent conductivity. Default: 0.
        indices (list): lists of [start, stop)
    """
    def __init__(self, c, direction) -> None:
        self.c = c
        self.direction = direction.lower()
        self.epsr = c / epsilon0
        self.sigma = 0.
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Set edge indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): nodex index range ([start, stop)) in x direction
            y (tuple): nodex index range ([start, stop)) in y direction
            z (tuple): nodex index range ([start, stop)) in z direction
        """
        grid_helper.objects_on_edges.append(self)
        dx, dy, dz = grid_helper.cell_lengths
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # convert node index to edge index
        if self.direction == 'x':
            i_e -= 1
            if i_e <= i_s:
                raise ValueError('i_e not larger than i_s')
            coef = ((i_e - i_s) * dx) / ((j_e - j_s) * (k_e - k_s) * dy * dz)

        elif self.direction == 'y':
            j_e -= 1
            if j_e <= j_s:
                raise ValueError('j_e not larger than j_s')
            coef = ((j_e - j_s) * dy) / ((i_e - i_s) * (k_e - k_s) * dx * dz)

        elif self.direction == 'z':
            k_e -= 1
            if k_e <= k_s:
                raise ValueError('k_e not larger than k_s')
            coef = ((k_e - k_s) * dz) / ((i_e - i_s) * (j_e - j_s) * dx * dy)

        else:
            raise ValueError(f'Cannot match direction {self.direction}.')

        self.epsr = coef * self.c / epsilon0
        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]


class Inductor:
    """
    Inductor at edges.

    Args:
        l (float): Inductance.
        direction (str): Direction, 'x', 'y' or 'z'.

    Attributes:
        coef (float): updating coefficients.
        indices (list): lists of [start, stop)
    """
    def __init__(self, l, direction) -> None:
        self.l = l
        self.direction = direction.lower()
        self.coef = 1. / l
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Set edge indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): nodex index range ([start, stop)) in x direction
            y (tuple): nodex index range ([start, stop)) in y direction
            z (tuple): nodex index range ([start, stop)) in z direction
        """
        grid_helper.dynamic_sources_on_edges.append(self)
        dx, dy, dz = grid_helper.cell_lengths
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # convert node index to edge index
        if self.direction == 'x':
            i_e -= 1
            if i_e <= i_s:
                raise ValueError('i_e not larger than i_s')
            coef = ((i_e - i_s) * dx) / ((j_e - j_s) * (k_e - k_s) * dy * dz)

        elif self.direction == 'y':
            j_e -= 1
            if j_e <= j_s:
                raise ValueError('j_e not larger than j_s')
            coef = ((j_e - j_s) * dy) / ((i_e - i_s) * (k_e - k_s) * dx * dz)

        elif self.direction == 'z':
            k_e -= 1
            if k_e <= k_s:
                raise ValueError('k_e not larger than k_s')
            coef = ((k_e - k_s) * dz) / ((i_e - i_s) * (j_e - j_s) * dx * dy)

        else:
            raise ValueError(f'Cannot match direction {self.direction}.')

        self.coef = coef / self.l
        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]


class VoltageSource:
    """
    Voltage source at edges.

    Args:
        amplitude (float): Voltage.
        r (float): Resistance.
        polarization (str): Direction, 'xn', 'xp', 'yn', 'yp' or 'zn', 'zp'.

    Attributes:
        direction (str): 'x', 'y', or 'z'
        polar (float): +1 if polarization in ['xp', 'yp', 'zp'], else -1.
        epsr (float): Equivalent epsr. Default: 0.
        sigma (float): Equivalent conductivity. Default: 1/r.
        j (float): Averaged current. Default: amplitude/r.
        indices (list): lists of [start, stop)
    """

    def __init__(self, amplitude, r, polarization) -> None:
        self.amplitude = amplitude
        self.r = r
        self.direction = polarization.lower()[0]
        self.polar = -1. if polarization.lower()[-1] == 'n' else 1.
        self.epsr = 0.
        self.sigma = 1. / r
        self.j = amplitude / r
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Set edge indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): nodex index range ([start, stop)) in x direction
            y (tuple): nodex index range ([start, stop)) in y direction
            z (tuple): nodex index range ([start, stop)) in z direction
        """
        grid_helper.objects_on_edges.append(self)
        grid_helper.sources_on_edges.append(self)
        dx, dy, dz = grid_helper.cell_lengths
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # convert node index to edge index
        if self.direction == 'x':
            i_e -= 1
            if i_e <= i_s:
                raise ValueError('i_e not larger than i_s')
            area = (j_e - j_s) * (k_e - k_s) * dy * dz
            coef = ((i_e - i_s) * dx) / area

        elif self.direction == 'y':
            j_e -= 1
            if j_e <= j_s:
                raise ValueError('j_e not larger than j_s')
            area = (i_e - i_s) * (k_e - k_s) * dx * dz
            coef = ((j_e - j_s) * dy) / area

        elif self.direction == 'z':
            k_e -= 1
            if k_e <= k_s:
                raise ValueError('k_e not larger than k_s')
            area = (i_e - i_s) * (j_e - j_s) * dx * dy
            coef = ((k_e - k_s) * dz) / area

        else:
            raise ValueError(f'Cannot match direction {self.direction}.')

        self.sigma = coef / self.r
        self.j = self.amplitude / self.r / area * self.polar
        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]


class VoltageMonitor:
    """
    Voltage monitor at edges.

    Args:
        polarization (str): 'xn', 'xp', 'yn', 'yp' or 'zn', 'zp'.

    Attributes:
        direction (str): 'x', 'y' or 'z'
        polar (float): +1 if polarization in ['xp', 'yp', 'zp'], else -1.
        coef (float): average factor
        indices (list): lists of [start, stop)
    """
    def __init__(self, polarization) -> None:
        self.direction = polarization.lower()[0]
        self.polar = -1. if polarization.lower()[-1] == 'n' else 1.
        self.coef = 1.
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Set edge indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): nodex index range ([start, stop)) in x direction
            y (tuple): nodex index range ([start, stop)) in y direction
            z (tuple): nodex index range ([start, stop)) in z direction
        """
        grid_helper.voltage_monitors.append(self)
        dx, dy, dz = grid_helper.cell_lengths
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # convert node index to edge index
        if self.direction == 'x':
            i_e -= 1
            if i_e <= i_s:
                raise ValueError('i_e not larger than i_s')
            self.coef = -dx / ((j_e - j_s) * (k_e - k_s))

        elif self.direction == 'y':
            j_e -= 1
            if j_e <= j_s:
                raise ValueError('j_e not larger than j_s')
            self.coef = -dy / ((i_e - i_s) * (k_e - k_s))

        elif self.direction == 'z':
            k_e -= 1
            if k_e <= k_s:
                raise ValueError('k_e not larger than k_s')
            self.coef = -dz / ((i_e - i_s) * (j_e - j_s))

        else:
            raise ValueError(f'Cannot match direction {self.direction}.')

        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]
        self.coef = self.coef * self.polar


class CurrentMonitor:
    """
    Current monitor at a surface orthogonal to (x,y,z) direction.

    Args:
        polarization (str): 'xn', 'xp', 'yn', 'yp' or 'zn', 'zp'.

    Attributes:
        direction (str): 'x', 'y' or 'z'
        polar (float): +1 if polarization in ['xp', 'yp', 'zp'], else -1.
        coef (float): average factor
        indices (list): lists of [start, stop)
    """
    def __init__(self, polarization) -> None:
        self.direction = polarization.lower()[0]
        self.polar = -1. if polarization.lower()[-1] == 'n' else 1.
        self.coef = self.polar
        self.indices = []

    def grid_registry(self, grid_helper, x, y, z):
        """
        Check indices.

        Args:
            grid_helper (GridHelper):
            x (tuple): nodex index range ([start, stop)) in x direction
            y (tuple): nodex index range ([start, stop)) in y direction
            z (tuple): nodex index range ([start, stop)) in z direction
        """
        grid_helper.current_monitors.append(self)
        nx, ny, nz = grid_helper.cell_numbers
        i_s, i_e = x
        j_s, j_e = y
        k_s, k_e = z

        # check node locations
        if i_s <= 0:
            raise ValueError('i_s is not positive')
        if j_s <= 0:
            raise ValueError('j_s is not positive')
        if k_s <= 0:
            raise ValueError('k_s is not positive')
        if i_e > nx:
            raise ValueError('i_e is larger than nx')
        if j_e > ny:
            raise ValueError('j_e is larger than ny')
        if k_e > nz:
            raise ValueError('k_e is larger than nz')

        if self.direction == 'x' and i_e - i_s != 1:
            raise ValueError(f'Ending index is not 1 larger than starting index in {self.direction}')

        if self.direction == 'y' and j_e - j_s != 1:
            raise ValueError(f'Ending index is not 1 larger than starting index in {self.direction}')

        if self.direction == 'z' and k_e - k_s != 1:
            raise ValueError(f'Ending index is not 1 larger than starting index in {self.direction}')

        self.indices = [[i_s, i_e], [j_s, j_e], [k_s, k_e]]

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
"""grid"""
from __future__ import division
import numpy


class Grids(object):
    def __init__(self, coords, weights, opts):
        self._coords = coords
        self.original_coords = coords.copy() # grids for numerical integration
        self.weights = weights
        self.opts = opts

        self.a = opts['CubeLength']
        self.na = opts['CubePoint']
        self.sym = opts['Symmetric'] if opts['Symmetric'] is not None else None

        self._apply_symm()
        self._generate_offset()
        self._extend_coords()
        print('Grids::__init__() done.')
        # END OF __init__()

    def _apply_symm(self):
        if self.sym is None or self.sym == 'none':
            self.coords = self._coords.copy() 
        elif self.sym == 'xz' or self.sym == 'zx':
            self.coords = _mesh_xz(self._coords)
        elif self.sym == 'xz+':
            self.coords = _mesh_xz_half(self._coords, 1)
        elif self.sym == 'xz-':
            self.coords = _mesh_xz_half(self._coords, -1)
        else:
            warnings.warn('Unknown Symmetric option ``%s". Fallback to ``None"' % (self.sym), RuntimeWarning)
            self.coords = self._coords.copy() 
        
        # END OF _apply_symm()

    def _generate_offset(self):
        na3 = self.na * self.na * self.na
        offset = numpy.empty([na3, 3])
        dd = 1. / (self.na - 1)
        p = 0
        for i in range(self.na):
            for j in range(self.na):
                for k in range(self.na):
                    offset[p][0] = -0.5 + dd * i
                    offset[p][1] = -0.5 + dd * j
                    offset[p][2] = -0.5 + dd * k
                    p += 1
        self.offset = offset * self.a
        # END of _generate_offset()


    def _extend_coords(self):
        na = self.na
        na3 = na * na * na
        extended_coords = numpy.empty([len(self.coords)*na3, 3])
        p = 0
        for i, c in enumerate(self.coords):
            extended_coords[p:p+na3] = c + self.offset
            p += na3
        self.extended_coords = extended_coords
        print('Grids::_extend_coords() done.')
        # END of _extend_coords()



def _mesh_xz(mesh):
    print('Transform grids onto half xz')
    coords = numpy.zeros([len(mesh), 3])
    coords[:, 0] = numpy.sqrt(
            numpy.sum(
                mesh[:, :2] * mesh[:, :2],
                axis=1
                ))
    coords[:, 2] = mesh[:, 2]
    return coords


def _mesh_xz_half(mesh, sgn=1):
    print('Transform grids onto half xz-plane')
    coords = numpy.zeros([len(mesh), 3])
    coords[:, 0] = numpy.sqrt(
            numpy.sum(
                mesh[:, :2] * mesh[:, :2],
                axis=1
                ))
    coords[:, 2] = numpy.abs(mesh[:, 2]) * sgn
    return coords


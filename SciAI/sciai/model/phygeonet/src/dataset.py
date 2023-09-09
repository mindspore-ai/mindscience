# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Dataset handling"""
import numpy as np


class AbstractDataset:
    """Base class dataset"""
    def __init__(self, mesh_list):
        self.mesh_list = mesh_list

    def __len__(self):
        return len(self.mesh_list)

    @staticmethod
    def get_single(mesh):
        cord = np.zeros([2, mesh.x.shape[0], mesh.x.shape[1]])
        cord[0, :, :] = mesh.x
        cord[1, :, :] = mesh.x
        invariant_input = np.zeros([2, mesh.j_ho.shape[0], mesh.j_ho.shape[1]])
        invariant_input[0, :, :] = mesh.j_ho
        invariant_input[1, :, :] = mesh.jinv_ho
        return invariant_input, cord, mesh.xi, mesh.eta, mesh.j_ho, \
               mesh.jinv_ho, mesh.dxdxi_ho, mesh.dydxi_ho, \
               mesh.dxdeta_ho, mesh.dydeta_ho


class VaryGeoDataset(AbstractDataset):
    """docstring for hcubeMeshDataset"""

    def __getitem__(self, idx):
        return self.get_single(self.mesh_list[idx])


class FixGeoDataset:
    """docstring for FixGeoDataset"""

    def __init__(self, para_list, mesh, of_solution_list):
        self.para_list = para_list
        self.mesh = mesh
        self.of_solution_list = of_solution_list

    def __len__(self):
        return len(self.para_list)

    def __getitem__(self, idx):
        cord = np.zeros([2, self.mesh.x.shape[0], self.mesh.x.shape[1]])
        cord[0, :, :] = self.mesh.x
        cord[1, :, :] = self.mesh.y
        para_start = np.ones(self.mesh.x.shape[0]) * self.para_list[idx]
        para_end = np.zeros(self.mesh.x.shape[0])
        para = np.linspace(para_start, para_end, self.mesh.x.shape[1]).T
        return (para, cord, self.mesh.xi, self.mesh.eta, self.mesh.j_ho,
                self.mesh.jinv_ho, self.mesh.dxdxi_ho, self.mesh.dydxi_ho,
                self.mesh.dxdeta_ho, self.mesh.dydeta_ho, self.of_solution_list[idx])


class VarygeodatasetPairedsolution(AbstractDataset):
    """docstring for VarygeodatasetPairedsolution"""

    def __init__(self, mesh_list, solution_list):
        super(VarygeodatasetPairedsolution, self).__init__(mesh_list)
        self.solution_list = solution_list

    def __getitem__(self, idx):
        return (*self.get_single(self.mesh_list[idx]),
                self.solution_list[idx][:, :, 0],
                self.solution_list[idx][:, :, 1],
                self.solution_list[idx][:, :, 2])

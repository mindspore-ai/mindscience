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
"""Postprocess"""
import os

import numpy as np
from scipy.interpolate import interp1d
from mindflow import load_yaml_config


def get_grid_interp(shroud_file='./data/shroud.dat',
                    hub_file='./data/hub.dat',
                    grid_num_s=64,
                    grid_num_z=128,
                    ):
    """get_grid_interp"""
    hub = np.loadtxt(hub_file)/1000
    shroud = np.loadtxt(shroud_file)/1000
    line_xx = np.linspace(-0.059, 0.119, grid_num_z)
    y_hub = interp1d(hub[:, 0], hub[:, 1], kind='linear')(line_xx)
    y_shroud = interp1d(shroud[:, 0], shroud[:, 1], kind='linear')(line_xx)
    tmp = [np.linspace(y_hub[i], y_shroud[i], grid_num_s) for i in range(grid_num_z)]
    grid_yy = np.concatenate(tmp, axis=0)
    grid_yy = grid_yy.reshape([1, grid_num_z, grid_num_s]).T
    grid_xx = np.tile(line_xx, [grid_num_s, 1])
    grid_xx = grid_xx.reshape([grid_num_s, grid_num_z, 1])
    return np.concatenate((grid_xx, grid_yy), axis=2)


class CFDPost2D:
    """the CFD post process of 2D meridian surface"""
    def __init__(self, data=None, grid=None):
        self.grid = grid
        self.data_2d = data
        self.num = self.data_2d.shape[0]
        self.n_1d = self.data_2d.shape[1]
        self.n_2d = self.data_2d.shape[2]
        self._basic_const_dict = \
        load_yaml_config(os.path.join('./configs', 'optimization.yaml'))['basic_const']
        self._init_field_save_dict()
        self._init_performance_save_dict()
        self._init_field_calculate_dict()
        self._init_performance_calculate_dict()

    def get_field(self, quanlity):
        """get field"""
        if self._field_save_dict[quanlity] is None:
            if quanlity in self.inputs_dict:
                rst = self.data_2d[..., self.inputs_dict[quanlity]]
            else:
                func = self._field_calculate_dict[quanlity]
                para = [self.get_field(x) for x in self._field_para_dict[quanlity]]
                rst = func(*para)
            self._field_save_dict[quanlity] = rst
        else:
            rst = self._field_save_dict[quanlity]
        return rst

    def get_performance(self, performance,
                        z_axis_1=0, z_axis_2=None):
        """get_performance"""
        if z_axis_2 is None:
            z_axis_2 = self.n_2d - 1
        rst = self.calculate_performance_averaged(performance, z_axis_1=z_axis_1, z_axis_2=z_axis_2)
        return rst

    def get_mass_weight_field(self, quanlity):
        """get_mass_weight_field"""
        if quanlity in ('Static Pressure', 'Density Flow'):
            rst = self.get_field(quanlity)
        elif quanlity in ('Gird Node R',):
            rst = self.get_field(quanlity)
            rst = np.power(rst[:, -1, None, :], 2) - np.power(rst[:, 0, None, :], 2)
        else:
            density_flow = self.get_field('Density') * self.get_field('Vz')
            rst = density_flow * self.get_field(quanlity) / np.mean(density_flow, axis=1, keepdims=True)
        return np.mean(rst, axis=1, keepdims=False)

    def calculate_performance_averaged(self, performance, z_axis_1=None, z_axis_2=None):
        """calculate_performance_averaged"""
        para = self._performance_para_dict[performance]
        para_values = []
        zlist = [z_axis_1, z_axis_2]
        func = self._performance_calculate_dict[performance]
        for i in (0, 1):
            for name in para[i]:
                if name in self._field_save_dict:
                    values = self.get_mass_weight_field(name)
                elif name in self._performance_save_dict:
                    values = self.get_performance(name, z_axis_1=z_axis_1, z_axis_2=z_axis_1)
                para_values.append(values[:, zlist[i]])
        return func(*para_values)

    def _init_field_save_dict(self):
        """get_field_save_dict"""
        self.inputs_dict = {
            'Static Pressure': 0,
            'Static Temperature': 1,
            'Density': 2,
            'Vx': 3,
            'Vy': 4,
            'Vz': 5,
            'Relative Total Temperature': 6,
            'Absolute Total Temperature': 7,
        }
        self._quanlity_list = [
            'Static Pressure',
            'Relative Total Pressure',
            'Absolute Total Pressure',
            'Static Temperature',
            'Relative Total Temperature',
            'Absolute Total Temperature',
            'Vx', 'Vy', 'Vz', '|V|',
            'Density', 'Density Flow',
        ]
        self._field_save_dict = {}
        for quanlity in self._quanlity_list:
            self._field_save_dict.update({quanlity: None})
        self._field_save_dict.update({'Gird Node Z': np.tile(self.grid[None, :, :, 0], [self.num, 1, 1])})
        self._field_save_dict.update({'Gird Node R': np.tile(self.grid[None, :, :, 1], [self.num, 1, 1])})
        self._field_save_dict.update(
            {'R/S Interface': np.where(self.grid[None, :, :, 0] < self._basic_const_dict['rs_interface'], 0, 1)}
        )

    def _init_field_calculate_dict(self):
        """get_field_calculate_dict"""
        self._field_calculate_dict = {}
        self._field_para_dict = {}
        for quanlity in self._quanlity_list:
            self._field_calculate_dict.update({quanlity: None})
            self._field_para_dict.update({quanlity: None})
        self._field_calculate_dict['|V|'] = lambda x1, x2, x3: np.power(x1 * x1 + x2 * x2 + x3 * x3, 0.5)
        self._field_para_dict['|V|'] = ('Vx', 'Vy', 'Vz')
        self._field_calculate_dict['Absolute Total Temperature'] = \
        lambda x1, x2: x1 + x2 * x2 / 2 / self._basic_const_dict['Cp']
        self._field_para_dict['Absolute Total Temperature'] = ('Static Temperature', '|V|')
        self._field_calculate_dict['Absolute Total Pressure'] =\
        lambda x1, x2, x3: x1 * np.power(x2 / x3, self._basic_const_dict['kappa'] / (self._basic_const_dict['kappa'] - 1))
        self._field_para_dict['Absolute Total Pressure'] = \
        ('Static Pressure', 'Absolute Total Temperature', 'Static Temperature')
        self._field_calculate_dict['Relative Total Pressure'] =\
        lambda x1, x2, x3: x1 * np.power(x2 / x3, self._basic_const_dict['kappa'] / (self._basic_const_dict['kappa'] - 1))
        self._field_para_dict['Relative Total Pressure'] = \
        ('Static Pressure', 'Relative Total Temperature', 'Static Temperature')
        self._field_calculate_dict['Density Flow'] = lambda x1, x2: x1 * x2
        self._field_para_dict['Density Flow'] = ('Density', 'Vz')

    def _init_performance_save_dict(self):
        """get_performance_save_dict"""
        self._performance_list = [
            'Static_pressure_ratio',
            'Static_temperature_ratio',
            'Total_total_efficiency',
            'Mass_flow',
        ]
        self._performance_save_dict = {}
        for performance in self._performance_list:
            self._performance_save_dict.update({performance: None})

    def _init_performance_calculate_dict(self):
        """init_performance_calculate_dict"""
        self._performance_calculate_dict = {}
        self._performance_para_dict = {}
        for performance in self._performance_list:
            self._performance_calculate_dict.update({performance: None})
            self._performance_para_dict.update({performance: None})
        self._performance_calculate_dict['Static_pressure_ratio'] = lambda x1, x2: x1 / x2
        self._performance_para_dict['Static_pressure_ratio'] = [('Static Pressure',) for _ in range(2)]
        self._performance_calculate_dict['Static_temperature_ratio'] = lambda x1, x2: x1 / x2
        self._performance_para_dict['Static_temperature_ratio'] = [('Static Temperature',) for _ in range(2)]
        self._performance_calculate_dict['Total_total_efficiency'] = self._get_efficiency
        self._performance_para_dict['Total_total_efficiency'] = \
            [('Absolute Total Temperature', 'Absolute Total Pressure') for _ in range(2)]
        self._performance_calculate_dict['Mass_flow'] = lambda x1, x2, x3, x4: (x1 * x2 + x3 * x4) * np.pi / 2
        self._performance_para_dict['Mass_flow'] = [('Density Flow', 'Gird Node R') for _ in range(2)]

    def _get_bar_value(self, values, z_position, bar_length=2):
        """get bar value"""
        if bar_length > 0:
            tmp = values[..., max(z_position - bar_length, 0): min(z_position + bar_length, self.n_2d)]
            return np.mean(tmp, axis=-1)
        return values[..., z_position]

    def _get_efficiency(self, t1, p1, t2, p2): # turbins only
        tmp = np.power(p2 / p1, (self._basic_const_dict['kappa'] - 1) / self._basic_const_dict['kappa'])
        rst = (1 - (t2 / t1)) / (1 - tmp)
        return rst

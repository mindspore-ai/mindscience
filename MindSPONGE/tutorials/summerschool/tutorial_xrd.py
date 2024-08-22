# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
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
"""MindSPONGE tutorial XRD"""
import sys
import numpy as np
import mindspore as ms
sys.path.insert(0, '../../src') # pylint: disable=C0413
from sponge.colvar import XRD3D
from sponge import Molecule, WithEnergyCell, ForceField, Sponge, UpdaterMD
from sponge.function import VelocityGenerator
from sponge.sampling import Metadynamics
from sponge.callback import WriteH5MD, RunInfo
from sponge.control import Langevin

ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

system = Molecule(template='water.spce.yaml')
system.set_pbc_box([1.285/4, 1.285/4, 1.286/4])
system.repeat_box([4, 4, 4])
coord = np.loadtxt('water_spce_coordinate.txt', dtype=np.float32) / 10
system.set_coordianate(coord)

ff = ForceField(system, 'SPCE', 0.6)

xrd3d = XRD3D(theta=11.95,
              lamb=1.54,
              index=[3*i for i in range(64)],
              s=0.001,
              qi=[1.0]*64,
              pbc_box=system.pbc_box)

metad = Metadynamics(colvar=xrd3d,
                     update_pace=50,
                     height=10,
                     sigma=0.2,
                     grid_min=-1,
                     grid_max=8,
                     grid_bin=90,
                     temperature=230,
                     bias_factor=200)

velocity = VelocityGenerator(temperature=230)(system.coordinate.shape, system.atom_mass)
optim2 = UpdaterMD(system, time_step=1e-3, velocity=velocity, temperature=230,
                   integrator='leap_frog', thermostat=Langevin(system, 230, time_constant=0.1))
run_info = RunInfo(1000)
write_h5md = WriteH5MD(system, 'test.h5md', 50, write_metrics=True)

network = WithEnergyCell(system, ff, metad)
md = Sponge(network, optimizer=optim2, metrics={'xrd3d': xrd3d})
md.run(20000, callbacks=[run_info, write_h5md])

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
"""solve riemann 2d flow"""
from mindspore import context
from mindflow import load_yaml_config, vis_2d
from mindflow import cfd
from mindflow.cfd.runtime import RunTime
from mindflow.cfd.simulator import Simulator

from src.ic import riemann2d_ic

context.set_context(device_target="GPU", device_id=0)

config = load_yaml_config('numeric.yaml')

simulator = Simulator(config)
runtime = RunTime(config['runtime'], simulator.mesh_info, simulator.material)

mesh_x, mesh_y, _ = simulator.mesh_info.mesh_xyz()
pri_var = riemann2d_ic(mesh_x, mesh_y)
con_var = cfd.cal_con_var(pri_var, simulator.material)

while runtime.time_loop(pri_var):
    pri_var = cfd.cal_pri_var(con_var, simulator.material)
    runtime.compute_timestep(pri_var)
    con_var = simulator.integration_step(con_var, runtime.timestep)
    runtime.advance()

pri_var = cfd.cal_pri_var(con_var, simulator.material)
vis_2d(pri_var, 'riemann2d.jpg')

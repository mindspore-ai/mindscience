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
# ==============================================================================
"""solve lax tube flow"""
import argparse

from mindspore import context
from src.ic import lax_ic_1d

from mindflow import cfd, load_yaml_config, vis_1d
from mindflow.cfd.runtime import RunTime
from mindflow.cfd.simulator import Simulator

parser = argparse.ArgumentParser(description="Sod compute")
parser.add_argument(
    "--mode",
    type=str,
    default="GRAPH",
    choices=["GRAPH", "PYNATIVE"],
    help="Running in GRAPH_MODE OR PYNATIVE_MODE",
)
parser.add_argument(
    "--save_graphs",
    type=bool,
    default=False,
    choices=[True, False],
    help="Whether to save intermediate compilation graphs",
)
parser.add_argument("--save_graphs_path", type=str, default="./graphs")
parser.add_argument(
    "--device_target",
    type=str,
    default="GPU",
    choices=["GPU", "Ascend"],
    help="The target device to run, support 'Ascend', 'GPU'",
)
parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
parser.add_argument("--config_file_path", type=str, default="./numeric.yaml")
parser.add_argument("--reconstructor", type=str, choices=["WENO3", "WENO5", "WENO7"], default="WENO5")
parser.add_argument("--riemann_computer", type=str, choices=["HLLC", "Roe", "Rusanov"], default="Roe")

args = parser.parse_args()

context.set_context(
    mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
    save_graphs=args.save_graphs,
    save_graphs_path=args.save_graphs_path,
    device_target=args.device_target,
    device_id=args.device_id,
)
print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")

config = load_yaml_config(args.config_file_path)
config["space_solver"]["convective_flux"]["reconstructor"] = args.reconstructor
config["space_solver"]["convective_flux"]["riemann_computer"] = args.riemann_computer

simulator = Simulator(config)
runtime = RunTime(config["runtime"], simulator.mesh_info, simulator.material)

mesh_x, _, _ = simulator.mesh_info.mesh_xyz()
pri_var = lax_ic_1d(mesh_x)
con_var = cfd.cal_con_var(pri_var, simulator.material)

while runtime.time_loop(pri_var):
    pri_var = cfd.cal_pri_var(con_var, simulator.material)
    runtime.compute_timestep(pri_var)
    con_var = simulator.integration_step(con_var, runtime.timestep)
    runtime.advance()

pri_var = cfd.cal_pri_var(con_var, simulator.material)
vis_1d(pri_var, "lax.jpg")

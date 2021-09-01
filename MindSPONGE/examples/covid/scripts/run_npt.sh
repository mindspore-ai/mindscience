#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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

mkdir -p pres
cd pres
cat > pres.in << EOF
S3 press
  mode = npt
  step_limit = 60000
  dt = 2e-3
  constrain_mode = simple_constrain
  write_information_interval = 2500
  thermostat = langevin_liu
  barostat = berendsen
EOF
python ../../src/run_npt.py --i ./pres.in --amber_parm ../../data/ace2.parm7 --c ../heat/heat.rst7  --r pres.rst7
cd ..

#mkdir -p product
#cd product
#cat > md.in << EOF
#S4 product
# mode = npt
# step_limit = 750000
# dt = 4e-3
# constrain_mode = simple_constrain
# write_information_interval = 2500
# write_restart_file_interval = 250000
# thermostat = langevin_liu
# langevin_liu_velocity_max = 20
# langevin_liu_gamma = 10.0
# barostat = berendsen
#EOF
#python ../../src/run_npt.py --i ./md.in --amber_parm ../../data/ace2.parm7 --c ../pres/pres.rst7  --r product.rst7
#cd ..

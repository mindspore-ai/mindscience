#!/bin/bash
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
mkdir sample
line=13
total=0
for x_step in $(seq 0.02 0.01 0.23) 
do
   for y_step in $(seq 0.02 0.01 0.16)
   do
       let total=total+1
       echo "file: $x_step, $y_step, $total"
       mkdir ./sample/sample_cylinder_Ascan_2D_${x_step}_${y_step}
       cp cylinder_Ascan_2D.in ./sample/sample_cylinder_Ascan_2D_${x_step}_${y_step}
       cd ./sample/sample_cylinder_Ascan_2D_${x_step}_${y_step}
       design="#cylinder: ${x_step} ${y_step} 0 ${x_step} ${y_step} 0.002 0.010 pec"
       sed -i "${line}c ${design}" cylinder_Ascan_2D.in
       python -m gprMax cylinder_Ascan_2D.in>train.log$total 2>&1 &  
       cd ../../
   done 
done 

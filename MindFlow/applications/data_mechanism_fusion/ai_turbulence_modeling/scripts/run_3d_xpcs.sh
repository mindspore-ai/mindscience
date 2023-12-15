#!/bin/bash
# ============================================================================
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_cluster.sh DATA_PATH RANK_TABLE_FILE RANK_SIZE RANK_START ./dataset/experiment/changeCondition_20230605163432"
echo "For example: bash run_3d_16pcs.sh /path/rank_table.json 16 0 ./dataset/experiment/changeCondition_20230605163432"
echo "It is better to use the absolute path."
echo "The time interval between multiple machines to execute the script should not exceed 120s"
echo "=============================================================================================================="

execute_path=$(pwd)
echo ${execute_path}
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
echo ${self_path}

export RANK_TABLE_FILE=$1
export RANK_SIZE=$2
RANK_START=$3

DEVICE_START=0
for((i=1;i<=7;i++));
do
  export RANK_ID=$[i+RANK_START]
  export DEVICE_ID=$[i+DEVICE_START]
  echo "start training for device `expr $i + $DEVICE_START`"
  echo "start training for rank $RANK_ID"
  env > env$i.log
  nohup python train_3d.py --config_file_path="./configs/TurbAI_3D_ResMLP.yaml" > train$i.log 2>&1 &
done

export DEVICE_ID=$DEVICE_START
export RANK_ID=$RANK_START
echo "start training for device $DEVICE_START"
echo "start training for rank $RANK_START"
env > env$i.log
python train_3d.py --config_file_path="./configs/TurbAI_3D_ResMLP.yaml"

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
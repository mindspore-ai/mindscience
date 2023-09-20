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
echo "bash run.sh RANK_SIZE DEVICE_START"
echo "For example: bash run.sh 8 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

RANK_SIZE=$1
DEVICE_START=$2

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/scripts/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_4pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/scripts/rank_table_4pcs.json
    export RANK_SIZE=4
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/scripts/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_1pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/scripts/rank_table_1pcs.json
    export RANK_SIZE=1
}

test_dist_${RANK_SIZE}pcs


for((i=1;i<${RANK_SIZE};i++))
do
    export DEVICE_ID=`expr $i + $DEVICE_START`
    export RANK_ID=$i
    echo "start training for device `expr $i + $DEVICE_START`"
    echo "start training for rank $RANK_ID"
    env > ${DATAPATH}/env$i.log
    nohup python train_3d.py --config_file_path="./configs/TurbAI_3D_ResMLP.yaml" > ${DATAPATH}/train$i.log 2>&1 &
done


export DEVICE_ID=$DEVICE_START
export RANK_ID=0
echo "start training for device $DEVICE_START"
echo "start training for rank 0"
env > ${DATAPATH}/env$i.log
python train_3d.py --config_file_path="./configs/TurbAI_3D_ResMLP.yaml"

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
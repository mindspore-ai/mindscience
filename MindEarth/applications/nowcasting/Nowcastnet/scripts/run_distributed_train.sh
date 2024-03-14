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
# ============================================================================
if [ $# != 3 ]
then
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DEVICE_NUM] [DEVICE_START_ID]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=$2
RANK_TABLE_FILE=$(realpath $1)
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"
export DEVICE_START_ID=$3


for((i=$DEVICE_START_ID;i<$[$DEVICE_NUM+$DEVICE_START_ID];i++))
do
    rm -rf device$i
    mkdir device$i
    cp ../main.py ./device$i
    cp -r ../src ./device$i
    cp -r ../configs ./device$i
    cp -r ../../../../mindearth ./device$i
    cd ./device$i
    mkdir logs
    export DEVICE_ID=$i
    export RANK_ID=$[$i-$DEVICE_START_ID]
    export GLOG_v=3
    echo "start training for device $i"
    env > env$i.log
    nohup python -u main.py --device_id $i --run_mode train >parallel_subset.log 2>&1 &
    cd ../
done
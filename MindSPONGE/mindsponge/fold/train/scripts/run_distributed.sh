#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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

if [ $# != 1 ]
then
    echo "Usage: sh run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE]"
exit 1
fi

RANK_SIZE=$1
RANK_TABLE_FILE=$2
export RANK_TABLE_FILE=$RANK_TABLE_FILE

test_dist_8pcs()
{
    export RANK_SIZE=8
}

test_dist_4pcs()
{
    export RANK_SIZE=4
}

test_dist_2pcs()
{
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=0;i<${RANK_SIZE};i++))
do
    # rm -rf device$i
    # mkdir device$i
    # cp ./resnet50_distributed_training.py ./resnet.py ./device$i
    # cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp -r ../data ./train_parallel$i
    cp -r ../config ./train_parallel$i
    cp -r ../module ./train_parallel$i
    cp -r ../common ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env$i.log
    python train.py --run_distribute=True > log.log 2>&1 &
    cd ../
done
# rm -rf device0
# mkdir device0
# cp ./resnet50_distributed_training.py ./resnet.py ./device0
# cd ./device0
# export DEVICE_ID=0
# export RANK_ID=0
# echo "start training for device 0"
# env > env0.log
# pytest -s -v ./resnet50_distributed_training.py > train.log0 2>&1
# if [ $? -eq 0 ];then
#     echo "training success"
# else
#     echo "training failed"
#     exit 2
# fi
# cd ../
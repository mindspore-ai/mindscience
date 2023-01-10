#!/bin/bash
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
# ============================================================================

echo "====================================================================================================================================================="
echo "Please run the script as: "
echo "Usage: bash run_distribute_train.sh [DEVICE_TARGET] [DEVICE_NUM] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [SAVE_DIR] [RANK_TABLE_FILE] [START_DEVICE_ID]"
echo "for example: bash run_distribute_train.sh Ascend 2 ../exampledata/finetune bbbp classification ../convert_grover_base.ckpt ../ckpt hccl_2p.json"
echo "DATASET must be in ['bbbp', 'clintox', 'bace', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipo', 'qm7', 'qm8']."
echo "DATASET_TYPE must be in ['classification', 'regression']"
echo "====================================================================================================================================================="

if [ $# -ne 9 ]
then
    echo "Usage: bash run_distribute_train.sh [DEVICE_TARGET] [DEVICE_NUM] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [SAVE_DIR] [RANK_TABLE_FILE] [START_DEVICE_ID]"
exit 1
fi

DEVICE_TARGET=$1
export RANK_SIZE=$2

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_DIR=$(get_real_path $3)
DATASET=$4
DATASET_TYPE=$5
CKPT=$(get_real_path $6)
SAVE_DIR=$(get_real_path $7)
RANK_TABLE=$(get_real_path $8)
export RANK_START_ID=$9
export RANK_TABLE_FILE=$RANK_TABLE

echo "device: "$DEVICE_TARGET
echo "device num:" $RANK_SIZE
echo "data dir: "$DATA_DIR
echo "dataset name:"$DATASET
echo "dataset type: "$DATASET_TYPE
echo "pretrained model: "$CKPT
echo "save dir: "$SAVE_DIR
echo "rank table: "$RANK_TABLE
echo "start device id: "$RANK_START_ID

for((i=0;i<RANK_SIZE;i++))
do
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp ../*.yaml ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    export RANK_ID=$i
    export DEVICE_ID=$((i + RANK_START_ID))
    echo "start training for device $DEVICE_ID"
    env > env.log
    python train.py \
                --device_target=$DEVICE_TARGET \
                --run_distribute=True \
                --data_path_finetune=$DATA_DIR \
                --data_file_finetune=$DATASET \
                --dataset_type=$DATASET_TYPE \
                --resume_grover=$CKPT \
                --save_dir=$SAVE_DIR \
                --mixed=True > log.txt 2>&1 &
    cd ..
done

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
echo "Usage: bash run_standalone_train.sh [DEVICE_TARGET] [DEVICE_ID] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [SAVE_DIR]"
echo "for example: bash run_standalone_train.sh Ascend 1 ../exampledata/finetune bbbp classification ../convert_grover_base.ckpt ../ckpt"
echo "DATASET must be in ['bbbp', 'clintox', 'bace', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipo', 'qm7', 'qm8']."
echo "DATASET_TYPE must be in ['classification', 'regression']"
echo "====================================================================================================================================================="

if [ $# -ne 7 ]
then
    echo "Usage: bash run_standalone_train.sh [DEVICE_TARGET] [DEVICE_ID] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [SAVE_DIR]"
exit 1
fi

DEVICE_TARGET=$1
DEVICEID=$2

export DEVICE_NUM=1
export DEVICE_ID=$DEVICEID
export RANK_ID=0
export RANK_SIZE=1

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

echo "device: "$DEVICE_TARGET
echo "device id:" $DEVICEID
echo "data dir: "$DATA_DIR
echo "dataset name:"$DATASET
echo "dataset type: "$DATASET_TYPE
echo "pretrained model: "$CKPT
echo "save dir: "$SAVE_DIR

rm -rf ./train
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp ../*.yaml ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train.py \
                --device_target=$DEVICE_TARGET \
                --run_distribute=False \
                --data_path_finetune=$DATA_DIR \
                --data_file_finetune=$DATASET \
                --dataset_type=$DATASET_TYPE \
                --resume_grover=$CKPT \
                --save_dir=$SAVE_DIR \
                --mixed=True > log.txt 2>&1 &
cd ..

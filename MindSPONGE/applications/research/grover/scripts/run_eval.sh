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
echo "Usage: bash run_eval.sh [DEVICE_TARGET] [DEVICE_ID] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [EVAL_DIR] [METRICS] [SCALER_DIR]"
echo "for example: bash run_eval.sh Ascend 0 ../exampledata/finetune bbbp classification ../ckpt/bbbp/grover_100.ckpt ../outputs auc ../ckpt"
echo "DATASET must be in ['bbbp', 'clintox', 'bace', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipo', 'qm7', 'qm8']."
echo "DATASET_TYPE must be in ['classification', 'regression']"
echo "====================================================================================================================================================="

if [ $# -ne 9 ]
then
    echo "Usage: bash run_eval.sh [DEVICE_TARGET] [DEVICE_ID] [DATA_DIR] [DATASET] [DETASET_TYPE] [PRETRAINED] [EVAL_DIR] [METRICS] [SCALER_DIR]"
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
EVAL_DIR=$(get_real_path $7)
METRICS=$8
SCALER_DIR=$(get_real_path $9)

echo "device: "$DEVICE_TARGET
echo "device id:" $DEVICEID
echo "data dir: "$DATA_DIR
echo "dataset name:"$DATASET
echo "dataset type: "$DATASET_TYPE
echo "pretrained model: "$CKPT
echo "eval dir: "$EVAL_DIR
echo "metrics: "$METRICS
echo "scaler dir: "$SCALER_DIR

rm -rf ./eval
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cd ./eval || exit
echo "start training for device $DEVICE_ID"
env > env.log
python eval.py --device_target=$DEVICE_TARGET \
               --data_path_eval=$DATA_DIR \
               --data_file_eval=$DATASET \
               --dataset_type=$DATASET_TYPE \
               --pretrained=$CKPT \
               --eval_dir=$EVAL_DIR \
               --metrics=$METRICS \
               --save_dir=$SCALER_DIR > log.txt 2>&1 &
cd ..

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
set -e
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train.sh device_id"
echo "For example: bash run_standalone_train.sh 4"
echo "This example is expected to run on the Ascend environment."
echo "=============================================================================================================="

ID_N=$1
DEVICE_TARGET=${2:-"Ascend"}
CONFIG_FILE=${3:-"./configs/FourCastNet.yaml"}
CONFIG_FILE_PATH=$(realpath $CONFIG_FILE)
echo "CONFIG_FILE_PATH=${CONFIG_FILE_PATH}"

rm -rf single_train_device$ID_N
mkdir single_train_device$ID_N
cp ./main.py ./single_train_device$ID_N
cp ${CONFIG_FILE_PATH} ./single_train_device$ID_N
cp -r ./src ./single_train_device$ID_N
cd ./single_train_device$ID_N
export DEVICE_ID=$ID_N
export GLOG_v=3
echo "start training on device $ID_N"
env > env$ID_N.log
nohup python -u main.py\
    --device_id $ID_N\
    --run_mode "train"\
    --config_file_path ${CONFIG_FILE_PATH}\
    --device_target ${DEVICE_TARGET}\
    >train.log 2>&1 &
cd ../
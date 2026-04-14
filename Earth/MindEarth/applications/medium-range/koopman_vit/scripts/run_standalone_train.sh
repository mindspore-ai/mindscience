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

# 获取命令行参数
ID_N=$1
DEVICE_TARGET=${2:-"Ascend"}
CONFIG_FILE=${3:-"./configs/vit_kno_1.4.yaml"}
CONFIG_FILE_PATH=$(realpath $CONFIG_FILE)

# 打印配置文件路径
echo "CONFIG_FILE_PATH=${CONFIG_FILE_PATH}"

# 创建运行目录
rm -rf single_device$ID_N
mkdir single_device$ID_N

# 复制文件到运行目录
cp ./main.py ./single_device$ID_N
cp ${CONFIG_FILE_PATH} ./single_device$ID_N
cp -r ./src ./single_device$ID_N

# 进入运行目录
cd ./single_device$ID_N

# 设置环境变量
export DEVICE_ID=$ID_N
export GLOG_v=3

# 打印开始训练信息
echo "start training for device $ID_N"

# 导出环境变量到日志文件
env > env$ID_N.log

# 运行训练脚本
nohup python -u main.py \
    --device_id $ID_N \
    --run_mode "train" \
    --config_file_path ${CONFIG_FILE_PATH} \
    --device_target ${DEVICE_TARGET} \
    >single_subset.log 2>&1 &

# 返回上一级目录
cd ../
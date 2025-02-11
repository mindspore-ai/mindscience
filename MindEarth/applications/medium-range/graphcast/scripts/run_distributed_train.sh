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
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [CONFIG_FILE] [DEVICE_NUM]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

RANK_TABLE_FILE=$(realpath  $1)
export MINDSPORE_HCCL_CONFIG_PATH=${RANK_TABLE_FILE}
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"
CONFIG_FILE=$(realpath $2)
echo "CONFIG_FILE=${CONFIG_FILE}"
DEVICE_NUM=$(realpath $3)
echo "CONFIG_FILE=${DEVICE_NUM}"

i=1
rm -rf msrun
mkdir msrun
cp ./main.py ./msrun
cp ${CONFIG_FILE} ./msrun
cp -r ./src ./msrun
cd ./msrun
export DEVICE_ID=$i
export RANK_ID=$[$i-$i]
export GLOG_v=3
echo "start training for msrun"
env > env.log
nohup msrun --worker_num=${DEVICE_NUM} --local_worker_num=${DEVICE_NUM} --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 main.py --device_id $i --config_file_path ${CONFIG_FILE} >train${i}.log 2>&1 &
cd ../

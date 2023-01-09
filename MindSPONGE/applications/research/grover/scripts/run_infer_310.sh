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
echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [DATASET_TYPE] [METRICS] [BATCHSIZE]"
echo "for example: bash run_infer_310.sh ../GROVERbbbp.mindir bbbp ../exampledata/finetune y 2 classification auc 32"
echo "DATASET_NAME must be in ['bbbp', 'clintox', 'bace', 'tox21', 'toxcast', 'freesolv', 'esol', 'lipo', 'qm7', 'qm8']."
echo "NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'."
echo "DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
echo "DATASET_TYPE must be in ['classification', 'regression']"
echo "METRICS can be in ['auc', 'rmse', 'mae']"
echo "====================================================================================================================================================="

if [[ $# -lt 7 || $# -gt 8 ]]
then
  echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID] [DATASET_TYPE] [METRICS] [BATCHSIZE]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
dataset_name=$2
dataset_path=$(get_real_path $3)

if [ "$4" == "y" ] || [ "$4" == "n" ];then
    need_preprocess=$4
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi
data_type=$6
metrics=$7
batch_size=$8

echo "mindir name: "$model
echo "dataset name: "$dataset_name
echo "dataset path: "$dataset_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id
echo "dataset type: "$data_type
echo "metrics: "$metrics
echo "batch_size: "$batch_size

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
else
    export ASCEND_HOME=/usr/local/Ascend/latest
fi
export PATH=$ASCEND_HOME/compiler/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=$ASCEND_HOME/opp

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --data_dir=$dataset_path --result_path=./preprocess_Result/ --dataset=$dataset_name --dataset_type=$data_type --scaler_path=../ckpt --batch_size=$batch_size
}

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result

    ../ascend310_infer/out/main --mindir_path=$model --input_path=./preprocess_Result --dataset=$dataset_name --device_id=$device_id &> infer.log

}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --label_path=./preprocess_Result --dataset=$dataset_name --dataset_type=$data_type --metrics=$metrics --batch_size=$batch_size --scaler_path=../ckpt &> metrics.log
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess dataset failed"
        exit 1
    fi
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
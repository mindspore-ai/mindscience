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
if [ $# != 2 ]
then
    echo "Usage: bash run_eval_gpu.sh [CKPT_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CKPT_PATH=$(get_real_path $1)
echo $CKPT_PATH

if [ ! -f $CKPT_PATH ]
then
    echo "error: CKPT_PATH=$CKPT_PATH is not a file"
exit 1
fi

cd ..
export DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=$2

echo "start eval for device $DEVICE_ID"
env > env.log
python eval.py \
    --device_target=GPU \
    --ckpt_path=$CKPT_PATH > log.txt 2>&1 &

#!/bin/bash

if [ $# != 2 ]
then
    echo "Usage: bash run_eval_ascend.sh [CKPT_PATH] [DEVICE_ID]"
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

export DEVICE_NUM=1
export DEVICE_ID=$2

export RANK_ID=$2
export RANK_SIZE=1

cd ..
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log

python eval.py --checkpoint_path=$CKPT_PATH > eval.log 2>&1 &

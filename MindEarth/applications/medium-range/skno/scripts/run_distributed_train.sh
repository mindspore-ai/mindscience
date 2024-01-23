#!/bin/bash
set -e
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh  RANK_SIZE"
echo "For example: bash run.sh 8"
echo "This example is expected to run on the Ascend environment."
echo "=============================================================================================================="
if [ $# != 3 ]
then
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE] [DEVICE_NUM] [DEVICE_START_ID]"
exit 1
fi

if [ ! -f $1 ]
then
    echo "error: RANK_TABLE_FILE=$1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=$2
RANK_TABLE_FILE=$(realpath $1)
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"
export DEVICE_START_ID=$3
export root_dir=test_dist_${DEVICE_NUM}pcs

for((i=$DEVICE_START_ID;i<$[$DEVICE_NUM+$DEVICE_START_ID];i++))
do
    rm -rf ${root_dir}/device$i
    mkdir -p ${root_dir}/device$i
    cp ./main.py ./configs/skno.yaml ./${root_dir}/device$i
    cp -r ./src ./${root_dir}/device$i/
    cd ./${root_dir}/device$i
    export DEVICE_ID=$i
    export RANK_ID=$[$i-$DEVICE_START_ID]
    echo "start training for device $i"
    env > env$i.log
    nohup python ./main.py\
      --yaml ./skno.yaml\
      --run_mode "train" \
      >train.log$i 2>&1 &
    cd ../../
done
echo "The program launch succeed, the log is under device0/train.log0."

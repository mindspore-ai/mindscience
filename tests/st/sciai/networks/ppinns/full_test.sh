#!/bin/bash

TEST_DIR=$(pwd)

PPINNMS_DIR=${TEST_DIR}/../../../../../SciAI/sciai/model/ppinns/
TRAIN_FILE=train.py
TRAIN_PATH=${PPINNMS_DIR}${TRAIN_FILE}

echo -e "TEST starts, run graph mode..."
SECONDS=0
CMD="mpiexec --allow-run-as-root -n 6 python ${TRAIN_PATH} --epochs 50000 --lbfgs true --lbfgs_epochs 50000"
loss=$(timeout 3600 "$CMD" | grep -oP "loss:\ (\d+\.\d+)"  | tail -n 1 | awk '{print $2}')

if (( $(echo "$loss > 0.002" | bc -l) )); then
  echo -e "===============TEST FAILED, loss($loss) is greater than 0.002 in ${SECONDS}s==============="
elif [ ! "$loss" ]; then
  echo -e "===============TEST FAILED, cmd execution failed in ${SECONDS}s==============="
else
  echo -e "===============TEST PASSED, loss($loss) is less than 0.002 in ${SECONDS}s==============="
fi
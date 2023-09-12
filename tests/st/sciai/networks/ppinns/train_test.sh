#!/bin/bash

TEST_DIR=`pwd`

PPINNMS_DIR=${TEST_DIR}/../../../../../SciAI/sciai/model/ppinns/
TRAIN_FILE=train.py
TRAIN_PATH=${PPINNMS_DIR}${TRAIN_FILE}

echo -e "TEST starts, run graph mode..."
SECONDS=0
CMD="mpiexec --allow-run-as-root -n 2 python ${TRAIN_PATH}"
loss=$(timeout 300 $CMD | grep -oP "loss:\ (\d+\.\d+)"  | tail -n 1 | awk '{print $2}')

if (( $(echo "$loss > 2" | bc -l) )); then
  echo -e "===============TEST FAILED, loss($loss) is greater than 2 in ${SECONDS}s==============="
elif [ ! $loss ]; then
  echo -e "===============TEST FAILED, cmd execution failed in ${SECONDS}s==============="
else
  echo -e "===============TEST PASSED, loss($loss) is less than 2 in ${SECONDS}s==============="
fi

SECONDS=0
CMD="mpiexec --allow-run-as-root -n 2 python ${TRAIN_PATH} --load_ckpt true --epochs 100"
loss=$(timeout 300 $CMD | grep -oP "loss:\ (\d+\.\d+)"  | tail -n 1 | awk '{print $2}')

if (( $(echo "$loss > 2" | bc -l) )); then
  echo -e "===============TEST FAILED, loss($loss) is greater than 2 in ${SECONDS}s==============="
elif [ ! $loss ]; then
  echo -e "===============TEST FAILED, cmd execution failed in ${SECONDS}s==============="
else
  echo -e "===============TEST PASSED, loss($loss) is less than 2 in ${SECONDS}s==============="
fi

CKPT_PATH=${PPINNMS_DIR}/checkpoints/fine_solver_1_float32/result_iter_1.ckpt
SECONDS=0
CMD="mpiexec --allow-run-as-root -n 2 python ${TRAIN_PATH} \
--t_range 0 10 --nt_coarse 1001 --nt_fine 200001  --n_train 10000 --layers 1 20 20 1 \
--save_ckpt false --save_fig false  --load_ckpt true --save_ckpt_path ./checkpoints --save_output false \
--save_data_path ./output --load_ckpt_path ${CKPT_PATH} \
--figures_path ./figures --log_path ./logs --print_interval 10 --epochs 100 --lbfgs false --lbfgs_epochs 300 \
--amp_level O3"
loss=$(timeout 300 $CMD | grep -oP "loss:\ (\d+\.\d+)"  | tail -n 1 | awk '{print $2}')

if (( $(echo "$loss > 2" | bc -l) )); then
  echo -e "===============TEST FAILED, loss($loss) is greater than 2 in ${SECONDS}s==============="
elif [ ! $loss ]; then
  echo -e "===============TEST FAILED, cmd execution failed in ${SECONDS}s==============="
else
  echo -e "===============TEST PASSED, loss($loss) is less than 2 in ${SECONDS}s==============="
fi
#!/bin/bash

input="$1"

count=$(echo "$input" | tr ',' '\n' | grep -c '[0-9]')

IFS=';' read -r -a input5 <<< $3

raw_feat=${input5[0]}
restr=${input5[1]:-None}
ckpt_path=${input5[2]}
iter=${input5[3]}
num_recycle=${input5[4]}

export MS_ASCEND_CHECK_OVERFLOW_MODE=SATURATION_MODE
# export MS_MEMORY_STATISTIC=1
# export MS_KERNEL_LAUNCH_SKIP=all
export ASCEND_RT_VISIBLE_DEVICES=$input
export HCCL_CONNECT_TIMEOUT=6000
# export MS_ALLOC_CONF="memory_tracker:True"
# export MS_DEV_DUMP_IR_PASSES="hwopt_d_after_stream_assign,valid,graph_build"

# export GLOG_v=2
# export MS_DEV_DUMP_IR_PASSES="step_parallel,validate,stream"
# export GRAPH_OP_RUN=1
#export MS_DEV_DDE_ONLY_MARK=1
# export MINDSPORE_DUMP_CONFIG=/autotest/protein/mindscience/MindSPONGE/applications/MEGAProtein/dump_af.json

ulimit -u unlimited
ulimit -s 102400
ulimit -SHn 65535
mpirun -n $count --output-filename ./log_distribute2 --merge-stderr-to-stdout --allow-run-as-root python infer_main.py --seq_len $2 --raw_feat $raw_feat --restr $restr --ckpt_path $ckpt_path --iter $iter --num_recycle $num_recycle --device_num $count > ./log_distribute2/test_distribute_log 2>&1
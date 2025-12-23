#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================

rm -rf msrun_log
mkdir msrun_log
export ASCEND_RT_VISIBLE_DEVICES=4,5
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_parallel.sh"
echo "=============================================================================================================="

msrun --worker_num=2 --local_worker_num=2 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 train.py \
  --run_name protenix_train \
  --base_dir ./output \
  --train_crop_size 384 \
  --checkpoint_interval 400 \
  --lr 0.001 \
  --data.msa.enable false \
  --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list ./test_dataset/processed_data/pdb_list.txt \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.mmcif_dir ./test_dataset \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.indices_fpath ./test_dataset/processed_data/train_demo_384.csv.gz \
  --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.bioassembly_dict_dir ./test_dataset/processed_data \
  --data.test_sets recentPDB_1536_sample384_0925 \
  --data.recentPDB_1536_sample384_0925.base_info.pdb_list ./test_dataset/processed_data/pdb_list.txt \
  --data.recentPDB_1536_sample384_0925.base_info.mmcif_dir ./test_dataset \
  --data.recentPDB_1536_sample384_0925.base_info.indices_fpath ./test_dataset/processed_data/train_demo_384.csv.gz \
  --data.recentPDB_1536_sample384_0925.base_info.bioassembly_dict_dir ./test_dataset/processed_data

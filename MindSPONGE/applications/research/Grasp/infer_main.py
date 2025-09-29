import argparse
import os
import glob
import time
import datetime
import pickle
import pandas as pd
import numpy as np
from restraint_sample import BINS

import mindspore
from mindspore import context
import mindspore.communication as D
from mindspore import Tensor, ops


parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--raw_feat', default='./grasp2/features.pkl', help='Location of raw features pickle input') #/job/dataset/csp/raw_feat/5JDS.pkl './examples_7STZ_2152/features.pkl'  './5JDS.pkl'  './6HTX.pkl' 'T0001_features.pkl' ./grasp/features.pkl
parser.add_argument('--output_dir', default='./compare_with_parallel', help='Output directory for predictions') #/job/output/test
parser.add_argument('--restr', default="./grasp2/restr_5perc.pkl", help='Location of restraints pickle input, if not provided, will infer without restraints') # ./grasp2/restr_5perc.pkl
parser.add_argument('--ckpt_path', default="./step_14000.ckpt", help='ckpt path')#/job/output/ckpt_dir/ft-grasp-v11-64/step_8000.ckpt params_model_1_multimer_v3_ms.ckpt
parser.add_argument('--data_config', default="./config/data-infer.yaml", help='data process config') # ./config/data-infer.yaml
parser.add_argument('--model_config', default="./config/model-infer.yaml", help='model config') # ./config/model-infer.yaml
parser.add_argument('--seq_len', default=8192, type=int) # sequence will be padded to this length 256
parser.add_argument('--mixed_precision', default=1, type=int)
parser.add_argument('--multimer', default=1, type=int)
parser.add_argument('--device_num', default=8, type=int)
parser.add_argument('--iter', default=5, type=int)
parser.add_argument('--num_recycle', default=20, type=int)



arguments = parser.parse_args()
# context.set_context(device_target="Ascend", device_id=6)
# context.set_context(device_target="Ascend", device_id=7, mode=mindspore.GRAPH_MODE, save_graphs=1, save_graphs_path='./compare_with_parallel/single_graphs/') #, save_graphs=1, save_graphs_path='./compare_with_parallel/single_graphs/'
# from utils_infer_single import infer_config, infer_batch, DataGenerator, ModelGenerator, grasp_infer

context.set_context(device_target="Ascend", 
                    mode=mindspore.GRAPH_MODE, 
                    max_call_depth=24000, 
                    max_device_memory='58GB',
                    # save_graphs=True,
                    # save_graphs_path='./compare_with_parallel/graphs/'
                    # save_graphs=True
                    # memory_optimize_level="O1",
                    # jit_syntax_level=0
                    # variable_memory_max_size="30GB"
                    # save_graphs=1, save_graphs_path='./compare_with_parallel/graphs_25/'
                    )#, save_graphs=1, save_graphs_path='./compare_with_parallel/graphs/', save_graphs=1, save_graphs_path='./compare_with_parallel/graphs_24/', jit_config={"jit_level": "O0"} , memory_optimize_level="O1", jit_syntax_level=0
split_rank = arguments.device_num
data_strategy=((split_rank,),(split_rank,),(1,split_rank),(1,split_rank,1),(1,split_rank,1,1),
                (split_rank,),(split_rank,),(split_rank,),(split_rank,),(1,split_rank),
                (split_rank,1),(1, split_rank, 1),(1,split_rank),(1,split_rank),(1,split_rank),
                (split_rank,1),(split_rank,1),(split_rank,1,1), (split_rank, 1), (split_rank,) ,(split_rank,1,1),(split_rank,1),(1,split_rank,1))
# data_strategy=((1, split_rank, 1),)
# data_strategy=((split_rank,1,1),(split_rank,1,1), (1, split_rank), (split_rank, 1, 1), (split_rank, 1))
mindspore.set_auto_parallel_context(device_num=split_rank, parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL, dataset_strategy=data_strategy, enable_alltoall=False)  # 数据集按数据并行的方式切分，且shard的输出张量也按数据并行方式切分, search_mode="sharding_propagation", 
D.init()
from utils_infer import infer_config, infer_batch, DataGenerator, ModelGenerator, grasp_infer


# print(arguments)
model_gen = ModelGenerator(arguments, arguments.ckpt_path)

with open(arguments.raw_feat, 'rb') as f:
    raw_feature = pickle.load(f)

restr = None
if arguments.restr != "None":
    with open(arguments.restr, 'rb') as f:
        restr = pickle.load(f)

print("debug raw_feat keys", raw_feature.keys())
t1 = time.time()
grasp_infer(model_gen=model_gen, 
                 ckpt_id=8000,
                 raw_feature=raw_feature,
                 restraints=restr,
                 output_prefix=f'{arguments.output_dir}/test6_{arguments.seq_len}', 
                 iter=arguments.iter,
                 seed=9,
                 num_recycle=arguments.num_recycle,
                 device_num=arguments.device_num
                 )

t2 = time.time()
print("Inference done!")
print("time cost: ", t2 - t1)
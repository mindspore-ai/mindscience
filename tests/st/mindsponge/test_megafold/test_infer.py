# Copyright 2023 Huawei Technologies Co., Ltd
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
"""eval script"""
import argparse
import os
import stat
import time
import pickle
import numpy as np

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, load_checkpoint
from mindsponge.cell.amp import amp_convert
from mindsponge.common.config_load import load_config

from data import Feature, get_raw_feature
from model import MegaFold, compute_confidence


parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', default="./config/data.yaml", help='data process config')
parser.add_argument('--model_config', default="./config/model_ge.yaml", help='model config')
parser.add_argument('--input_path', default="./examples/pkl", help='processed raw feature path')
parser.add_argument('--checkpoint_path', default="./MEGA_Fold_1.ckpt", help='checkpoint path')
parser.add_argument('--device_id', default=0, type=int, help='DEVICE_ID')
parser.add_argument('--seq_len', type=int, help='Run pdb assessment.')
parser.add_argument('--is_910a', type=bool, default=False, help='is is_910a or not')
arguments = parser.parse_args()

if arguments.is_910a:
    TIME_DICT = {
        "256": [400, 30],
        "512": [600, 80],
        "1024": [1000, 250],
        "2048": [5500, 1300],
    }
else:
    TIME_DICT = {
        "256": [350, 20],
        "512": [400, 60],
        "1024": [500, 200],
        "2048": [2000, 1200],
        "3072": [3000, 3200],
    }


def fold_infer(args):
    '''mega fold inference'''
    context.set_context(ascend_config={"precision_mode": "must_keep_origin_dtype"})
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    data_cfg.eval.crop_size = args.seq_len
    model_cfg.seq_length = data_cfg.eval.crop_size

    print("crop_size", model_cfg.seq_length)

    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    load_checkpoint(args.checkpoint_path, megafold)
    if args.mixed_precision:
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megafold, fp32_white_list)
    else:
        megafold.to_float(mstype.float32)

    seq_files = os.listdir(args.input_path)

    feature_generator = None
    exec_time_list = []
    for seq_file in seq_files:
        raw_feature = get_raw_feature(os.path.join(args.input_path, seq_file), feature_generator, True)
        ori_res_length = raw_feature['msa'].shape[1]
        processed_feature = Feature(data_cfg, raw_feature)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                                   mixed_precision=args.mixed_precision)
        prev_pos = Tensor(prev_pos)
        prev_msa_first_row = Tensor(prev_msa_first_row)
        prev_pair = Tensor(prev_pair)
        for i in range(data_cfg.common.num_recycle):
            feat_i = [Tensor(x[i]) for x in feat]
            t_start = time.time()
            result = megafold(*feat_i,
                              prev_pos,
                              prev_msa_first_row,
                              prev_pair)
            t_end = time.time()
            time_cost = t_end - t_start
            if i == 0:
                first_time = time_cost
            else:
                exec_time_list.append(time_cost)
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = result
        final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
        confidence, _ = compute_confidence(predicted_lddt_logits, return_lddt=True)
        exec_time = sum(exec_time_list) / (data_cfg.common.num_recycle - 1)
        compile_time = first_time - exec_time
        exec_time = exec_time * data_cfg.common.num_recycle
    return final_atom_positions, confidence, compile_time, exec_time

def check_res(final_atom_positions, confidence, compile_time, exec_time, args):
    '''check result'''
    with open("pos_target.pkl", "rb") as f:
        final_atom_positions_gt = pickle.load(f)
    pos_error = np.mean(np.abs(final_atom_positions_gt - final_atom_positions).astype(np.float64))
    out_res = f"pos_error:{pos_error} \n" \
              f"confidence: {confidence} \n" \
              f"compile_time: {compile_time} \n" \
              f"exec_time: {exec_time} \n"
    print(out_res)
    os_flags = os.O_RDWR | os.O_CREAT
    os_modes = stat.S_IRWXU
    res_path = f'./Megafold_{args.seq_len}_result.log'
    with os.fdopen(os.open(res_path, os_flags, os_modes), 'w') as fout:
        fout.write(out_res)

    assert pos_error < 1
    assert confidence > 91
    assert compile_time < TIME_DICT[str(args.seq_len)][0]
    assert exec_time < TIME_DICT[str(args.seq_len)][0]


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        memory_optimize_level="O1",
                        max_call_depth=6000,
                        device_id=arguments.device_id)
    arguments.mixed_precision = 1
    res = fold_infer(arguments)
    check_res(*res, arguments)

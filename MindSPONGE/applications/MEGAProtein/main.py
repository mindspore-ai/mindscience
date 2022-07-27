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
"""eval script"""
import argparse
import pickle
import os
import json
import time
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, Parameter
from mindspore import load_checkpoint, load_param_into_net
from mindsponge.cell.initializer import do_keep_cell_fp32
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction
from data import Feature
from model import MegaFold, compute_confidence


parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', help='data process config')
parser.add_argument('--model_config', help='model config')
parser.add_argument('--pkl_path', help='processed raw feature path')
parser.add_argument('--checkpoint_path', help='checkpoint path')
parser.add_argument('--device_id', default=1, type=int, help='DEVICE_ID')
parser.add_argument('--mixed_precision', default=1, type=int, help='whether to use mixed precision')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
arguments = parser.parse_args()


def load_pkl(pickle_path):
    '''load pkl'''
    f = open(pickle_path, "rb")
    data = pickle.load(f)
    f.close()
    return data


def fold_infer(args):
    '''mega fold inference'''
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    param_dict = load_checkpoint(args.checkpoint_path)
    new_param_dict = {}
    for key in param_dict.keys():
        if 'template_embedding._flat_templates_slice' in key or           \
           'template_embedding._flag_query_slice' in key or               \
           'template_embedding.template_embedder.idx_num_block' in key or \
           'template_embedding.template_embedder.idx_batch_loop' in key:
            continue
        else:
            new_param_dict[key] = Parameter(Tensor(param_dict[key]), name=key)
    load_param_into_net(megafold, new_param_dict)
    if args.mixed_precision:
        megafold.to_float(mstype.float16)
        do_keep_cell_fp32(megafold)
    else:
        megafold.to_float(mstype.float32)

    seq_files = os.listdir(args.pkl_path)
    for seq_file in seq_files:
        t1 = time.time()
        seq_name = seq_file.split('.')[0]
        raw_feature = load_pkl(args.pkl_path + seq_file)
        ori_res_length = raw_feature['msa'].shape[1]
        processed_feature = Feature(data_cfg, raw_feature)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                                   mixed_precision=args.mixed_precision)
        t2 = time.time()
        for i in range(data_cfg.common.num_recycle):
            feat_i = [Tensor(x[i]) for x in feat]
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = megafold(*feat_i,
                                                                                      prev_pos,
                                                                                      prev_msa_first_row,
                                                                                      prev_pair)
        t3 = time.time()
        final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
        final_atom_mask = feat[16][0][:ori_res_length]
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
        confidence = compute_confidence(predicted_lddt_logits)
        unrelaxed_protein = from_prediction(final_atom_positions, final_atom_mask,
                                            feat[4][0][:ori_res_length], feat[17][0][:ori_res_length])
        pdb_file = to_pdb(unrelaxed_protein)
        os.makedirs(f'./result/seq_{seq_name}_{model_cfg.seq_length}', exist_ok=True)
        with open(os.path.join(f'./result/seq_{seq_name}_{model_cfg.seq_length}',
                               f'unrelaxed_model_{seq_name}.pdb'), 'w') as file:
            file.write(pdb_file)
        t4 = time.time()
        timings = {"pre_process_time": round(t2 - t1, 2),
                   "predict time ": round(t3 - t2, 2),
                   "pos_process_time": round(t4 - t3, 2),
                   "all_time": round(t4 - t1, 2),
                   "confidence": confidence}
        print(timings)
        with open(f'./result/seq_{seq_name}_{model_cfg.seq_length}/timings', 'w') as f:
            f.write(json.dumps(timings))

if __name__ == "__main__":
    if arguments.run_platform == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            max_device_memory="31GB",
                            device_id=arguments.device_id)
    elif arguments.run_platform == 'GPU':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            max_device_memory="31GB",
                            device_id=arguments.device_id,
                            enable_graph_kernel=True,
                            graph_kernel_flags="--enable_expand_ops_only=Softmax --enable_cluster_ops_only=Add")
    else:
        raise Exception("Only support GPU or Ascend")

    fold_infer(arguments)

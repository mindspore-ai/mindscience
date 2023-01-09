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
import os
import stat
import json
import time
import numpy as np

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, nn
from mindsponge.cell.amp import amp_convert
from mindsponge.common.config_load import load_config
from common.protein import to_pdb, from_prediction

from data import MultimerFeature, RawFeatureGenerator, multimer_get_crop_size, multimer_get_raw_feature
from model import MegaFoldMultimer, compute_confidence


parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', default='./config/data.yaml', help='data process config')
parser.add_argument('--model_config', default='./config/model.yaml', help='model config')
parser.add_argument('--multimer_input_path', nargs='+', help='Multimer processed raw feature path')
parser.add_argument('--use_pkl', default=False, help="use pkl as input or fasta file as input, in default use fasta")
parser.add_argument('--checkpoint_path', help='checkpoint path')
parser.add_argument('--device_id', default=0, type=int, help='DEVICE_ID')
parser.add_argument('--mixed_precision', default=0, type=int,
                    help='whether to use mixed precision, 0 for full fp32 and 1 for fp32/fp16 mixed,\
                          only Ascend supports mixed precision, GPU should use fp32')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
arguments = parser.parse_args()


def fold_multimer_infer(args):
    '''mega multimer fold inferenct'''
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    data_cfg.eval.crop_size = multimer_get_crop_size(args.multimer_input_path, args.use_pkl)
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold_multimer = MegaFoldMultimer(model_cfg, mixed_precision=args.mixed_precision)
    load_checkpoint(args.checkpoint_path, megafold_multimer)
    if args.mixed_precision:
        dtype = np.float16
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megafold_multimer, fp32_white_list)
    else:
        dtype = np.float32
        megafold_multimer.to_float(mstype.float32)

    if not args.use_pkl:
        feature_generator = RawFeatureGenerator(database_search_config=data_cfg.database_search)
    else:
        feature_generator = None
    t1 = time.time()
    if args.use_pkl:
        seq_name = (args.multimer_input_path[0]).split('/')[-1].split('.')[0]
    else:
        seq_name = (args.multimer_input_path[0]).split('/')[-1].split('_')[0]
    raw_feature = multimer_get_raw_feature(args.multimer_input_path, feature_generator, args.use_pkl)
    ori_res_length = raw_feature['msa'].shape[1]
    processed_feature = MultimerFeature(args.mixed_precision)
    prev_pos = Tensor(np.zeros([data_cfg.eval.crop_size, 37, 3]).astype(dtype))
    prev_msa_first_row = Tensor(np.zeros([data_cfg.eval.crop_size, 256]).astype(dtype))
    prev_pair = Tensor(np.zeros([data_cfg.eval.crop_size, data_cfg.eval.crop_size, 128]).astype(dtype))
    t2 = time.time()
    for i in range(data_cfg.common.num_recycle):
        feat = processed_feature.pipeline(model_cfg, data_cfg, raw_feature)
        feat_i = [Tensor(x) for x in feat]
        prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = megafold_multimer(*feat_i,
                                                                                           prev_pos,
                                                                                           prev_msa_first_row,
                                                                                           prev_pair)
        prev_pos.asnumpy()
        prev_msa_first_row.asnumpy()
        prev_pair.asnumpy()
        predicted_lddt_logits.asnumpy()
        t3 = time.time()
        if i == data_cfg.common.num_recycle - 1:
            final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
            final_atom_mask = feat[16][:ori_res_length]
            predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
            confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)

            b_factors = plddt[:, None] * final_atom_mask

            unrelaxed_protein = from_prediction(final_atom_positions,
                                                final_atom_mask,
                                                feat[0][:ori_res_length],
                                                feat[1][:ori_res_length],
                                                b_factors,
                                                feat[5][:ori_res_length] - 1,
                                                remove_leading_feature_dimension=False)
            pdb_file = to_pdb(unrelaxed_protein)
            os.makedirs(f'./result/{seq_name}', exist_ok=True)
            os_flags = os.O_RDWR | os.O_CREAT
            os_modes = stat.S_IRWXU
            pdb_path = f'./result/{seq_name}/unrelaxed_{seq_name}.pdb'
            with os.fdopen(os.open(pdb_path, os_flags, os_modes), 'w') as fout:
                fout.write(pdb_file)
            t4 = time.time()
            timings = {"pre_process_time": round(t2 - t1, 2),
                       "predict time ": round(t3 - t2, 2),
                       "pos_process_time": round(t4 - t3, 2),
                       "all_time": round(t4 - t1, 2),
                       "confidence": round(confidence, 2)}

            print(timings)
            with os.fdopen(os.open(f'./result/{seq_name}/timings', os_flags, os_modes), 'w') as fout:
                fout.write(json.dumps(timings))


if __name__ == "__main__":
    if arguments.run_platform == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            mempool_block_size="31GB",
                            max_call_depth=6000,
                            device_id=arguments.device_id)
    elif arguments.run_platform == 'GPU':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            max_call_depth=6000,
                            graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                               "--composite_op_limit_size=50",
                            device_id=arguments.device_id,
                            enable_graph_kernel=True)
    else:
        raise Exception("Only support Ascend")
    fold_multimer_infer(arguments)

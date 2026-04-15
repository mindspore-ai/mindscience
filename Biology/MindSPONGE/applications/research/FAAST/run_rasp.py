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
import ast
import os
import stat
import time
import pynvml
import numpy as np
from data import Feature, RawFeatureGenerator, get_crop_size, get_raw_feature
from mindspore import Tensor, nn, load_checkpoint
import mindspore.common.dtype as mstype
import mindspore.context as context
from mindsponge.cell.amp import amp_convert
from mindsponge.common import residue_constants
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction
from model import MegaFold, compute_confidence
from search import mk_hhsearch_db

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', default="./config/data.yaml", help='data process config')
parser.add_argument('--use_custom', type=ast.literal_eval, default=False, help='whether use custom')
parser.add_argument('--model_config', default="./config/model.yaml", help='model config')
parser.add_argument('--input_path', help='processed raw feature path')
parser.add_argument('--restraints_path', type=str, help='Location of training restraints file.')
parser.add_argument('--use_pkl', type=ast.literal_eval, default=False,
                    help="use pkl as input or fasta file as input, in default use fasta")
parser.add_argument('--use_template', type=ast.literal_eval, default=False,
                    help="use_template or not, in default use template")
parser.add_argument('--checkpoint_file', help='checkpoint path')
parser.add_argument('--device_id', default=0, type=int, help='DEVICE_ID')
parser.add_argument('--a3m_path', type=str, help='a3m_path')
parser.add_argument('--template_path', type=str, help='template_path')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
arguments = parser.parse_args()


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = np.where(np.tile(is_gly[..., None].astype("int32"), \
                                   [1, ] * len(is_gly.shape) + [3, ]).astype("bool"), \
                           all_atom_positions[..., ca_idx, :], \
                           all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def contact_evaluation(final_atom_positions, aatype, contact_mask_input):
    '''contact_evaluation'''
    if contact_mask_input.sum() < 1:
        return 1.0
    contact_mask_input = contact_mask_input.astype(np.float32)
    pseudo_beta_pred = pseudo_beta_fn(aatype, final_atom_positions, None)  # CA as CB for glycine
    cb_distance_pred = np.sqrt((np.square(pseudo_beta_pred[None] - pseudo_beta_pred[:, None])).sum(-1) + 1e-8)
    has_contact_pred = (cb_distance_pred <= 10).astype(np.float32)  # 8.0 or 10.0

    contact_pred_rate_input = ((has_contact_pred == contact_mask_input) * \
                               contact_mask_input).sum() / (contact_mask_input.sum() + 1e-8)

    return round(contact_pred_rate_input, 4)


def make_contact_info(ori_seq_len, ur_path):
    '''make_contact_info'''
    num_residues = ori_seq_len
    contact_info_mask = np.zeros((num_residues, num_residues))
    if not ur_path:
        return contact_info_mask

    with open(ur_path, encoding='utf-8') as f:
        all_urs = f.readlines()
    all_urs = [i.split('!')[0].rstrip() for i in all_urs]
    useful_urs = []
    for urls in all_urs:
        i = urls.split(" ")
        temp = []
        temp.append(int(i[0]))
        temp.append(int(i[-1]))
        useful_urs.append(temp)

    for i in useful_urs:
        contact_info_mask[i[0], i[1]] = 1
    contact_info_mask = (contact_info_mask + contact_info_mask.T) > 0
    contact_info_mask = contact_info_mask.astype(np.float32)

    return contact_info_mask


def fold_infer(args):
    '''rasp inference'''
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    data_cfg.eval.crop_size = get_crop_size(args.input_path, args.use_pkl)
    model_cfg.seq_length = data_cfg.eval.crop_size
    if args.run_platform == "GPU":
        pynvml.nvmlInit()
        pynvml.nvmlSystemGetDriverVersion()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = info.total / 1024 / 1024 / 1024
        if total <= 25:
            model_cfg.slice = model_cfg.slice_new
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)

    if args.mixed_precision:
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megafold, fp32_white_list)
    else:
        megafold.to_float(mstype.float32)

    temp_names = os.listdir(args.input_path)
    prot_names = []

    if args.use_custom:
        mk_hhsearch_db(args.template_path)
    if not args.use_pkl:
        os.makedirs(args.a3m_path, exist_ok=True)
        os.makedirs(args.template_path, exist_ok=True)
        feature_generator = RawFeatureGenerator(data_cfg.database_search, args.a3m_path, args.template_path,
                                                args.use_custom, args.use_template)
        for key in temp_names:
            if "fas" in key:
                prot_names.append(key)
    else:
        feature_generator = None
        for key in temp_names:
            if "pkl" in key:
                prot_names.append(key)

    load_checkpoint(args.checkpoint_file, megafold)

    for prot_file in prot_names:
        prot_name = prot_file.split('.')[0]
        raw_feature = get_raw_feature(os.path.join(args.input_path, prot_file), feature_generator, args.use_pkl,
                                      prot_name)
        ori_res_length = raw_feature['msa'].shape[1]
        ur_path = f"{args.restraints_path}/{prot_name}.txt"
        contact_info_mask_new = make_contact_info(model_cfg.seq_length, ur_path)
        contact_info_mask_new = Tensor(contact_info_mask_new, mstype.float32)
        processed_feature = Feature(data_cfg, raw_feature)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg, \
                                                        mixed_precision=args.mixed_precision)

        prev_pos = Tensor(prev_pos)
        prev_msa_first_row = Tensor(prev_msa_first_row)
        prev_pair = Tensor(prev_pair)
        t1 = time.time()
        for i in range(4):
            feat_i = [Tensor(x[i]) for x in feat]
            result = megafold(*feat_i,
                              prev_pos,
                              prev_msa_first_row,
                              prev_pair,
                              contact_info_mask_new)
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = result
        eval_res = contact_evaluation(prev_pos.asnumpy()[:ori_res_length],
                                      feat[4][0][:ori_res_length],
                                      contact_info_mask_new[:ori_res_length, :ori_res_length].asnumpy())
        t2 = time.time()
        final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
        final_atom_mask = feat[16][0][:ori_res_length]
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
        confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)
        print("confidence of predicted structrue :", confidence, " , time :", t2 - t1, ", restraint recall :", eval_res)
        b_factors = plddt[:, None] * final_atom_mask

        unrelaxed_protein = from_prediction(final_atom_positions,
                                            final_atom_mask,
                                            feat[4][0][:ori_res_length],
                                            feat[17][0][:ori_res_length],
                                            b_factors)
        pdb_file = to_pdb(unrelaxed_protein)
        os.makedirs("./result/", exist_ok=True)
        unrelaxed_pdb_file_path = os.path.join("./result/", f'{prot_name}.pdb')
        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        with os.fdopen(os.open(unrelaxed_pdb_file_path, os_flags, os_modes), 'w') as fout:
            fout.write(pdb_file)


if __name__ == "__main__":
    if arguments.run_platform == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE,
                            memory_optimize_level="O1",
                            device_target="Ascend",
                            max_call_depth=6000,
                            device_id=arguments.device_id)
        arguments.mixed_precision = 1
    elif arguments.run_platform == 'GPU':
        context.set_context(mode=context.GRAPH_MODE,
                            memory_optimize_level="O1",
                            device_target="GPU",
                            max_call_depth=6000,
                            device_id=arguments.device_id,)
        arguments.mixed_precision = 0
    fold_infer(arguments)

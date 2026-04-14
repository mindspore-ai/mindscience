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
# pylint: disable=C0411
# pylint: disable=C0413

"""eval script"""
import argparse
import os
import ast
import stat
import pickle
import pynvml
import sys
import numpy as np

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, load_checkpoint
from mindsponge.cell.amp import amp_convert
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction
from mindsponge.common import residue_constants
from commons.analysis import predur_vs_predpdb, filter_ur_with_pdb
from data import Feature, RawFeatureGenerator, get_crop_size, get_raw_feature
from model import MegaFold, compute_confidence
from assign_settings import assign_all_settings
from assign.assign import assign_iteration
from assign.init_assign import init_assign_call

sys.path.append("../../common_utils/")
from openmm_relaxation.run_relax import run_relax

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', default="./config/data.yaml", help='data process config')
parser.add_argument('--model_config', default="./config/model.yaml", help='model config')
parser.add_argument('--use_custom', type=ast.literal_eval, default=False, help='whether use custom')
parser.add_argument('--input_path', help='processed raw feature path')
parser.add_argument('--pdb_path', type=str, help='Location of training pdb file.')
parser.add_argument('--peak_and_cs_path', type=str, default="./pdb_peaklist", help='peak list and chemical shift path.')
parser.add_argument('--use_pkl', type=ast.literal_eval, default=False,
                    help="use pkl as input or fasta file as input, in default use fasta")
parser.add_argument('--use_template', type=ast.literal_eval, default=False,
                    help="use_template or not, in default use template")
parser.add_argument('--checkpoint_file', help='checkpoint path')
parser.add_argument('--device_id', default=0, type=int, help='DEVICE_ID')
parser.add_argument('--a3m_path', type=str, help='a3m_path')
parser.add_argument('--template_path', type=str, help='template_path')
parser.add_argument('--output_path', type=str, help='final result path')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')

arguments = parser.parse_args()


def init_assign_with_pdb(prot_path, ur_path, ur_tuple_path, ref_pdb=None):
    '''init_assign_with_pdb'''
    print("\nInitial assignment:")
    prot_name = prot_path.split("/")[-1]
    ur_list, ur_list_tuple = init_assign_call(prot_path=prot_path)
    if ref_pdb:
        print(f"Filtering restraint with given structure.")
        ur_list_tuple = filter_ur_with_pdb(ur_list, ref_pdb)
    os_flags = os.O_RDWR | os.O_CREAT
    os_modes = stat.S_IRWXU
    with os.fdopen(os.open(ur_path + "/" + prot_name + ".pkl", os_flags, os_modes), "wb") as fout:
        pickle.dump(ur_list, fout)
    with os.fdopen(os.open(ur_tuple_path + "/" + prot_name + ".pkl", os_flags, os_modes), "wb") as fout:
        pickle.dump(ur_list_tuple, fout)


def restraint_evaluation(final_atom_positions, aatype, restraint_mask_input):
    '''restraint_evaluation'''
    if restraint_mask_input.sum() < 1:
        return 1.0
    restraint_mask_input = restraint_mask_input.astype(np.float32)
    pseudo_beta_pred = pseudo_beta_fn(aatype, final_atom_positions, None)  # CA as CB for glycine
    cb_distance_pred = np.sqrt((np.square(pseudo_beta_pred[None] - pseudo_beta_pred[:, None])).sum(-1) + 1e-8)
    has_restraint_pred = (cb_distance_pred <= 10).astype(np.float32)  # 8.0 or 10.0

    restraint_pred_rate_input = ((has_restraint_pred == restraint_mask_input) * \
                                  restraint_mask_input).sum() / (restraint_mask_input.sum() + 1e-8)

    return round(restraint_pred_rate_input, 4)


def analysis(predur_path, predpdb_path, filter_names, iter_idx):
    '''analysis'''
    output_predur_vs_predpdb, confs = predur_vs_predpdb(predur_path=predur_path,
                                                        predpdb_path=predpdb_path,
                                                        filter_names=filter_names,
                                                        return_conf=True)
    if len(output_predur_vs_predpdb.shape) == 2:
        output_predur_vs_predpdb = output_predur_vs_predpdb[:, [0, 6, 12, 5, 11]]
    else:
        output_predur_vs_predpdb = output_predur_vs_predpdb[[0, 6, 12, 5, 11]]

    outputs_all = output_predur_vs_predpdb

    keys = ["protein name", "restraints number per residue",
            "long restraints number per residue",
            "restraints structure coincidence rate",
            "long restraints structure coincidence rate"]
    print(f"Iteration {iter_idx}:")

    for outputs in outputs_all:
        for key, output in zip(keys, outputs):
            print(key, ": ", output)
        print()

    return confs


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = np.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = np.where(np.tile(is_gly[..., None].astype("int32"), \
                                   [1,] * len(is_gly.shape) + [3,]).astype("bool"), \
                           all_atom_positions[..., ca_idx, :], \
                           all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def make_restraint_info(ori_seq_len, ur_path, distance_threshold=1, sample_ur_rate=0.1):
    '''make_restraint_info'''
    num_residues = ori_seq_len
    restraint_info_mask = np.zeros((num_residues, num_residues))
    if not ur_path:
        return restraint_info_mask

    with open(ur_path, "rb") as f:
        useful_urs = pickle.load(f)

    useful_urs = [[i, j] for i, j in useful_urs if abs(i - j) >= distance_threshold]
    ur_num = int(len(useful_urs) * sample_ur_rate)
    np.random.shuffle(useful_urs)
    useful_urs = useful_urs[:ur_num]

    for i, j in useful_urs:
        restraint_info_mask[int(i) - 1, int(j) - 1] = 1
    restraint_info_mask = (restraint_info_mask + restraint_info_mask.T) > 0
    restraint_info_mask = restraint_info_mask.astype(np.float32)

    return restraint_info_mask


def eval_main(prot_names, megafold, model_cfg, data_cfg, feature_generator):
    'eval_main'
    peaklist_path = arguments.peak_and_cs_path
    for prot_file in prot_names:
        res_path = "./megaassign/"
        res_path_all = "./oriassign/"
        os.makedirs(res_path_all, exist_ok=True)

        for iter_idx in range(len(assign_all_settings)):
            all_settings = assign_all_settings.get(iter_idx)
            print(f"Settings for iteration {iter_idx}")
            print(all_settings, flush=True)
            sample_ur_rate = all_settings.get("infer_pdb")["sample_ur_rate"]

            local_res_path = os.path.join(res_path, f"iter_{iter_idx}")
            local_ur_tuple_path = os.path.join(local_res_path, "ur_tuple")

            next_res_path = os.path.join(res_path, "iter_" + str(iter_idx + 1))
            next_ur_path = os.path.join(next_res_path, "ur")
            next_ur_tuple_path = os.path.join(next_res_path, "ur_tuple")

            local_unrelaxed_pdb_path = os.path.join(local_res_path, "structure")
            local_relaxed_pdb_path = os.path.join(local_res_path, "structure_relaxed")
            for path in [local_res_path, local_ur_tuple_path, local_unrelaxed_pdb_path, local_relaxed_pdb_path,
                         next_ur_path, next_ur_tuple_path, next_res_path]:
                os.makedirs(path, exist_ok=True)
            print("local_res_path ", local_res_path, flush=True)
            os.makedirs(local_res_path, exist_ok=True)

            for repeat_idx in range(all_settings["infer_pdb"]["num_repeats"]):
                prot_name = prot_file.split('.')[0]
                if all_settings["init_assign"]:
                    ur_file_path = None
                else:
                    ur_file_path = os.path.join(local_ur_tuple_path, f"{prot_name}.pkl")
                raw_feature = get_raw_feature(os.path.join(arguments.input_path, prot_file), feature_generator,
                                              arguments.use_pkl, prot_name)
                ori_res_length = raw_feature['msa'].shape[1]
                restraint_info_mask_new = make_restraint_info(model_cfg.seq_length, ur_file_path,
                                                              sample_ur_rate=sample_ur_rate)
                restraint_info_mask_new = Tensor(restraint_info_mask_new, mstype.float32)
                processed_feature = Feature(data_cfg, raw_feature)
                feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg, \
                                                                mixed_precision=arguments.mixed_precision)
                prev_pos = Tensor(prev_pos)
                prev_msa_first_row = Tensor(prev_msa_first_row)
                prev_pair = Tensor(prev_pair)

                for i in range(4):
                    feat_i = [Tensor(x[i]) for x in feat]
                    result = megafold(*feat_i,
                                      prev_pos,
                                      prev_msa_first_row,
                                      prev_pair,
                                      restraint_info_mask_new)
                    prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = result

                eval_res = restraint_evaluation(prev_pos.asnumpy()[:ori_res_length],
                                                feat[4][0][:ori_res_length],
                                                restraint_info_mask_new[:ori_res_length, :ori_res_length].asnumpy())
                restraint_pred_rate_input = eval_res

                final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
                final_atom_mask = feat[16][0][:ori_res_length]
                predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
                confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)

                b_factors = plddt[:, None] * final_atom_mask

                unrelaxed_protein = from_prediction(final_atom_positions,
                                                    final_atom_mask,
                                                    feat[4][0][:ori_res_length],
                                                    feat[17][0][:ori_res_length],
                                                    b_factors)
                pdb_file = to_pdb(unrelaxed_protein)

                unrelaxed_pdb_file_path = os.path.join(local_unrelaxed_pdb_path, f'{prot_name}_{repeat_idx}.pdb')
                relaxed_pdb_file_path = os.path.join(local_relaxed_pdb_path, f'{prot_name}_{repeat_idx}.pdb')
                os_flags = os.O_RDWR | os.O_CREAT
                os_modes = stat.S_IRWXU
                with os.fdopen(os.open(unrelaxed_pdb_file_path, os_flags, os_modes), 'w') as fout:
                    fout.write(pdb_file)
                print(f">>>>>>>>>>>>>>>>>>>>>>Protein name: {prot_name}, iteration: {iter_idx}, "
                      f"repeat: {repeat_idx}, number of input restraint pair: "
                      f"{int(restraint_info_mask_new.asnumpy().sum())}, confidence: {round(confidence, 2)}, "
                      f"input restraint recall: {restraint_pred_rate_input}.", flush=True)
                run_relax(unrelaxed_pdb_file_path, relaxed_pdb_file_path)

            names = os.listdir(arguments.input_path)
            names = [name.split(".")[0] for name in names if name.split(".")[-1] == "pkl"]
            names.sort()
            if all_settings["init_assign"]:
                prot_path = os.path.join(peaklist_path, prot_name)
                init_assign_with_pdb(prot_path, next_ur_path, next_ur_tuple_path, ref_pdb=relaxed_pdb_file_path)
            else:
                assign_iteration(next_ur_tuple_path,
                                 next_ur_path,
                                 local_relaxed_pdb_path,
                                 peaklist_path,
                                 all_settings,
                                 filter_names=names)
                _ = analysis(predur_path=next_ur_path,
                             predpdb_path=local_relaxed_pdb_path,
                             filter_names=names,
                             iter_idx=iter_idx)

        save_res_local = f"{res_path_all}/{prot_name}"
        os.system(f"mv {res_path} {save_res_local}")


def fold_infer(args):
    '''faast infer'''
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
    if not args.use_pkl:
        os.makedirs(args.a3m_path, exist_ok=True)
        os.makedirs(args.template_path, exist_ok=True)
        if args.use_custom:
            mk_hhsearch_db(args.template_path)
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

    eval_main(prot_names, megafold, model_cfg, data_cfg, feature_generator)


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

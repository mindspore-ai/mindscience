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
import numpy as np
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore import load_checkpoint
from mindsponge.cell.initializer import do_keep_cell_fp32
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction
from data import Feature, RawFeatureGenerator
from data.dataset import create_dataset
from model import MegaFold, compute_confidence
from module.fold_wrapcell import TrainOneStepCell, WithLossCell

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', help='data process config')
parser.add_argument('--model_config', help='model config')
parser.add_argument('--input_path', help='processed raw feature path')
parser.add_argument('--use_pkl', default=True, help="use pkl as input or fasta file as input, in default use pkl")
parser.add_argument('--checkpoint_path', help='checkpoint path')
parser.add_argument('--device_id', default=1, type=int, help='DEVICE_ID')
parser.add_argument('--mixed_precision', default=0, type=int,
                    help='whether to use mixed precision, only Ascend supports mixed precision, GPU should use fp32')
parser.add_argument('--is_training', type=bool, default=False, help='is training or not')
parser.add_argument('--pdb_data_dir', type=str, help='Location of training pdb file.')
parser.add_argument('--raw_feature_dir', type=str, help='Location of raw inputs file.')
parser.add_argument('--resolution_data', type=str, default=None, help='Location of resolution data file.')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')
parser.add_argument('--gradient_clip', type=float, default=0.1, help='gradient clip value')
parser.add_argument('--total_steps', type=int, default=9600000, help='total steps')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
parser.add_argument('--run_distribute', type=bool, default=False, help='run distribute')
parser.add_argument('--template_mmcif_dir', help="path of template mmCIF structures, each named <pdb>.cif")
parser.add_argument('--max_template_date', default="2100-01-01", help="maximum template release date to consider")
parser.add_argument('--kalign_binary_path', help="path of executable path of Kalign")
parser.add_argument('--hhsearch_binary_path', help="path of executable path of HHsearch")
parser.add_argument('--mmseqs_binary', help="path of executable path of mmseqs")
parser.add_argument('--pdb70_database_path', help="database use for HHsearch")
parser.add_argument('--database_envdb_dir', help="database use for mmseqs")
parser.add_argument('--obsolete_pdbs_path', help="path to a file containing a mapping from obsolete PDB IDs to the"
                                                 " replacement files")
parser.add_argument('--uniref30_path', help="database used for searching msa by mmseqs")
parser.add_argument('--a3m_result_path', help="result path for saving msa file of target input")


arguments = parser.parse_args()


def get_raw_feature(input_path, feature_generator, use_pkl):
    '''get raw feature of protein by loading pkl file or searching from database'''
    if use_pkl:
        f = open(input_path, "rb")
        data = pickle.load(f)
        f.close()
        return data
    return feature_generator.monomer_feature_generate(input_path)


def fold_infer(args):
    '''mega fold inference'''
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    load_checkpoint(args.checkpoint_path, megafold)
    if args.mixed_precision:
        megafold.to_float(mstype.float16)
        do_keep_cell_fp32(megafold)
    else:
        megafold.to_float(mstype.float32)

    seq_files = os.listdir(args.input_path)

    if not args.use_pkl:
        feature_generator = RawFeatureGenerator(template_mmcif_dir=args.template_mmcif_dir,
                                                max_template_date=args.max_template_date,
                                                kalign_binary_path=args.kalign_binary_path,
                                                obsolete_pdbs_path=args.obsolete_pdbs_path,
                                                hhsearch_binary_path=args.hhsearch_binary_path,
                                                pdb70_database_path=args.pdb70_database_path,
                                                database_envdb_dir=args.database_envdb_dir,
                                                mmseqs_binary=args.mmseqs_binary,
                                                uniref30_path=args.uniref30_path,
                                                a3m_result_path=args.a3m_result_path,
                                                )
    else:
        feature_generator = None
    for seq_file in seq_files:
        t1 = time.time()
        seq_name = seq_file.split('.')[0]
        raw_feature = get_raw_feature(os.path.join(args.input_path, seq_file), feature_generator, args.use_pkl)
        ori_res_length = raw_feature['msa'].shape[1]
        processed_feature = Feature(data_cfg, raw_feature)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                                   mixed_precision=args.mixed_precision)
        prev_pos = Tensor(prev_pos)
        prev_msa_first_row = Tensor(prev_msa_first_row)
        prev_pair = Tensor(prev_pair)
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


def fold_train(args):
    """megafold train"""
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    model_cfg.is_training = True
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    if args.mixed_precision:
        megafold.to_float(mstype.float16)
        do_keep_cell_fp32(megafold)
    else:
        megafold.to_float(mstype.float32)

    net_with_criterion = WithLossCell(megafold, model_cfg)
    opt = nn.Adam(params=megafold.trainable_params(), learning_rate=1e-4, eps=1e-6)
    train_net = TrainOneStepCell(net_with_criterion, opt, sens=args.loss_scale,
                                 gradient_clip_value=args.gradient_clip)

    train_net.set_train(False)
    step = 0
    np.random.seed(1)
    max_recycles = [int(np.random.uniform(size=1, low=0, high=4)) for _ in range(args.total_steps)]
    max_recycles[step] = 0
    np.random.seed()
    names = os.listdir(args.pdb_data_dir)
    names_list = []
    for name in names:
        names_list.append(name.split(".pdb")[0])
    train_dataset = create_dataset(args.pdb_data_dir, args.raw_feature_dir, names_list, data_cfg,
                                   args.resolution_data, num_parallel_worker=4, is_parallel=args.run_distribute,
                                   shuffle=True)
    dataset_iter = train_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    for d in dataset_iter:
        max_recycle = max_recycles[step]
        inputs_feats = d["target_feat"], d["msa_feat"], d["msa_mask"], d["seq_mask_batch"], d["aatype_batch"], \
                       d["template_aatype"], d["template_all_atom_masks"], d["template_all_atom_positions"], \
                       d["template_mask"], d["template_pseudo_beta_mask"], d["template_pseudo_beta"], \
                       d["extra_msa"], d["extra_has_deletion"], \
                       d["extra_deletion_value"], d["extra_msa_mask"], d["residx_atom37_to_atom14"], \
                       d["atom37_atom_exists_batch"], d["residue_index_batch"]
        prev_pos, prev_msa_first_row, prev_pair = Tensor(d["prev_pos"]), Tensor(d["prev_msa_first_row"]), \
                                                  Tensor(d["prev_pair"])
        ground_truth = d["pseudo_beta_gt"], d["pseudo_beta_mask_gt"], d["all_atom_mask_gt"], \
                       d["true_msa"], d["bert_mask"], d["residx_atom14_to_atom37"], \
                       d["restype_atom14_bond_lower_bound"], d["restype_atom14_bond_upper_bound"], \
                       d["atomtype_radius"], d["backbone_affine_tensor"], d["backbone_affine_mask"], \
                       d["atom14_gt_positions"], d["atom14_alt_gt_positions"], d["atom14_atom_is_ambiguous"], \
                       d["atom14_gt_exists"], d["atom14_atom_exists"], d["atom14_alt_gt_exists"], \
                       d["all_atom_positions"], d["rigidgroups_gt_frames"], d["rigidgroups_gt_exists"], \
                       d["rigidgroups_alt_gt_frames"], d["torsion_angles_sin_cos_gt"], d["use_clamped_fape"], \
                       d["filter_by_solution"], d["chi_mask"]
        # forward recycle 3 steps
        train_net.add_flags_recursive(train_backward=False)
        train_net.phase = 'train_forward'
        ground_truth = [Tensor(gt) for gt in ground_truth]
        for recycle in range(max_recycle):
            inputs_feat = [Tensor(feat[recycle]) for feat in inputs_feats]
            prev_pos, prev_msa_first_row, prev_pair = train_net(*inputs_feat, prev_pos, prev_msa_first_row,
                                                                prev_pair, *ground_truth)
        inputs_feat = [Tensor(feat[max_recycle]) for feat in inputs_feats]
        # forward + backward
        train_net.add_flags_recursive(train_backward=True)
        train_net.phase = 'train_backward'
        loss = train_net(*inputs_feat, prev_pos, prev_msa_first_row, prev_pair, *ground_truth)
        loss_info = f"step is: {step}, total_loss: {loss[0]}, fape_sidechain_loss: {loss[1]}," \
                    f" fape_backbone_loss: {loss[2]}, angle_norm_loss: {loss[3]}, distogram_loss: {loss[4]}," \
                    f" masked_loss: {loss[5]}, plddt_loss: {loss[6]}"
        print(loss_info, flush=True)
        step += 1


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

    if not arguments.is_training:
        fold_infer(arguments)
    else:
        fold_train(arguments)
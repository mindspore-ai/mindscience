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
import ast
import numpy as np

import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, save_checkpoint, load_checkpoint, load_param_into_net
from mindsponge.cell.amp import amp_convert
from mindsponge.cell.mask import LayerNormProcess
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction

from data import Feature, RawFeatureGenerator, create_dataset, get_crop_size, get_raw_feature, process_pdb
from model import MegaFold, compute_confidence, MegaEvogen
from model.assessment import CombineModel, load_weights
from module.fold_wrapcell import TrainOneStepCell, WithLossCell, WithLossCellAssessment
from module.evogen_block import absolute_position_embedding
from module.lr import cos_decay_lr

parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', default="./config/data.yaml", help='data process config')
parser.add_argument('--model_config', default="./config/model.yaml", help='model config')
parser.add_argument('--evogen_config', default="./config/evogen.yaml", help='evogen config')
parser.add_argument('--input_path', help='processed raw feature path')
parser.add_argument('--pdb_path', type=str, help='Location of training pdb file.')
parser.add_argument('--use_pkl', type=ast.literal_eval, default=False,
                    help="use pkl as input or fasta file as input, in default use fasta")
parser.add_argument('--checkpoint_path', help='checkpoint path')
parser.add_argument('--checkpoint_path_assessment', help='assessment model checkpoint path')
parser.add_argument('--device_id', default=0, type=int, help='DEVICE_ID')
parser.add_argument('--is_training', type=ast.literal_eval, default=False, help='is training or not')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='run distribute')
parser.add_argument('--resolution_data', type=str, default=None, help='Location of resolution data file.')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')
parser.add_argument('--gradient_clip', type=float, default=0.1, help='gradient clip value')
parser.add_argument('--total_steps', type=int, default=9600000, help='total steps')
parser.add_argument('--decoy_pdb_path', type=str, help='Location of decoy pdb file.')
parser.add_argument('--run_assessment', type=int, default=0, help='Run pdb assessment.')
parser.add_argument('--run_evogen', type=int, default=0, help='Run pdb assessment.')
arguments = parser.parse_args()


def fold_infer(args):
    '''mega fold inference'''
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    data_cfg.eval.crop_size = get_crop_size(args.input_path, args.use_pkl)
    model_cfg.seq_length = data_cfg.eval.crop_size
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

    if not args.use_pkl:
        feature_generator = RawFeatureGenerator(database_search_config=data_cfg.database_search)
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
            result = megafold(*feat_i,
                              prev_pos,
                              prev_msa_first_row,
                              prev_pair)
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = result
        t3 = time.time()
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


def fold_train(args):
    """megafold train"""
    data_cfg = load_config(args.data_config)
    data_cfg.common.max_extra_msa = 1024
    data_cfg.eval.max_msa_clusters = 128
    model_cfg = load_config(args.model_config)
    model_cfg.is_training = True
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    if args.mixed_precision:
        fp32_white_list = (nn.Softmax, nn.LayerNorm, LayerNormProcess)
        amp_convert(megafold, fp32_white_list)
    else:
        megafold.to_float(mstype.float32)

    net_with_criterion = WithLossCell(megafold, model_cfg)
    if args.run_platform == 'GPU':
        lr = cos_decay_lr(start_step=model_cfg.GPU.start_step, lr_init=0.0,
                          lr_min=model_cfg.GPU.lr_min, lr_max=model_cfg.GPU.lr_max,
                          decay_steps=model_cfg.GPU.lr_decay_steps,
                          warmup_steps=model_cfg.GPU.warmup_steps)
    else:
        lr = model_cfg.ascend.lr
    opt = nn.Adam(params=megafold.trainable_params(), learning_rate=lr, eps=1e-6)
    train_net = TrainOneStepCell(net_with_criterion, opt, sens=args.loss_scale,
                                 gradient_clip_value=args.gradient_clip)

    train_net.set_train(False)
    step = 0
    np.random.seed(1)
    max_recycles = [int(np.random.uniform(size=1, low=0, high=4)) for _ in range(args.total_steps)]
    max_recycles[step] = 0
    np.random.seed()
    names = os.listdir(args.pdb_path)
    names_list = []
    for name in names:
        names_list.append(name.split(".pdb")[0])
    train_dataset = create_dataset(args.pdb_path, args.input_path, names_list, data_cfg,
                                   args.resolution_data, num_parallel_worker=4, is_parallel=args.run_distribute,
                                   shuffle=True, mixed_precision=args.mixed_precision)
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
        ckpt_path = './ckpt/'
        if step % 50 == 0:
            ckpt_name = f"{ckpt_path}/step_{step}.ckpt"
            save_checkpoint(train_net, ckpt_name)
            print(f"checkpoint of step {step} is saved in ./ckpt folder")


def assessment_infer(args):
    '''mega fold inference'''
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    data_cfg.eval.crop_size = get_crop_size(args.input_path, args.use_pkl)
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val
    data_cfg.eval.subsample_templates = False

    megaassessment = CombineModel(model_cfg, mixed_precision=args.mixed_precision)
    load_checkpoint(args.checkpoint_path_assessment, megaassessment)
    if args.mixed_precision:
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megaassessment, fp32_white_list)
    else:
        megaassessment.to_float(mstype.float32)

    seq_files = os.listdir(args.input_path)
    if not args.use_pkl:
        feature_generator = RawFeatureGenerator(database_search_config=data_cfg.database_search)
    else:
        feature_generator = None
    for seq_file in seq_files:
        t1 = time.time()
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
            prev_pos, prev_msa_first_row, prev_pair, _ = megaassessment(*feat_i,
                                                                        prev_pos,
                                                                        prev_msa_first_row,
                                                                        prev_pair)
        for pdb_name in os.listdir(args.decoy_pdb_path):
            decoy_atom_positions, decoy_atom_mask, align_mask = \
            process_pdb(feat[4][0], ori_res_length, os.path.join(args.decoy_pdb_path, pdb_name))
            plddt = megaassessment(*feat_i, prev_pos, prev_msa_first_row, prev_pair,
                                   Tensor(decoy_atom_positions), Tensor(decoy_atom_mask), run_pretrain=False)
            t3 = time.time()
            plddt = plddt.asnumpy()[align_mask == 1]
            confidence = np.mean(plddt)
            t4 = time.time()
            timings = {"seq_name": seq_file,
                       "decoy_pdb_name": pdb_name,
                       "pre_process_time": round(t2 - t1, 2),
                       "predict time ": round(t3 - t2, 2),
                       "pos_process_time": round(t4 - t3, 2),
                       "all_time": round(t4 - t1, 2),
                       "confidence": confidence}
            print(timings)


def assessment_train(args):
    """megafold train"""
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    model_cfg.is_training = True
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val

    megaassessment = CombineModel(model_cfg, mixed_precision=args.mixed_precision)
    param_dict = load_weights(args.checkpoint_path, model_cfg)
    load_param_into_net(megaassessment, param_dict)
    if args.mixed_precision:
        fp32_white_list = (nn.Softmax, nn.LayerNorm)
        amp_convert(megaassessment, fp32_white_list)
    else:
        megaassessment.to_float(mstype.float32)

    net_with_criterion = WithLossCellAssessment(megaassessment, model_cfg)
    opt = nn.Adam(params=megaassessment.trainable_params(), learning_rate=1e-4, eps=1e-6)
    train_net = TrainOneStepCell(net_with_criterion, opt, sens=args.loss_scale,
                                 gradient_clip_value=args.gradient_clip, train_fold=False,
                                 train_assessment=True)

    train_net.set_train(False)
    step = 0
    np.random.seed(1)
    max_recycles = [int(np.random.uniform(size=1, low=0, high=4)) for _ in range(args.total_steps)]
    max_recycles[step] = 0
    np.random.seed()
    names = os.listdir(args.pdb_path)
    names_list = []
    for name in names:
        names_list.append(name.split(".pdb")[0])
    train_dataset = create_dataset(args.pdb_path, args.input_path, names_list, data_cfg,
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
                       d["rigidgroups_alt_gt_frames"], d["torsion_angles_sin_cos_gt"], d["chi_mask"]
        # forward recycle 4 steps
        train_net.add_flags_recursive(train_backward=False)
        train_net.phase = 'train_forward'
        ground_truth = [Tensor(gt) for gt in ground_truth]
        for i in range(4):
            inputs_feat = [Tensor(feat[i]) for feat in inputs_feats]
            ground_truth = [Tensor(gt) for gt in ground_truth]
            prev_pos, prev_msa_first_row, prev_pair = megaassessment(*inputs_feat,
                                                                     prev_pos,
                                                                     prev_msa_first_row,
                                                                     prev_pair)
            if i == max_recycle:
                final_atom_positions_recycle, final_atom_mask_recycle = prev_pos, \
                                                                        Tensor(d["atom37_atom_exists_batch"][0])

        # forward + backward
        train_net.add_flags_recursive(train_backward=True)
        train_net.phase = 'train_backward'
        inputs_feat = [Tensor(feat[max_recycle]) for feat in inputs_feats]
        loss = train_net(*inputs_feat, prev_pos, prev_msa_first_row, prev_pair, *ground_truth,
                         final_atom_positions_recycle, final_atom_mask_recycle, run_pretrain=True)
        loss_info = f"step is: {step}, total loss is :{loss[0]}, fape_side: {loss[1]}, fape_backbone: {loss[2]}," \
                    f"anglenorm: {loss[3]}, predict_lddt_loss: {loss[4]}, distogram_focal_loss: {loss[5]}," \
                    f"distogram_regression_loss: {loss[6]}, plddt2_loss: {loss[7]}, mask_loss: {loss[8]}," \
                    f"confidence_loss: {loss[9]}, cameo_loss: {loss[10]}"
        print(loss_info, flush=True)
        step += 1
        ckpt_path = './ckpt/'
        if step % 50 == 0:
            ckpt_name = f"{ckpt_path}/step_{step}.ckpt"
            save_checkpoint(train_net, ckpt_name)
            print(f"checkpoint of step {step} is saved in ./ckpt folder")


def evogen_augmentation(args):
    '''evogen_augmentation'''

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    evogen_model_cfg = load_config(args.evogen_config)

    data_cfg.eval.crop_size = get_crop_size(args.input_path, args.use_pkl)
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val
    model_cfg.slice.extra_msa_stack.msa_row_attention_with_pair_bias = 0
    model_cfg.is_training = False
    model_cfg.template.enabled = False

    ape_table = absolute_position_embedding(1024, 256, min_timescale=1, max_timescale=1e4)

    evogen_model_cfg.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias.ape_table = ape_table
    data_cfg.max_extra_msa = 2
    data_cfg.num_recycle = 1
    data_cfg.eval.max_msa_clusters = 128
    megaevogen = MegaEvogen(evogen_model_cfg, model_cfg, mixed_precision=args.mixed_precision)
    megaevogen.to_float(mstype.float32)

    data_cfg.common.num_recycle = 1
    load_checkpoint(args.checkpoint_path, megaevogen)
    seq_files = os.listdir(args.input_path)

    if not args.use_pkl:
        feature_generator = RawFeatureGenerator(database_search_config=data_cfg.database_search)
    else:
        feature_generator = None
    for seq_file in seq_files:
        t1 = time.time()
        seq_name = seq_file.split('.')[0]
        raw_feature = get_raw_feature(os.path.join(args.input_path, seq_file), feature_generator, args.use_pkl)
        ori_res_length = raw_feature['msa'].shape[1]
        processed_feature = Feature(data_cfg, raw_feature, model_cfg=evogen_model_cfg, is_evogen=True)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                                   mixed_precision=args.mixed_precision)
        prev_pos = Tensor(prev_pos)
        prev_msa_first_row = Tensor(prev_msa_first_row)
        prev_pair = Tensor(prev_pair)
        t2 = time.time()

        # fake data
        fake_template_aatype = Tensor(0, dtype=mstype.int32)
        fake_template_all_atom_masks = Tensor(0, dtype=mstype.int32)
        fake_template_all_atom_positions = Tensor(0, dtype=mstype.int32)
        fake_template_mask = Tensor(0, dtype=mstype.int32)
        fake_template_pseudo_beta_mask = Tensor(0, dtype=mstype.int32)
        fake_template_pseudo_beta = Tensor(0, dtype=mstype.int32)
        extra_msa_length = 2
        fake_extra_msa = Tensor(np.zeros((extra_msa_length, data_cfg.eval.crop_size), dtype=np.int32))
        fake_extra_has_deletion = Tensor(np.zeros((extra_msa_length, data_cfg.eval.crop_size), dtype=np.float32))
        fake_extra_deletion_value = Tensor(np.zeros((extra_msa_length, data_cfg.eval.crop_size), dtype=np.float32))
        fake_extra_msa_mask = Tensor(np.zeros((extra_msa_length, data_cfg.eval.crop_size), dtype=np.float32))
        fake_data = [fake_template_aatype, fake_template_all_atom_masks, fake_template_all_atom_positions,
                     fake_template_mask, fake_template_pseudo_beta_mask, fake_template_pseudo_beta, fake_extra_msa,
                     fake_extra_has_deletion, fake_extra_deletion_value, fake_extra_msa_mask]

        for i in range(data_cfg.common.num_recycle):
            feat_i = [Tensor(x[i]) for x in feat]
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = megaevogen(
                *feat_i, *fake_data, prev_pos, prev_msa_first_row, prev_pair)

        t3 = time.time()
        final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
        final_atom_mask = feat[4][0][:ori_res_length]
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
        confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)

        b_factors = plddt[:, None] * final_atom_mask

        unrelaxed_protein = from_prediction(final_atom_positions,
                                            final_atom_mask,
                                            feat[2][0][:ori_res_length],
                                            feat[5][0][:ori_res_length],
                                            b_factors)

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


if __name__ == "__main__":
    if arguments.run_platform == 'Ascend' and not arguments.is_training:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            memory_optimize_level="O1",
                            max_call_depth=6000,
                            device_id=arguments.device_id)
        arguments.mixed_precision = 1
    elif arguments.run_platform == 'Ascend' and arguments.is_training:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            max_device_memory="29GB",
                            device_id=arguments.device_id)
        arguments.mixed_precision = 1
    elif arguments.run_platform == 'GPU':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            max_call_depth=6000,
                            graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                               "--composite_op_limit_size=50",
                            device_id=arguments.device_id,
                            enable_graph_kernel=True)
        arguments.mixed_precision = 0
    else:
        raise Exception("Only support GPU or Ascend")

    if arguments.run_assessment:
        if not arguments.is_training:
            assessment_infer(arguments)
        else:
            assessment_train(arguments)
    elif arguments.run_evogen:
        evogen_augmentation(arguments)

    else:
        if not arguments.is_training:
            fold_infer(arguments)
        else:
            fold_train(arguments)

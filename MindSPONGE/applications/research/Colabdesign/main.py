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
"""design main"""
import argparse
import os
import stat
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore import Parameter
from mindspore import Tensor, load_checkpoint
from mindsponge.cell.initializer import do_keep_cell_fp32
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction

from MEGAProtein.model.fold import compute_confidence  # 需要导入MEGAProtein
from model.design_fold import Colabdesign
from module.design_wrapcell import TrainOneStepCell, WithLossCell
from module.utils import get_weights, get_lr, get_opt, get_seqs
from data.prep import DesignPrep

parser = argparse.ArgumentParser(description='Inputs for main.py')
parser.add_argument('--data_config', default="../../MEGAProtein/config/data.yaml",
                    help='Megafold data process config')
parser.add_argument('--design_config', default="./config/design_config.yaml",
                    help='design config change loss_weights to get different loss combinations')
parser.add_argument('--model_config', default="../../config/model.yaml", help='Megafold model config')
parser.add_argument('--protocol', type=str, default='fixbb', help='fixbb or hallucination.')
parser.add_argument('--opt_choice', type=str, default='sgd', help='opt choice sgd or adam')
parser.add_argument('--pdb_path', type=str, help='Location of training pdb file.')
parser.add_argument('--ckpt_url', help='checkpoint path')
parser.add_argument('--device_id', default=7, type=int, help='DEVICE_ID')
parser.add_argument('--soft_iters', default=100, type=int, help='iters of soft')
parser.add_argument('--temp_iters', default=100, type=int, help='iters of temp')
parser.add_argument('--hard_iters', default=100, type=int, help='iters of hard')
parser.add_argument('--mixed_precision', default=0, type=int,
                    help='whether to use mixed precision, 0 for full fp32 and 1 for fp32/fp16 mixed,\
                          only Ascend supports mixed precision, GPU should use fp32')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')
arguments = parser.parse_args()


def fold_design(args):
    """megafold design"""
    data_cfg = load_config(args.data_config)
    data_cfg.common.max_extra_msa = 1024
    data_cfg.eval.max_msa_clusters = 128
    design_cfg = load_config(args.design_config)
    model_cfg = load_config(args.model_config)
    model_cfg.is_training = True
    model_cfg.seq_length = design_cfg.seq_length
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val
    data_prep = DesignPrep(model_cfg, data_cfg)
    inputs_feats, new_feature, ori_seq_len = data_prep.prep_feature(pdb_filename=args.pdb_path, chain="A",
                                                                    protocol=args.protocol)
    seq_vector = Parameter(Tensor(new_feature.get('params_seq'), mstype.float32), requires_grad=True)
    prev_pos, prev_msa_first_row, prev_pair = Tensor(new_feature.get('prev_pos')), Tensor(
        new_feature.get('prev_msa_first_row')), Tensor(new_feature.get('prev_pair'))
    fold_net = Colabdesign(model_cfg, args.mixed_precision, seq_vector, ori_seq_len, design_cfg, protocol=args.protocol)
    if args.mixed_precision:
        fold_net.to_float(mstype.float16)
        do_keep_cell_fp32(fold_net)
    else:
        fold_net.to_float(mstype.float32)
    load_checkpoint(args.ckpt_url, fold_net)

    # process 3 stage hyper-parameter
    total_epoch = args.soft_iters + args.temp_iters + args.hard_iters
    temp_weights, soft_weights, hard_weights = get_weights(design_cfg, args.soft_iters, args.temp_iters,
                                                           args.hard_iters)
    lr = get_lr(temp_weights, soft_weights, total_epoch)
    model_params = [seq_vector]
    weight_decay = 0.0
    opt = get_opt(model_params, lr, weight_decay, args.opt_choice)
    net_with_criterion = WithLossCell(fold_net)
    train_net = TrainOneStepCell(net_with_criterion, opt, sens=args.loss_scale)
    step = 0
    np.random.seed(1)
    temp_weights = Tensor(temp_weights, mstype.float32)
    soft_weights = Tensor(soft_weights, mstype.float32)
    hard_weights = Tensor(hard_weights, mstype.float32)
    best = 999
    for epoch in range(total_epoch):
        train_net.add_flags_recursive(save_best=False)
        train_net.phase = 'save_best'
        inputs_feat = [Tensor(feat) for feat in inputs_feats]
        temp_step = temp_weights[epoch]
        soft_step = soft_weights[epoch]
        hard_step = hard_weights[epoch]
        loss = train_net(
            *inputs_feat,
            temp_step,
            soft_step,
            hard_step,
            prev_pos,
            prev_msa_first_row,
            prev_pair)
        loss_info = f"step is: {step}, total_loss: {loss} , temp: {temp_step}, " \
                    f"soft: {soft_step} ,hard: {hard_step}"
        if loss < best:
            train_net.add_flags_recursive(save_best=True)
            train_net.phase = 'save_best'
            best = loss
            pdb_result = fold_net(
                *inputs_feat,
                temp_step,
                soft_step,
                hard_step,
                prev_pos,
                prev_msa_first_row,
                prev_pair)
            final_atom_positions = pdb_result[0]
            atom37_atom_exists = pdb_result[1]
            predicted_lddt_logits = pdb_result[2]
            aatype = pdb_result[3]
            residue_index = pdb_result[4]
            seq_hard = pdb_result[5].asnumpy()

        print(loss_info, flush=True)
        step += 1
    final_seqs = get_seqs(seq_hard)
    final_atom_positions = final_atom_positions.asnumpy()[:ori_seq_len]
    final_atom_mask = atom37_atom_exists.asnumpy()[:ori_seq_len]
    predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_seq_len]
    _, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)

    b_factors = plddt[:, None] * final_atom_mask

    unrelaxed_protein = from_prediction(final_atom_positions,
                                        final_atom_mask,
                                        aatype[:ori_seq_len].asnumpy(),
                                        residue_index[:ori_seq_len].asnumpy(),
                                        b_factors)
    pdb_file = to_pdb(unrelaxed_protein)
    os.makedirs(f'./result/', exist_ok=True)
    os_flags = os.O_RDWR | os.O_CREAT
    os_modes = stat.S_IRWXU
    pdb_path = f'./result/{args.protocol}.pdb'
    seqs_path = f'./result/{args.protocol}.fa'
    with os.fdopen(os.open(pdb_path, os_flags, os_modes), 'w') as fout:
        fout.write(pdb_file)
    with os.fdopen(os.open(seqs_path, os_flags, os_modes), 'w') as fout:
        fout.write(final_seqs)


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        max_device_memory="29GB",
                        device_id=arguments.device_id)
    fold_design(arguments)

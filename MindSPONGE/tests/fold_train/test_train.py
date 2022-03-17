# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""train"""

import argparse
import os
import time
import numpy as np
import mindspore.communication.management as D
import mindspore.context as context
import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore import load_checkpoint
from config import config, global_config
from data.tools.get_train_data import create_dataset
from model import AlphaFold
from src.af_wrapcell import TrainOneStepCell, WithLossCell

set_seed(1)

parser = argparse.ArgumentParser(description='AlphaFold')
parser.add_argument('--pdb_data_dir', type=str, default=None, help='Location of training pdb file.')
parser.add_argument('--raw_feature_dir', type=str, default=None, help='Location of raw inputs file.')
parser.add_argument('--resolution_data', type=str, default=None, help='Location of resolution data file.')
parser.add_argument('--ckpt_url', type=str, default=None, help='ckpt url')
parser.add_argument('--seq_len', type=int, default=256, help='seq len')
parser.add_argument('--extra_msa_length', type=int, default=1024, help='extra msa length')
parser.add_argument('--max_msa_clusters', type=int, default=128, help='max msa clusters')
parser.add_argument('--extra_msa_num', type=int, default=4, help='extra msa number')
parser.add_argument('--evo_num', type=int, default=48, help='evo number')
parser.add_argument('--total_steps', type=int, default=9600000, help='total steps')
parser.add_argument('--loss_scale', type=float, default=1024.0, help='loss scale')
parser.add_argument('--gradient_clip', type=float, default=0.1, help='gradient clip value')
parser.add_argument('--train', type=bool, default=False, help='train or eval mode')
parser.add_argument('--run_distribute', type=bool, default=False, help='run distribute')
args_opt = parser.parse_args()

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", variable_memory_max_size="29GB",
                        device_id=int(os.getenv('DEVICE_ID')))
    if args_opt.run_distribute:
        D.init()
        device_num = D.get_group_size()
        os.environ['HCCL_CONNECT_TIMEOUT'] = "7200"
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num, parameter_broadcast=False)

    SEQ_LENGTH = args_opt.seq_len
    EXTRA_MSA_LENGTH = args_opt.extra_msa_length
    MAX_MSA_CLUSTERS = args_opt.max_msa_clusters
    model_name = "model_1"

    model_config = config.model_config(model_name)
    global_config = global_config.global_config(SEQ_LENGTH)
    model_config.resolution = 1

    num_recycle = model_config.model.num_recycle
    model_config.data.eval.crop_size = SEQ_LENGTH
    model_config.data.common.max_extra_msa = EXTRA_MSA_LENGTH
    msa_channel = model_config.model.embeddings_and_evoformer.msa_channel
    pair_channel = model_config.model.embeddings_and_evoformer.pair_channel
    model_config.data.eval.max_msa_clusters = MAX_MSA_CLUSTERS

    fold_net = AlphaFold(model_config, global_config, args_opt.extra_msa_num, args_opt.evo_num)
    net_with_criterion = WithLossCell(fold_net)

    opt = nn.Adam(params=fold_net.trainable_params(), learning_rate=Tensor(1e-4), eps=1e-6)

    train_net = TrainOneStepCell(net_with_criterion, opt, sens=args_opt.loss_scale,
                                 gradient_clip_value=args_opt.gradient_clip)
    if args_opt.ckpt_url:
        load_checkpoint(args_opt.ckpt_url, train_net)
    train_net.set_train(args_opt.train)
    step = 0
    start_time = time.time()
    np.random.seed(1)
    max_recycles = [int(np.random.uniform(size=1, low=0, high=4)) for _ in range(args_opt.total_steps)]
    max_recycles[step] = 0
    np.random.seed()
    names = os.listdir(args_opt.pdb_data_dir)
    names_list = []
    for name in names:
        names_list.append(name.split(".pdb")[0])
    train_dataset = create_dataset(args_opt.pdb_data_dir, args_opt.raw_feature_dir, names_list, model_config,
                                   args_opt.resolution_data, num_parallel_worker=4, is_parallel=args_opt.run_distribute,
                                   shuffle=True)
    dataset_iter = train_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    for d in dataset_iter:
        t1 = time.time()
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
                       d["true_msa"][max_recycle], d["bert_mask"][max_recycle], d["residx_atom14_to_atom37"], \
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

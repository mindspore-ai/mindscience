# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""megafold"""

# pylint: disable=W0221

import time
import os
import numpy as np

from mindspore.common import mutable
from mindspore import Tensor
import mindspore.communication.management as D
from mindspore import jit, context, nn
from mindspore.context import ParallelMode

from mindsponge.common.protein import to_pdb, from_prediction
from .module.fold_wrapcell import TrainOneStepCell, WithLossCell
from .module.lr import cos_decay_lr
from .nn_arch import Megafold, compute_confidence
from ..model import Model


class MEGAFold(Model):
    "MEGAFold"
    name = "MEGAFold"
    feature_list = ['target_feat', 'msa_feat', 'msa_mask', 'seq_mask', 'aatype',
                    'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions',
                    'template_mask', 'template_pseudo_beta_mask', 'template_pseudo_beta',
                    'extra_msa', 'extra_has_deletion', 'extra_deletion_value', 'extra_msa_mask',
                    'residx_atom37_to_atom14', 'atom37_atom_exists', 'residue_index',
                    'prev_pos', 'prev_msa_first_row', 'prev_pair']

    label_list = ["pseudo_beta", "pseudo_beta_mask", "all_atom_mask", "true_msa",
                  "bert_mask", "residx_atom14_to_atom37", "restype_atom14_bond_lower_bound",
                  "restype_atom14_bond_upper_bound", "atomtype_radius",
                  "backbone_affine_tensor", "backbone_affine_mask", "atom14_gt_positions",
                  "atom14_alt_gt_positions", "atom14_atom_is_ambiguous", "atom14_gt_exists",
                  "atom14_atom_exists", "atom14_alt_gt_exists", "all_atom_positions",
                  "rigidgroups_gt_frames", "rigidgroups_gt_exists", "rigidgroups_alt_gt_frames",
                  "torsion_angles_sin_cos", "use_clamped_fape", "filter_by_solution", "chi_mask"]

    def __init__(self, config):
        self.checkpoint_url =\
            'https://download.mindspore.cn/mindscience/mindsponge/MEGAFold/checkpoint/MEGA_Fold_1.ckpt'

        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
            context.set_context(graph_kernel_flags="--disable_expand_ops=Softmax \
                                --disable_cluster_ops=ReduceSum --composite_op_limit_size=50",
                                enable_graph_kernel=True)
        else:
            self.mixed_precision = True

        self.config = config
        self.use_jit = self.config.use_jit
        self.network = Megafold(self.config, self.mixed_precision)

        super().__init__(self.checkpoint_url, self.network, self.name)
        self.checkpoint_path = "./MEGA_Fold_1.ckpt"


        if self.config.is_training:
            if config.train.is_parallel:
                D.init()
                device_num = D.get_group_size()
                os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True,
                                                  device_num=device_num,
                                                  parameter_broadcast=False)
            train_config = self.config.train
            lr = cos_decay_lr(start_step=train_config.start_step, lr_init=0.0,
                              lr_min=train_config.lr_min, lr_max=train_config.lr_max,
                              decay_steps=train_config.total_steps,
                              warmup_steps=train_config.warmup_steps)
            opt = nn.Adam(params=self.network.trainable_params(), learning_rate=lr, eps=1e-6)

            net_with_criterion = WithLossCell(self.network, self.config)
            self.train_net = TrainOneStepCell(net_with_criterion, opt, sens=train_config.loss_scale,
                                              gradient_clip_value=train_config.gradient_clip)
            self.train_net.set_train(False)
            self.step = train_config.start_step


    def forward(self, data):
        "forward"
        feat = []
        for key in self.feature_list:
            feat.append(data[key])
        if self.use_jit:
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits \
                = self._jit_forward(feat)
        else:
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits \
                = self._pynative_forward(feat)

        res = prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits
        return res


    def predict(self, data, **kwargs):
        "predict"
        num_recycle = self.config.data.num_recycle
        num_residues = data["num_residues"]
        recycle_feature_name = self.feature_list[:-3]
        prev_pos = Tensor(data['prev_pos'])
        prev_msa_first_row = Tensor(data['prev_msa_first_row'])
        prev_pair = Tensor(data['prev_pair'])
        for recycle in range(num_recycle):
            data_iter = {}
            for key in recycle_feature_name:
                data_iter[key] = Tensor(data[key][recycle])
            data_iter['prev_pos'] = prev_pos
            data_iter['prev_msa_first_row'] = prev_msa_first_row
            data_iter['prev_pair'] = prev_pair
            data_iter = mutable(data_iter)
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = self.forward(data_iter)
        final_atom_positions = prev_pos.asnumpy()[:num_residues]
        final_atom_mask = data_iter['atom37_atom_exists'].asnumpy()[:num_residues]
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:num_residues]
        confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)
        b_factors = plddt[:, None] * final_atom_mask

        unrelaxed_protein = from_prediction(final_atom_positions,
                                            final_atom_mask,
                                            data["aatype"][0][:num_residues],
                                            data["residue_index"][0][:num_residues],
                                            b_factors)
        pdb_file = to_pdb(unrelaxed_protein)

        print("Infer finished, confidence is ", round(confidence, 2))
        res = final_atom_positions, final_atom_mask, confidence, pdb_file
        return res


    def loss(self, data):
        "loss"


    def grad_operations(self, gradient):
        "grad_operations"


    @jit
    def backward(self, data):
        loss = self.train_net(*data)
        return loss


    def train_step(self, data):
        num_recycle = np.random.randint(low=1, high=5)
        self.train_net.add_flags_recursive(train_backward=False)
        self.train_net.phase = 'train_forward'
        recycle_feature_name = self.feature_list[:-3]

        prev_pos = Tensor(data['prev_pos'])
        prev_msa_first_row = Tensor(data['prev_msa_first_row'])
        prev_pair = Tensor(data['prev_pair'])

        for recycle in range(num_recycle - 1):
            inputs = {}
            for key in recycle_feature_name:
                inputs[key] = Tensor(data[key][recycle])
            inputs['prev_pos'] = prev_pos
            inputs['prev_msa_first_row'] = prev_msa_first_row
            inputs['prev_pair'] = prev_pair
            inputs = mutable(inputs)
            t1 = time.time()
            prev_pos, prev_msa_first_row, prev_pair, _ = self.forward(inputs)
            t2 = time.time()
            print("forward time : ", round(t2 - t1, 2))
        inputs = {}
        for key in self.feature_list[:-3]:
            inputs[key] = Tensor(data[key][num_recycle - 1])
        inputs['prev_pos'] = prev_pos
        inputs['prev_msa_first_row'] = prev_msa_first_row
        inputs['prev_pair'] = prev_pair
        for key in self.label_list:
            inputs[key] = Tensor(data[key])
        self.train_net.add_flags_recursive(train_backward=True)
        self.train_net.phase = 'train_backward'
        keys = self.feature_list + self.label_list
        feat = []
        for key in keys:
            feat.append(inputs.get(key, Tensor([])))
        feat = mutable(feat)
        t1 = time.time()
        loss = self.backward(feat)
        t2 = time.time()
        print("backward time : ", round(t2 - t1, 2))
        loss_info = f"step is: {self.step}, total_loss: {loss[0]}, ",\
                    f"fape_sidechain_loss: {loss[1]}, fape_backbone_loss: {loss[2]}, ",\
                    f"angle_norm_loss: {loss[3]}, distogram_loss: {loss[4]}, ",\
                    f"masked_loss: {loss[5]}, plddt_loss: {loss[6]}"
        self.step += 1
        return loss_info


    @jit
    def _jit_forward(self, feat):
        res = self.network(*feat)
        return res

    def _pynative_forward(self, feat):
        res = self.network(*feat)
        return res

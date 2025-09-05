# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""megaassessment"""

import time
import os
import ssl
import urllib
import numpy as np
from mindspore import jit, context, nn, load_param_into_net, Tensor
from mindspore.common import mutable
from .module.assessment_wrapcell import TrainOneStepCell, WithLossCell
from .nn_arch import CombineModel as megaassessment
from .nn_arch import load_weights
from ..model import Model


class MEGAAssessment(Model):
    """megaassessment model"""
    name = "MEGAssessment"
    feature_list = ['target_feat', 'msa_feat', 'msa_mask', 'seq_mask', 'aatype',
                    'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions',
                    'template_mask', 'template_pseudo_beta_mask', 'template_pseudo_beta', 'extra_msa',
                    'extra_has_deletion', 'extra_deletion_value', 'extra_msa_mask',
                    'residx_atom37_to_atom14', 'atom37_atom_exists', 'residue_index',
                    'prev_pos', 'prev_msa_first_row', 'prev_pair', 'decoy_atom_positions', 'decoy_atom_mask']

    label_list = ["pseudo_beta", "pseudo_beta_mask", "all_atom_mask", "true_msa", "bert_mask",
                  "residx_atom14_to_atom37", "restype_atom14_bond_lower_bound", "restype_atom14_bond_upper_bound",
                  "atomtype_radius", "backbone_affine_tensor", "backbone_affine_mask", "atom14_gt_positions",
                  "atom14_alt_gt_positions", "atom14_atom_is_ambiguous", "atom14_gt_exists", "atom14_atom_exists",
                  "atom14_alt_gt_exists", "all_atom_positions", "rigidgroups_gt_frames", "rigidgroups_gt_exists",
                  "rigidgroups_alt_gt_frames", "torsion_angles_sin_cos", "chi_mask"]

    def __init__(self, config):
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
            context.set_context(graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                                   "--composite_op_limit_size=50", enable_graph_kernel=True)
        else:
            self.mixed_precision = True

        self.config = config
        self.use_jit = self.config.use_jit
        self.use_jit = True
        self.network = megaassessment(self.config, self.mixed_precision)
        if self.config.is_training:
            self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/' \
                                  'MEGAFold/checkpoint/MEGA_Fold_1.ckpt'
            self.checkpoint_path = "./MEGA_Fold_1.ckpt"
            if not os.path.exists(self.checkpoint_path):
                print("Download checkpoint to ", self.checkpoint_path)
                # pylint: disable=protected-access
                ssl._create_default_https_context = ssl._create_unverified_context
                urllib.request.urlretrieve(self.checkpoint_url, self.checkpoint_path)
            param_dict = load_weights(self.checkpoint_path, self.config.model)
            load_param_into_net(self.network, param_dict)
        else:
            self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/' \
                                  'MEGAAssessment/checkpoint/MEGA_Assessment.ckpt'
            self.checkpoint_path = "./MEGA_Assessment.ckpt"
        net_with_criterion = WithLossCell(self.network, self.config)
        lr = 0.0001
        opt = nn.Adam(params=self.network.trainable_params(), learning_rate=lr, eps=1e-6)
        self.train_net = TrainOneStepCell(net_with_criterion, opt, sens=1, gradient_clip_value=0.1)
        super().__init__(self.checkpoint_url, self.network, self.name)

    # pylint: disable=arguments-differ
    def forward(self, data, run_pretrain=True):
        """forward"""
        if self.use_jit:
            outputs = self._jit_forward(data, run_pretrain=run_pretrain)
        else:
            outputs = self._pynative_forward(data, run_pretrain=run_pretrain)
        return outputs

    # pylint: disable=arguments-differ
    def predict(self, inputs):
        """predict"""
        recycle_feature_name = self.feature_list[:-5]
        prev_pos = Tensor(inputs['prev_pos'])
        prev_msa_first_row = Tensor(inputs['prev_msa_first_row'])
        prev_pair = Tensor(inputs['prev_pair'])
        data = {}
        for recycle in range(4):
            for key in recycle_feature_name:
                data[key] = Tensor(inputs[key][recycle])
            data['prev_pos'] = prev_pos
            data['prev_msa_first_row'] = prev_msa_first_row
            data['prev_pair'] = prev_pair
            data = mutable(data)
            t1 = time.time()
            prev_pos, prev_msa_first_row, prev_pair, _ = self.forward(data, run_pretrain=True)
            t2 = time.time()
            print(round(t2 - t1, 2))
        data['prev_pos'] = prev_pos
        data['prev_msa_first_row'] = prev_msa_first_row
        data['prev_pair'] = prev_pair
        data['decoy_atom_positions'] = Tensor(inputs['decoy_atom_positions'])
        data['decoy_atom_mask'] = Tensor(inputs['decoy_atom_mask'])

        plddt = self.forward(data, run_pretrain=False)
        plddt = plddt.asnumpy()[inputs['align_mask'] == 1]
        return plddt

    # pylint: disable=arguments-differ
    @jit
    def backward(self, feat):
        """backward"""
        loss = self.train_net(*feat)
        return loss

    # pylint: disable=arguments-differ
    def train_step(self, data):
        """train one step"""
        num_recycle = np.random.randint(low=1, high=5)
        self.train_net.add_flags_recursive(train_backward=False)
        self.train_net.phase = 'train_forward'
        recycle_feature_name = self.feature_list[:-5]
        prev_pos = Tensor(data['prev_pos'])
        prev_msa_first_row = Tensor(data['prev_msa_first_row'])
        prev_pair = Tensor(data['prev_pair'])
        for recycle in range(4):
            inputs = {}
            for key in recycle_feature_name:
                inputs[key] = Tensor(data[key][recycle])
            inputs['prev_pos'] = prev_pos
            inputs['prev_msa_first_row'] = prev_msa_first_row
            inputs['prev_pair'] = prev_pair
            inputs = mutable(inputs)
            t1 = time.time()
            prev_pos, prev_msa_first_row, prev_pair, _ = self.forward(inputs, run_pretrain=True)
            if recycle == num_recycle:
                final_atom_positions_recycle = prev_pos
            t2 = time.time()
            print("forward time : ", round(t2 - t1, 2))
        inputs = {}
        for key in self.feature_list[:-5]:
            inputs[key] = Tensor(data[key][num_recycle - 1])
        inputs['prev_pos'] = prev_pos
        inputs['prev_msa_first_row'] = prev_msa_first_row
        inputs['prev_pair'] = prev_pair
        for key in self.label_list:
            inputs[key] = Tensor(data[key])
        self.train_net.add_flags_recursive(train_backward=True)
        self.train_net.phase = 'train_backward'
        keys = self.feature_list[:-2] + self.label_list
        feat = []
        for key in keys:
            feat.append(inputs.get(key))
        feat.append(final_atom_positions_recycle)
        feat.append(inputs.get('atom37_atom_exists'))
        feat = mutable(feat)
        t1 = time.time()
        loss = self.backward(feat)
        t2 = time.time()
        print("backward time : ", round(t2 - t1, 2))
        return loss

    # pylint: disable=arguments-differ
    @jit
    def _jit_forward(self, data, run_pretrain=True):
        """forward with jit mode"""
        feat = []
        feature_list = self.feature_list
        if run_pretrain:
            feature_list = self.feature_list[:-2]
        for key in feature_list:
            feat.append(data[key])
        outputs = self.network(*feat, run_pretrain=run_pretrain)
        return outputs

    # pylint: disable=arguments-differ
    def _pynative_forward(self, data, run_pretrain=True):
        """forward with pynative mode"""
        feat = []
        feature_list = self.feature_list
        if run_pretrain:
            feature_list = self.feature_list[:-2]
        for key in feature_list:
            feat.append(data[key])
        outputs = self.network(*feat, run_pretrain=run_pretrain)
        return outputs

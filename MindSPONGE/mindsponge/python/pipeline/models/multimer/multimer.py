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
"""multimer"""
import time
from mindspore import jit, context, nn
from mindspore.common import mutable
from mindspore import Tensor
from .nn_arch import MultimerArch, compute_confidence
from ..model import Model


class Multimer(Model):
    """Multimer"""
    name = "Multimer"
    feature_list = ['aatype', 'residue_index', 'template_aatype', 'template_all_atom_mask',
                    'template_all_atom_positions', 'asym_id', 'sym_id', 'entity_id', 'seq_mask', 'msa_mask',
                    'target_feat', 'msa_feat', 'extra_msa', 'extra_deletion_matrix', 'extra_msa_mask',
                    'residx_atom37_to_atom14', 'atom37_atom_exists',
                    'prev_pos', 'prev_msa_first_row', 'prev_pair']

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
        self.white_list = (nn.Softmax, nn.LayerNorm)
        self.checkpoint_url = \
            'https://download.mindspore.cn/mindscience/mindsponge/Multimer/checkpoint/Multimer_Model_1.ckpt'
        self.checkpoint_path = "./Multimer_Model_1.ckpt"
        self.network = MultimerArch(self.config, self.mixed_precision)
        super().__init__(self.checkpoint_url, self.network, self.name, self.white_list)

    def forward(self, data):
        if self.use_jit:
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = self._jit_forward(data)
        else:
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = self._pynative_forward(data)
        return prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits

    # pylint: disable=arguments-differ
    def predict(self, inputs, num_recycle=4):
        num_residues = inputs["num_residues"]
        recycle_feature_name = self.feature_list[:-3]
        prev_pos = Tensor(inputs['prev_pos'])
        prev_msa_first_row = Tensor(inputs['prev_msa_first_row'])
        prev_pair = Tensor(inputs['prev_pair'])
        for recycle in range(num_recycle):
            data = {}
            for key in recycle_feature_name:
                data[key] = Tensor(inputs[key][recycle])
            data['prev_pos'] = prev_pos
            data['prev_msa_first_row'] = prev_msa_first_row
            data['prev_pair'] = prev_pair
            data = mutable(data)
            t1 = time.time()
            prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = self.forward(data)
            t2 = time.time()
            print(round(t2 - t1, 2))
        final_atom_positions = prev_pos.asnumpy()[:num_residues]
        final_atom_mask = data['atom37_atom_exists'].asnumpy()[:num_residues]
        predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:num_residues]
        confidence, plddt = compute_confidence(predicted_lddt_logits, return_lddt=True)
        b_factors = plddt[:, None] * final_atom_mask
        return final_atom_positions, final_atom_mask, confidence, b_factors

    def loss(self, data):
        pass

    def grad_operations(self, gradient):
        pass

    @jit
    def backward(self, data):
        pass

    def train_step(self, data):
        pass

    @jit
    def _jit_forward(self, data):
        feat = []
        for key in self.feature_list:
            feat.append(data[key])
        prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = self.network(*feat)
        return prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits

    def _pynative_forward(self, data):
        feat = []
        for key in self.feature_list:
            feat.append(data[key])
        prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = self.network(*feat)
        return prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits

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
"""evogen"""
import numpy as np
from mindspore import Tensor, context, jit
from mindspore.common import mutable

from mindsponge.data_transform import make_atom14_masks, one_hot
from ..model import Model
from .nn_arch import MegaEvogen


class MEGAEvoGen(Model):
    '''MEGAEvoGen'''
    feature_list = ['seq_mask', 'msa_mask', 'msa_input', 'query_input', 'additional_input', 'evogen_random_data',
                    'evogen_context_mask']

    def __init__(self, config):
        self.name = "MEGAEvoGen"
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
            context.set_context(graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                                   "--composite_op_limit_size=50", enable_graph_kernel=True)
        else:
            self.mixed_precision = True

        self.config = config
        self.use_jit = self.config.use_jit
        self.checkpoint_url = \
            'https://download.mindspore.cn/mindscience/mindsponge/MEGAEvoGen/checkpoint/MEGAEvoGen.ckpt'
        self.checkpoint_path = "./MEGAEvoGen.ckpt"
        self.network = MegaEvogen(self.config, self.mixed_precision)
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name,
                         mixed_precision=self.mixed_precision)

    # pylint: disable=invalid-name
    def forward(self, data):
        feat = []
        for key in self.feature_list:
            feat.append(data[key])
        feat = mutable(feat)
        if self.use_jit:
            reconstruct_msa = self._jit_forward(feat)
        else:
            reconstruct_msa = self._pynative_forward(feat)
        return reconstruct_msa

    # pylint: disable=arguments-differ
    @jit
    def backward(self, feat):
        pass

    # pylint: disable=arguments-differ
    def predict(self, data):
        if not self.config.use_pkl:
            new_data, inputs = data
        else:
            inputs = data
        for key in inputs:
            inputs[key] = Tensor(inputs[key])
        inputs = mutable(inputs)
        reconstruct_msa, reconstruct_msa_mask = self.forward(inputs)
        if not self.config.use_pkl:
            feature = {}
            aatype = new_data.get("aatype")
            feature["num_residues"] = np.array(aatype.shape[0], dtype=np.int32)
            aatype = np.pad(aatype, (0, self.config.crop_size-aatype.shape[0]), 'constant')
            aatype = np.expand_dims(aatype, 0)
            residue_index = new_data.get("residue_index")
            residue_index = np.pad(residue_index, (0, self.config.crop_size-residue_index.shape[0]), 'constant')
            residue_index = np.expand_dims(residue_index, 0)
            between_segment_residues = np.zeros((1, aatype.shape[1]), dtype=np.int32)
            has_break = np.clip(between_segment_residues.astype(np.float32), np.array(0), np.array(1))
            aatype_1hot = one_hot(21, aatype)
            target_feat = [np.expand_dims(has_break, axis=-1), aatype_1hot]
            feature["target_feat"] = np.concatenate(target_feat, axis=-1).astype(np.float32)
            feature["msa_feat"] = reconstruct_msa.unsqueeze(0).asnumpy()
            feature["msa_mask"] = reconstruct_msa_mask.unsqueeze(0).asnumpy()
            feature["seq_mask"] = np.expand_dims(inputs.get("seq_mask").asnumpy(), axis=0)
            feature["aatype"] = aatype.astype(np.int32)
            feature["template_aatype"] = np.zeros((1, 4, self.config.crop_size), dtype=np.int32)
            feature["template_all_atom_masks"] = np.zeros((1, 4, self.config.crop_size, 37), dtype=np.float32)
            feature["template_all_atom_positions"] = np.zeros((1, 4, self.config.crop_size, 37, 3), dtype=np.float32)
            feature["template_mask"] = np.zeros((1, 4), dtype=np.float32)
            feature["template_pseudo_beta_mask"] = np.zeros((1, 4, self.config.crop_size), dtype=np.float32)
            feature["template_pseudo_beta"] = np.zeros((1, 4, self.config.crop_size, 3), dtype=np.float32)
            extra_msa_length = 512
            feature["extra_msa"] = np.zeros((1, extra_msa_length, self.config.crop_size), dtype=np.int32)
            feature["extra_has_deletion"] = np.zeros((1, extra_msa_length, self.config.crop_size), dtype=np.float32)
            feature["extra_deletion_value"] = np.zeros((1, extra_msa_length, self.config.crop_size), dtype=np.float32)
            feature["extra_msa_mask"] = np.zeros((1, extra_msa_length, self.config.crop_size), dtype=np.float32)
            _, _, residx_atom37_to_atom14, atom37_atom_exists = make_atom14_masks(aatype)
            feature["residx_atom37_to_atom14"] = residx_atom37_to_atom14
            feature["atom37_atom_exists"] = atom37_atom_exists
            feature["residue_index"] = residue_index

            # pylint: disable=consider-iterating-dictionary
            for k in feature.keys():
                if k == "num_residues":
                    continue
                feature[k] = np.broadcast_to(feature.get(k), (4,) + feature.get(k).shape[1:])

            feature["prev_pos"] = np.zeros((aatype.shape[1], 37, 3)).astype(np.float32)
            feature["prev_msa_first_row"] = np.zeros((aatype.shape[1], 256)).astype(np.float32)
            feature["prev_pair"] = np.zeros((aatype.shape[1], aatype.shape[1], 128)).astype(np.float32)
            return feature
        return reconstruct_msa, reconstruct_msa_mask

    def loss(self, data):
        pass

    def grad_operations(self, gradient):
        pass

    def train_step(self, data):
        pass

    @jit
    def _jit_forward(self, feat):
        reconstruct_msa = self.network(*feat)
        return reconstruct_msa

    def _pynative_forward(self, feat):
        reconstruct_msa = self.network(*feat)
        return reconstruct_msa

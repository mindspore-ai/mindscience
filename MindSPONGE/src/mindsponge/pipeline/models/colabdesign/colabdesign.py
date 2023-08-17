# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""colabdesign"""
import numpy as np

import mindspore as ms
from mindspore import Parameter
from mindspore import Tensor
from mindspore import jit, context
from mindspore.common import mutable
from mindsponge.cell.mask import LayerNormProcess

from .module.design_wrapcell import TrainOneStepCell, WithLossCell
from .module.utils import get_weights, get_lr, get_opt, get_seqs
from .nn_arch import Colabdesign
from ..model import Model


class COLABDESIGN(Model):
    """ColabDesign"""
    name = "COLABDESIGN"
    feature_list = ["msa_feat", "msa_mask", "seq_mask_batch", \
                    "template_aatype", "template_all_atom_masks", "template_all_atom_positions", "template_mask", \
                    "template_pseudo_beta_mask", "template_pseudo_beta", \
                    "extra_msa", "extra_has_deletion", "extra_deletion_value", "extra_msa_mask", \
                    "residx_atom37_to_atom14", "atom37_atom_exists_batch", \
                    "residue_index_batch", "batch_aatype", "batch_all_atom_positions", "batch_all_atom_mask",
                    "opt_temp", \
                    "opt_soft", "opt_hard", "prev_pos", "prev_msa_first_row", "prev_pair"]

    def __init__(self, config):
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        self.fp32_white_list = None
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
            context.set_context(graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                                   "--composite_op_limit_size=50", enable_graph_kernel=True)
        else:
            self.mixed_precision = True
            self.fp32_white_list = (ms.nn.Softmax, ms.nn.LayerNorm, LayerNormProcess)

        self.config = config
        self.use_jit = self.config.use_jit
        self.checkpoint_url = \
            'https://download.mindspore.cn/mindscience/mindsponge/ColabDesign/checkpoint/ColabDesign.ckpt'
        self.checkpoint_path = "./colabdesign.ckpt"
        seq_vector = 0.01 * np.random.normal(0, 1, size=(1, 100, 20))
        self.network = Colabdesign(self.config, self.mixed_precision, Tensor(seq_vector, ms.float32), 100,
                                   protocol=self.config.protocol)
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name,
                         white_list=self.fp32_white_list, mixed_precision=self.mixed_precision)
        net_with_criterion = WithLossCell(self.network)
        soft_weights, _, temp_weights = get_weights(self.config, self.config.soft_iters, self.config.temp_iters,
                                                    self.config.hard_iters)
        epoch = self.config.soft_iters + self.config.temp_iters + self.config.hard_iters
        lr = get_lr(temp_weights, soft_weights, epoch)
        model_params = [Parameter(Tensor(seq_vector, ms.float32), name="seq_vector", requires_grad=True)]
        opt = get_opt(model_params, lr, 0.0, self.config.opt_choice)
        self.train_net = TrainOneStepCell(net_with_criterion, opt, sens=8192)

    # pylint: disable=arguments-differ
    def predict(self, data):
        temp, soft, hard = get_weights(self.config, self.config.soft_iters, self.config.temp_iters,
                                       self.config.hard_iters)
        best = 999
        for epoch in range(2):
            temp_step = temp[epoch]
            soft_step = soft[epoch]
            hard_step = hard[epoch]
            data[-6] = temp_step
            data[-5] = soft_step
            data[-4] = hard_step
            inputs_feats = [Tensor(feat) for feat in data]
            inputs_feats = mutable(inputs_feats)
            self.train_net.add_flags_recursive(save_best=False)
            self.train_net.phase = 'save_best'
            loss = self._jit_forward(inputs_feats)
            if loss < best:
                self.train_net.add_flags_recursive(save_best=True)
                self.train_net.phase = 'save_best'
                best = loss
                inputs_feats = [Tensor(feat) for feat in data]
                final_seq = self._jit_forward(inputs_feats)
        final_seqs = get_seqs(final_seq.asnumpy())
        print("fina_seqs", str(final_seqs))
        return best

    def forward(self, data):
        pass

    # pylint: disable=arguments-differ
    @jit
    def backward(self, feat):
        pass

    # pylint: disable=arguments-differ
    def train_step(self, data):
        pass

    def _pynative_forward(self, data):
        pass

    @jit
    def _jit_forward(self, data):
        loss = self.train_net(*data)
        return loss

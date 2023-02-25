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
"""colabdesign"""
import numpy as np

from mindspore import Parameter
from mindspore import Tensor, load_checkpoint
import mindspore as ms
from mindspore import jit, context

from .nn_arch import Colabdesign
from ..model import Model
from .module.design_wrapcell import TrainOneStepCell, WithLossCell
from .module.utils import get_weights, get_lr, get_opt


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
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
            context.set_context(graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                                   "--composite_op_limit_size=50", enable_graph_kernel=True)
        else:
            self.mixed_precision = True

        self.config = config
        self.use_jit = self.config.use_jit
        self.checkpoint_url = \
            'https://download.mindspore.cn/mindscience/mindsponge/Multimer/checkpoint/Multimer_Model_1.ckpt'
        self.checkpoint_path = "./colabdesign.ckpt"
        seq_vector = 0.01 * np.random.normal(0, 1, size=(1, 100, 20))
        self.network = Colabdesign(self.config, self.mixed_precision, Tensor(seq_vector, ms.float16), 100,
                                   protocol=self.config.protocol)
        load_checkpoint(self.checkpoint_path, self.network)
        net_with_criterion = WithLossCell(self.network)
        soft_weights, temp_weights = get_weights(self.config, self.config.soft_iters, self.config.temp_iters,
                                                 self.config.hard_iters)
        epoch = self.config.soft_iters + self.config.temp_iters + self.config.hard_iters
        lr = get_lr(temp_weights, soft_weights, epoch)
        model_params = [Parameter(Tensor(seq_vector, ms.float16))]
        opt = get_opt(model_params, lr, 0.0, self.config.opt_choice)
        self.train_net = TrainOneStepCell(net_with_criterion, opt, sens=8192)
        super().__init__(self.checkpoint_url, self.network, self.name)

    # pylint: disable=arguments-differ
    def predict(self, data):
        pass

    def forward(self, data):
        pass

    # pylint: disable=arguments-differ
    @jit
    def backward(self, feat):
        loss = self.train_net(*feat)
        return loss

    # pylint: disable=arguments-differ
    def train_step(self, data):
        features = []
        for feature in data:
            features.append(Tensor(data[feature]))

        loss = self.backward(features)

        return loss

    def _pynative_forward(self, data):
        pass

    @jit
    def _jit_forward(self, data):
        pass

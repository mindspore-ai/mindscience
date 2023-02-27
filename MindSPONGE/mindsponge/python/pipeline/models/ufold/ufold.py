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
"""ufold"""
import mindspore as ms
from mindspore import jit, nn
from mindspore import ops
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell
from mindspore.common import dtype as mstype
from mindspore.common import mutable

from ..model import Model
from .nn_arch import Unet as FCNNet

sign = ops.Sign()


def evaluate_exact_new(pred_a, true_a, eps=1e-11):
    """get pred, recall and f1_score"""
    tp_map = sign(ms.Tensor(pred_a) * ms.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = sign(ms.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps)/(tp + fn + eps)
    precision = (tp + eps)/(tp + fp + eps)
    f1_score_ms = (2 * tp + eps)/(2 * tp + fp + fn + eps)
    return precision, recall, f1_score_ms


class MyWithLossCell(nn.Cell):
    def __init__(self, network, loss_fn):
        super(MyWithLossCell, self).__init__()
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, x, y, label):
        out = self.network(x)
        return self.loss_fn(out*y, label)


class UFold(Model):
    """UFold"""
    def __init__(self, config):
        self.config = config
        self.use_jit = self.config.use_jit

        self.dataset_ckpt_name = {
            'ArchiveII': 'ufold_train',
            'bpnew': 'ufold_train',
            'TS0': 'ufold_train',
            'TS1': 'ufold_train_pdbfinetune',
            'TS2': 'ufold_train_pdbfinetune',
            'TS3': 'ufold_train_pdbfinetune',
            'All': 'ufold_train_99',
        }

        self.checkpoint_urls = {
            'ufold_train': 'https://download.mindspore.cn/mindscience/mindsponge/ufold/checkpoint/ufold_train.ckpt',
            'ufold_train_pdbfinetune':
                'https://download.mindspore.cn/mindscience/mindsponge/ufold/checkpoint/ufold_train_pdbfinetune.ckpt',
            'ufold_train_99':
                'https://download.mindspore.cn/mindscience/mindsponge/ufold/checkpoint/ufold_train_99.ckpt'
        }

        self.ckpt_name = self.dataset_ckpt_name.get(self.config.test_ckpt)
        self.checkpoint_url = self.checkpoint_urls.get(self.ckpt_name)
        self.checkpoint_path = "./" + self.ckpt_name + ".ckpt"
        self.result_no_train = []
        self.cast = ops.Cast()
        self.zeroslike = ops.ZerosLike()
        self.network = FCNNet(img_ch=17)
        self.pos_weight = ms.Tensor([300], mstype.float32)
        self.criterion_bce_weighted = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.u_optimizer = nn.Adam(params=self.network.trainable_params(), learning_rate=1e-4)
        self.loss_net = MyWithLossCell(self.network, self.criterion_bce_weighted)
        self.train_net = TrainOneStepCell(self.loss_net, self.u_optimizer)
        if self.config.is_training:
            self.train_net.set_train()
        super().__init__(self.checkpoint_url, self.network)


    def forward(self, data):
        if self.use_jit:
            mse = self._jit_forward(data)
        else:
            mse = self._pynative_forward(data)
        return mse

    # pylint: disable=arguments-differ
    def predict(self, data):
        _, seq_embeddings, _, _, _, _, _, _ = data.values()
        seq_embedding_batch = ms.Tensor(ops.Cast()(seq_embeddings, mstype.float32))
        pred_contacts = self.forward(seq_embedding_batch)
        return pred_contacts


    def loss(self, data):
        pass


    def grad_operations(self, gradient):
        pass


    @jit
    def backward(self, data):
        loss = self.train_net(*data)
        return loss


    def train_step(self, data):
        contacts, seq_embeddings, _, seq_lens, _, _ = data.values()
        contacts_batch = Tensor(ops.Cast()(contacts, mstype.float32))
        seq_embedding_batch = Tensor(ops.Cast()(seq_embeddings, mstype.float32))
        pred_contacts = self.network(seq_embedding_batch)
        contact_masks = ops.ZerosLike()(pred_contacts)
        contact_masks[:, :seq_lens, :seq_lens] = 1
        feat = [seq_embedding_batch, contact_masks, contacts_batch]
        feat = mutable(feat)
        loss = self.backward(feat)
        return loss


    @jit
    def _jit_forward(self, data):
        mse = self.network(data)
        return mse


    def _pynative_forward(self, data):
        mse = self.network(data)
        return mse

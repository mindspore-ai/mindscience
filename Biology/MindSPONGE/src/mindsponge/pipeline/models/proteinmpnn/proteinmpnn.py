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
"proteinmpnn"
import mindspore as ms
import mindspore.ops as ops
from mindspore import jit, context, nn
from mindspore import Tensor

from ..model import Model
from .nn_arch import ProteinMPNN
from .utils import loss_nll, ProcessLinspace, s_to_seq
from .proteinmpnn_wrapcell import CustomTrainOneStepCell, CustomWithLossCell, LossSmoothed, LRLIST
from .proteinmpnn_dataset import ProteinMpnnDataset


class ProteinMpnn(Model):
    """proteinmpnn"""
    name = "ProteinMpnn"

    def __init__(self, config):
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
        else:
            self.mixed_precision = True

        self.config = config
        self.use_jit = self.config.use_jit
        self.network = ProteinMPNN(self.config)
        self.white_list = ProcessLinspace
        self.dataset = ProteinMpnnDataset(self.config)
        self.checkpoint_url = ""
        self.checkpoint_path = ""
        if self.config.is_training:
            loss = LossSmoothed()
            net_with_loss = CustomWithLossCell(self.network, loss)
            lrlist = LRLIST(self.config.hidden_dim, 2, 4000)
            lr = lrlist.cal_lr(10 * 5436)
            optimizer = nn.Adam(self.network.trainable_params(), learning_rate=ms.Tensor(lr), beta1=0.9, beta2=0.98,
                                eps=1e-9)
            self.train_net = CustomTrainOneStepCell(net_with_loss, optimizer)
        else:
            self.checkpoint_url = \
                'https://download.mindspore.cn/mindscience/mindsponge/ProteinMPNN/checkpoint/proteinmpnn.ckpt'
            self.checkpoint_path = "./proteinmpnn.ckpt"
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name, self.white_list,
                         mixed_precision=self.mixed_precision)

    def forward(self, data):
        pass

    # pylint: disable=arguments-differ
    @jit
    def backward(self, feat):
        log_probs = self.train_net.network.backbone(*feat)
        return log_probs

    def train_step(self, data):
        data = self.dataset.process(data)
        log_probs = self.backward(data)
        loss, _, true_false = loss_nll(data[1], log_probs, data[-1])
        train_sum, train_weights = 0., 0.
        train_acc = 0.
        train_sum += ops.ReduceSum()(loss * data[-1]).asnumpy()
        train_acc += ops.ReduceSum()(true_false * data[-1]).asnumpy()
        train_weights += ops.ReduceSum()(data[-1]).asnumpy()
        train_loss = train_sum / train_weights
        return train_loss

    # pylint: disable=arguments-differ
    def predict(self, inputs):
        all_probs_list = []
        all_log_probs_list = []
        s_sample_list = []
        data = []
        for index, value in enumerate(inputs):
            if index == 3:
                data.append(ms.Tensor(inputs[index + 1] * value))
            elif index == 4:
                continue
            elif index == 8:
                break
            else:
                data.append(Tensor(value))

        if self.config.use_jit:
            net_work_use = self._jit_forward
        else:
            net_work_use = self._pynative_forward
        temperatures = [float(item) for item in "0.1".split()]
        mask_for_loss = inputs[2] * inputs[3] * inputs[4]
        for temp in temperatures:
            for _ in range(2):
                randn_2 = ops.StandardNormal()(inputs[3].shape).astype(ms.float32)
                sample_dict = self.network.sample(inputs[0], randn_2, inputs[1], inputs[3], inputs[6], inputs[5],
                                                  mask=inputs[2], temperature=temp, omit_aas_np=inputs[8],
                                                  bias_aas_np=inputs[9], chain_m_pos=inputs[4],
                                                  omit_aa_mask=inputs[10], pssm_coef=inputs[11],
                                                  pssm_bias=inputs[12], pssm_multi=0.0,
                                                  pssm_log_odds_flag=bool(0),
                                                  pssm_log_odds_mask=inputs[-1],
                                                  pssm_bias_flag=bool(0),
                                                  bias_by_res=inputs[-2])
                s_sample = sample_dict.get("s")
                data[1] = s_sample
                log_probs = net_work_use(data)
                mask_for_loss = inputs[2] * inputs[3] * inputs[4]
                all_probs_list.append(sample_dict.get("probs").asnumpy())
                all_log_probs_list.append(log_probs.asnumpy())
                s_sample_list.append(s_sample.asnumpy())
                s_sample = s_sample.astype(ms.int64)

            seq_recovery_rate = ops.ReduceSum()(ops.ReduceSum()(nn.OneHot(depth=21)(inputs[1][0]) \
                                                                * nn.OneHot(depth=21)(s_sample[0]), axis=-1) \
                                                * mask_for_loss[0]) / ops.ReduceSum()(mask_for_loss[0])
            native_seq = s_to_seq(inputs[1][0], inputs[3][0])
            print("final_seq", native_seq)

        return seq_recovery_rate

    @jit
    def _jit_forward(self, data):
        log_probs = self.network(*data)
        return log_probs

    def _pynative_forward(self, data):
        log_probs = self.network(*data)
        return log_probs

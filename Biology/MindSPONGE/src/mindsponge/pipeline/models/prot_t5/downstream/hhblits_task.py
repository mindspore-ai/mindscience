# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Pretrain property prediction task; hhblits dataset."""
from mindspore import nn
import mindspore.ops as ops

from .task_datasets import create_hhblits_dataset, map_label_to_category, \
            LABEL_MASKER, HHBLITS_D3_LABEL_TO_CATE, HHBLITS_D8_LABEL_TO_CATE
from .downstream_nets import TokensClassifier, EmbeddingTaskNet
from .downstream_task import BaseTask, lr_secheduler


class TokenLevelAccuracy:
    """Sequence token level classify task accuracy."""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.acc = nn.Accuracy('classification')
        self.acc.clear()

    def update(self, logits, labels):
        act_labels = labels.view(-1)
        act_logits = logits.view(-1, self.num_classes)
        valid_labels = act_labels[act_labels > -1]
        valid_logits = act_logits[act_labels > -1]

        self.acc.update(valid_logits, valid_labels)

    def get(self):
        return self.acc.eval()


class HHblitsTask(BaseTask):
    """Pretrain property prediction task of hhblits dataset. The network is a convolutional net and token level classifier."""
    def __init__(self, config):
        super().__init__(config)
        cnn_net = TokensClassifier()
        self.net = EmbeddingTaskNet(cnn_net, self.t5_config_path)

        if self.checkpoint_path:
            self.net.load_from_pretrained(self.checkpoint_path)

        if self.train_conf.train_data_path:
            self.train_dataset = create_hhblits_dataset(self.train_conf.train_data_path,
                                                        self.t5_tokenizer, self.train_conf.batch_size)
            batch_num = self.train_dataset.get_dataset_size()
            learning_rate = lr_secheduler(self.train_conf.lr, batch_num, self.train_conf.epochs)
            self.optimizer = nn.Adam(self.net.trainable_params(), learning_rate=learning_rate)
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=LABEL_MASKER)

        # eval
        if self.train_conf.eval_data_path:
            self.eval_dataset = create_hhblits_dataset(self.train_conf.eval_data_path,
                                                       self.t5_tokenizer, self.train_conf.batch_size)

    @staticmethod
    def __eval_fn(model_fn, dataset):
        """eval dataset accuracy."""
        metric_q3 = TokenLevelAccuracy(3)
        metric_q8 = TokenLevelAccuracy(8)

        for inputs, masks, d3labels, d8labels, _ in dataset:
            logits1, logits2, _ = model_fn(inputs, masks)
            metric_q3.update(logits1, d3labels)
            metric_q8.update(logits2, d8labels)

        dataset.reset()
        m3acc = metric_q3.get()
        m8acc = metric_q8.get()
        return m3acc, m8acc

    def eval_fn(self, dataset):
        """eval dataset"""
        return HHblitsTask.__eval_fn(self.net, dataset)

    def eval_acc(self, eval_data_path):
        """eval accuracy of data file"""
        eval_dataset = create_hhblits_dataset(eval_data_path, self.t5_tokenizer, self.train_conf.batch_size)
        m3acc, m8acc = self.eval_fn(eval_dataset)
        return m3acc, m8acc

    def token_level_crossentoryloss(self, logits, labels, num_classes, loss_fn):
        """token level crossentory loss"""
        activate_labels = labels.view(-1)
        activate_logits = logits.view(-1, num_classes)
        return loss_fn(activate_logits, activate_labels)

    def predict(self, data):
        """predict"""
        logits1, logits2, _ = self.net(*data)
        softmax = ops.Softmax(axis=-1)
        probabilities1 = softmax(logits1)
        probabilities2 = softmax(logits2)

        # get token index of predict max probabilities
        predicted_labels1 = ops.Argmax(axis=-1)(probabilities1)
        predicted_labels2 = ops.Argmax(axis=-1)(probabilities2)

        predicted_cates1 = map_label_to_category(predicted_labels1, HHBLITS_D3_LABEL_TO_CATE)
        predicted_cates2 = map_label_to_category(predicted_labels2, HHBLITS_D8_LABEL_TO_CATE)
        return predicted_cates1, predicted_cates2

    # pylint: disable=W0221
    def forward_fn(self, inputs, masks, d3labels, d8labels):
        """multitask loss"""
        logits1, logits2, _ = self.net(inputs, masks)
        loss1 = self.token_level_crossentoryloss(
            logits1, d3labels, 3, self.loss_fn)
        loss2 = self.token_level_crossentoryloss(
            logits2, d8labels, 8, self.loss_fn)
        return 0.5 * loss1 + 0.5 * loss2

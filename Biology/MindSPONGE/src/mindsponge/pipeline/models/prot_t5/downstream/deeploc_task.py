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
"""Pretrain property prediction task; deeploc dataset."""
from mindspore import nn
import mindspore.ops as ops

from .downstream_nets import MeanPoolingClassifier, EmbeddingTaskNet
from .downstream_task import BaseTask, lr_secheduler
from .task_datasets import create_deeploc_dataset, map_label_to_category, \
                    LOC_CATES, LOC_LABEL_TO_CATE, MEMBRANE_LABEL_TO_CATE


class DeeplocTask(BaseTask):
    """
        Pretrain property prediction task of deeploc dataset. The network is a mean pooling classifier of embeddings.
    """
    def __init__(self, config):
        """
            This method initializes the network, has train and eval interface.
            cate_name represents different tasks: "loc", "membrane"
        """
        super().__init__(config)
        self.cate_name = config.train.cate_name

        if self.cate_name == 'loc':
            self.label_to_cate = LOC_LABEL_TO_CATE
            num_classes = len(LOC_CATES)
        else:
            self.label_to_cate = MEMBRANE_LABEL_TO_CATE
            num_classes = 2

        # apply mean pooling in hidden states of prot T5 model encoder
        mpc_net = MeanPoolingClassifier(num_classes)
        self.net = EmbeddingTaskNet(mpc_net, self.t5_config_path)

        if self.checkpoint_path:
            self.net.load_from_pretrained(self.checkpoint_path)

        if self.train_conf.train_data_path:
            self.train_dataset = create_deeploc_dataset(self.train_conf.train_data_path, \
                            self.t5_tokenizer, batch_size=self.train_conf.batch_size, cate_name=self.cate_name)
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
            batch_num = self.train_dataset.get_dataset_size()
            learning_rate = lr_secheduler(self.train_conf.lr, batch_num, self.train_conf.epochs)
            self.optimizer = nn.Adam(self.net.trainable_params(), learning_rate=learning_rate)

        if self.train_conf.eval_data_path:
            self.eval_dataset = create_deeploc_dataset(self.train_conf.eval_data_path, \
                                self.t5_tokenizer, batch_size=self.train_conf.batch_size, cate_name=self.cate_name)

    @staticmethod
    def __eval_fn(model_fn, dataset):
        """eval give dataset with model; staticmethod"""
        metric = nn.Accuracy('classification')
        metric.clear()
        for inputs, masks, targets in dataset:
            logits = model_fn(inputs, masks)
            metric.update(logits, targets)

        accuracy = metric.eval()
        dataset.reset()
        return accuracy

    def eval_fn(self, dataset):
        """eval dataset"""
        return DeeplocTask.__eval_fn(self.net, dataset)

    # pylint: disable=W0221
    def forward_fn(self, inputs, masks, targets):
        """forward loss"""
        logits = self.net(inputs, masks)
        loss = self.loss_fn(logits, targets)
        return loss

    def eval_acc(self, eval_data_path):
        """eval accuracy data file"""
        eval_dataset = create_deeploc_dataset(eval_data_path, self.t5_tokenizer, \
                                batch_size=self.train_conf.batch_size, cate_name=self.cate_name)
        return self.eval_fn(eval_dataset)

    def predict(self, data):
        """predict"""
        logits = self.net(*data)
        softmax = ops.Softmax(axis=1)
        probabilities = softmax(logits)
        predicted_labels = ops.Argmax(axis=1)(probabilities)
        predicted_cates = map_label_to_category(predicted_labels, self.label_to_cate)
        return predicted_cates

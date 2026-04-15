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
"""Networks of downstream task."""
import logging

import mindspore as ms
from mindspore import nn, ops
from mindformers import T5ForConditionalGeneration


logger = logging.getLogger(__name__)
EMBEDIING_LENGTH = 1024
POOLING_CHANNELS = 32


class TokensClassifier(nn.Cell):
    """Acid token level predictor; using convolution net to convergence of local information."""
    def __init__(self):
        super().__init__()
        # CNN weights are trained on ProtT5 embeddings
        self.feature_extractor = nn.SequentialCell([
            nn.Conv2d(EMBEDIING_LENGTH, 32, kernel_size=(7, 1), pad_mode='pad', padding=(3, 3, 0, 0), has_bias=True),  # 7x32
            nn.ReLU(),
            nn.Dropout(p=0.1),
        ])

        n_final_in = 32
        self.dssp3_classifier = nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), \
                                    pad_mode='pad', padding=(3, 3, 0, 0), has_bias=True)
        self.dssp8_classifier = nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), \
                                    pad_mode='pad', padding=(3, 3, 0, 0), has_bias=True)
        self.diso_classifier = nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), \
                                    pad_mode='pad', padding=(3, 3, 0, 0), has_bias=True)

    def construct(self, embeddings, masks):
        """construct: IN: X = (B x L x F); OUT: (B x F x L, 1)"""
        x = embeddings * ops.expand_dims(masks, -1)
        x = ops.Transpose()(x, (0, 2, 1)).unsqueeze(-1)
        x = self.feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_clf = self.dssp3_classifier(x).squeeze(-1).transpose((0, 2, 1))  # OUT: (B x L x 3)
        d8_clf = self.dssp8_classifier(x).squeeze(-1).transpose((0, 2, 1))  # OUT: (B x L x 8)
        diso_clf = self.diso_classifier(x).squeeze(-1).transpose((0, 2, 1))  # OUT: (B x L x 2)
        return d3_clf, d8_clf, diso_clf


class MeanPoolingClassifier(nn.Cell):
    """Acid sequence level predictor; using mean pooling classifier."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dense = nn.Dense(EMBEDIING_LENGTH, POOLING_CHANNELS, activation='relu')
        self.classifier = nn.Dense(POOLING_CHANNELS, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def construct(self, embeddings, masks):
        """construct"""
        masks = ops.cast(masks, ms.float32)
        masked_inputs = embeddings * ops.expand_dims(masks, -1)
        mean_pooled = ops.ReduceMean(keep_dims=False)(masked_inputs, 1)
        mean_pooled = self.dropout(mean_pooled)
        compressed = self.dense(mean_pooled)
        output = self.classifier(compressed)
        return output


class EmbeddingTaskNet(nn.Cell):
    """Base net of Embedding part for downstream task."""
    def __init__(self, downstream_net, t5_config_path):
        super(EmbeddingTaskNet, self).__init__()
        self.downstream_net = downstream_net

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_config_path)
        self.t5.set_train(False)

        # freeze pretrain model parameters
        for param in self.t5.trainable_params():
            param.requires_grad = False

    def construct(self, inputs, masks):
        """construct"""
        masks = ops.cast(masks, ms.float32)
        embeddings = self.t5.encoder_forward(inputs, masks)
        output = self.downstream_net(embeddings, masks)
        return output

    def load_from_pretrained(self, config_path):
        """load downstream task checkpoint"""
        non_pretrained_param_dict = ms.load_checkpoint(config_path)
        param_not_load, _ = ms.load_param_into_net(self.downstream_net, non_pretrained_param_dict)
        self.downstream_net.set_train(False)
        self.set_train(False)
        logger.warning("Not Loaded param list: %s", param_not_load)

    def save_checkpoint(self, model_path):
        """save checkpoint"""
        non_pretrained_param_dict = {}
        for param in self.downstream_net.trainable_params():
            non_pretrained_param_dict[param.name] = param.data
        ms.save_checkpoint(non_pretrained_param_dict, model_path)

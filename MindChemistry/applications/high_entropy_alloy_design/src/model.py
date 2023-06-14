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
"""define models"""

from lightgbm import LGBMRegressor

import mindspore.nn as nn
from mindspore.common import initializer as init
from mindspore.common.initializer import HeNormal

from mindchemistry import AutoEncoder, FCNet, MLPNet


class WAE(nn.Cell):
    def __init__(self, params):
        super(WAE, self).__init__()
        uniform_scale = [init.Uniform(x ** -0.5) for x in params['channels']]
        self.wae_model = AutoEncoder(channels=params['channels'],
                                     weight_init=uniform_scale,
                                     has_bias=True,
                                     bias_init=uniform_scale,
                                     has_layernorm=params['layer_norm'],
                                     layernorm_epsilon=1e-5,
                                     has_activation=params['activation'],
                                     act='relu',
                                     out_act='softmax')

    def construct(self, inputs):
        return self.wae_model(inputs)

    def encode(self, inputs):
        return self.wae_model.encode(inputs)

    def decode(self, inputs):
        return self.wae_model.decode(inputs)


class Classifier(nn.Cell):
    def __init__(self, params):
        super(Classifier, self).__init__()
        uniform_scale = [init.Uniform(x ** -0.5) for x in params['channels']]
        self.cls_model = FCNet(channels=params['channels'],
                               weight_init=uniform_scale,
                               has_bias=True,
                               bias_init=uniform_scale,
                               has_dropout=params['dropout'],
                               has_layernorm=False,
                               has_activation=params['activation'],
                               act='sigmoid')

    def construct(self, inputs):
        return self.cls_model(inputs)


class MlpModel(nn.Cell):
    def __init__(self, params):
        super(MlpModel, self).__init__()
        # load BO searched params
        num_feature = params['num_feature'][int(params['stage_num']) - 1]
        num_output = params['num_output']
        layer_num = int(params['module__w'])
        hidden_num = int(params['module__n_hidden'])
        # model init
        self.mlp_model = MLPNet(in_channels=num_feature,
                                out_channels=num_output,
                                layers=layer_num,
                                neurons=hidden_num,
                                weight_init=HeNormal(),
                                has_bias=True,
                                has_dropout=False,
                                has_layernorm=False,
                                has_activation=True,
                                act=['relu'] * (layer_num - 1))

    def construct(self, inputs):
        return self.mlp_model(inputs)


def TreeModel(params):
    tree_params = {
        "num_leaves": int(round(params['num_leaves'])),
        'min_child_samples': int(round(params['min_child_samples'])),
        'learning_rate': params['learning_rate'],
        'n_estimators': int(round(params['n_estimators'])),
        'max_bin': int(round(params['max_bin'])),
        'colsample_bytree': max(min(params['colsample_bytree'], 1), 0),
        'subsample': max(min(params['subsample'], 1), 0),
        'max_depth': int(round(params['max_depth'])),
        'reg_lambda': max(params['reg_lambda'], 0),
        'reg_alpha': max(params['reg_alpha'], 0),
        'min_split_gain': params['min_split_gain'],
        'min_child_weight': params['min_child_weight'],
        'objective': 'regression',
        'verbose': -1
    }
    model = LGBMRegressor(**tree_params)
    return model

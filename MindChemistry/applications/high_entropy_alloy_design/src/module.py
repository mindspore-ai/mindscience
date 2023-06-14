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
"""2-phase modules"""
import os
import pandas as pd

from src.dataset import HighEntropyAlloy
from src.model import WAE, Classifier, MlpModel, TreeModel
from src.utils import sampler, gaussian_mixture_model
from src.trainer import train_wae, train_cls, train_mlp, train_tree


class GenerationModule():
    """ Generation Module"""

    def __init__(self, wae_params, cls_params):
        self.wae_params = wae_params
        self.cls_params = cls_params
        self.dataset = HighEntropyAlloy(self.wae_params['root'])
        self.input, self.label = self.dataset.process_train_gen_data()
        self.wae_model = WAE(self.wae_params)
        self.cls_model = Classifier(self.cls_params)

    def train(self):
        # train WAE
        wae_data = self.input
        latents = train_wae(self.wae_model, wae_data, self.wae_params)
        # train CLS
        cls_data = (latents, self.label)
        train_cls(self.cls_model, cls_data, self.cls_params)
        # generate Gaussian Mixture Model
        gm_model = gaussian_mixture_model(latents, self.wae_params)
        # generate samples
        sampler(gm=gm_model, classifier=self.cls_model, n_samples=5000, sigma=0.5)


class RankingModule():
    """Ranking Module"""

    def __init__(self, params):
        self.params = params
        self.dataset = HighEntropyAlloy(self.params['root'])
        self.target_dir = os.path.join(self.params['root'], 'data/')
        self.mlp_params_stage1 = pd.read_excel(self.target_dir + self.params['NN_stage1_dir'])
        self.mlp_params_stage2 = pd.read_excel(self.target_dir + self.params['NN_stage2_dir'])
        self.tree_params_stage1 = pd.read_excel(self.target_dir + self.params['Tree_stage1_dir'])
        self.tree_params_stage2 = pd.read_excel(self.target_dir + self.params['Tree_stage2_dir'])

    def train(self):
        # train 1st stage ranking models
        self.params['stage_num'] = 1
        self.params['model_name'] += str(self.params['stage_num'])
        for i in range(self.params['num_group']):
            for j in range(self.params['seed_start'], self.params['seed_end']):
                data = self.dataset.process_train_rank_data(stage_num=1, seed=j)
                self.params.update(self.mlp_params_stage1.iloc[i])
                mlp_model = MlpModel(self.params)
                train_mlp(mlp_model, data, j, self.params)
                self.params.update(self.tree_params_stage1.iloc[i])
                tree_model = TreeModel(self.params)
                train_tree(tree_model, data, j - 10, self.params)
        # train 2nd stage ranking models
        self.params['stage_num'] = 2
        self.params['model_name'] += str(self.params['stage_num'])
        for i in range(self.params['num_group']):
            for j in range(self.params['seed_start'], self.params['seed_end']):
                data = self.dataset.process_train_rank_data(stage_num=2, seed=j)
                self.params.update(self.mlp_params_stage2.iloc[i])
                mlp_model = MlpModel(self.params)
                train_mlp(mlp_model, data, j, self.params)
                self.params.update(self.tree_params_stage2.iloc[i])
                tree_model = TreeModel(self.params)
                train_tree(tree_model, data, j - 10, self.params)

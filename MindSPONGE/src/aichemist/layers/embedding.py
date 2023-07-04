# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
"""
Embedding
"""

import mindspore as ms
from mindspore import nn
from mindspore import ops


class AtomEncoder(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, emb_dim, feature_dims, use_scalar_feat=True, n_feats_to_use=None):
        # first element of feature_dims tuple is a list with the length of each categorical feature
        # and the second is the number of scalar features
        super().__init__()
        self.use_scalar_feat = use_scalar_feat
        self.n_feats_to_use = n_feats_to_use
        self.atom_embedding_list = nn.CellList()
        self.num_categorical_feat = len(feature_dims[0])
        self.num_scalar_feat = feature_dims[1]
        for i, dim in enumerate(feature_dims[0]):
            emb = nn.Embedding(dim, emb_dim)
            self.atom_embedding_list.append(emb)
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_feat > 0:
            self.linear = nn.Dense(self.num_scalar_feat, emb_dim)

    def construct(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_feat + self.num_scalar_feat
        for i in range(self.num_categorical_feat):
            x_embedding += self.atom_embedding_list[i](x[:, i].astype(ms.int32))
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_feat > 0 and self.use_scalar_feat:
            x_embedding += self.linear(x[:, self.num_categorical_feat:])
        if ops.isnan(x_embedding).any():
            print('nan')
        return x_embedding

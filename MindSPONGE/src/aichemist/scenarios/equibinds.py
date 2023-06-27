# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
Equibinds
"""
from ..core import Registry as R
from .. import core
from ..core import args_from_dict


@R.register('scenario.Equibinds')
class Equibinds(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """
    def __init__(self, net, metrics=None, criterion=None, norm=None):
        super().__init__()
        self.net = net
        self.metrics = metrics
        self.criterion = criterion
        self.norm = norm

    def preprocess(self, *datasets):
        """_summary_

        Returns:
            _type_: _description_
        """
        for ds in datasets:
            if ds is not None:
                ds.process()
                lig_input_edge_feats_dim = ds[0][0].edge_feat.shape[1]
                rec_input_edge_feats_dim = ds[0][1].edge_feat.shape[1]
        self.net.initialize(lig_input_edge_feats_dim=lig_input_edge_feats_dim,
                            rec_input_edge_feats_dim=rec_input_edge_feats_dim)
        return datasets

    def loss_fn(self, *args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        args, kwargs = args_from_dict(*args, **kwargs)
        ligs, recs, pockets, new_pockets, geoms = args
        ligs_coord_, ligs_keypts, recs_keypts, rotations, \
            translations, geom_reg_loss = self.net(ligs, recs, geoms)
        loss, _ = self.criterion(ligs, recs, ligs_coord_, pockets, new_pockets, ligs_keypts,
                                 recs_keypts, rotations, translations, geom_reg_loss)
        self.metrics.update(ligs_coord_, ligs.coord)
        return loss, (ligs_coord_, ligs.coord)

    def eval(self, *batch):
        """_summary_

        Returns:
            _type_: _description_
        """
        ligs, recs, pockets, new_pockets, geoms = batch
        loss, (ligs_coord_, ligs.coord) = self.loss_fn(ligs, recs, pockets, new_pockets, geoms)
        return loss, (ligs_coord_, ligs.coord)

    def predict(self, *batch):
        """_summary_

        Returns:
            _type_: _description_
        """
        ligs, recs, pockets, new_pockets, geoms = batch
        pred = self.net(ligs, recs, pockets, new_pockets, geoms)
        return pred

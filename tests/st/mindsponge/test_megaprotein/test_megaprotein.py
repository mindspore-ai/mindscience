# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Test MEGAFoldProtein examples."""
import numpy as np
import mindspore as ms
from mindsponge import PipeLine

ms.set_context(mode=ms.GRAPH_MODE)

# MEGA-EvoGen推理获取蛋白质生成MSA后的特征
fasta = "GYDKDLCEWSMTADQTEVETQIEADIMNIVKRDRPEMKAEVQKQLKSGGVMQYNYVLYCDKNFNNKNIIAEVVGE"
msa_generator = PipeLine(name="MEGAEvoGen")
msa_generator.set_device_id(0)
msa_generator.initialize(key="evogen_predict_256")
msa_generator.model.from_pretrained()
msa_feature = msa_generator.predict(fasta)

# MEGA-Fold推理获取蛋白质结构信息
fold_prediction = PipeLine(name="MEGAFold")
fold_prediction.set_device_id(0)
fold_prediction.initialize(key="predict_256")
fold_prediction.model.from_pretrained()
final_atom_positions, final_atom_mask, aatype, _, _ = fold_prediction.model.predict(msa_feature)

# MEGA-Assessment对蛋白质结构进行评价
protein_assessment = PipeLine(name="MEGAAssessment")
protein_assessment.set_device_id(0)
protein_assessment.initialize("predict_256")
protein_assessment.model.from_pretrained()
msa_feature['decoy_aatype'] = np.pad(aatype, (0, 256 - aatype.shape[0]))
msa_feature['decoy_atom_positions'] = np.pad(final_atom_positions,
                                             ((0, 256 - final_atom_positions.shape[0]), (0, 0), (0, 0)))
msa_feature['decoy_atom_mask'] = np.pad(final_atom_mask, ((0, 256 - final_atom_mask.shape[0]), (0, 0)))

res = protein_assessment.model.predict(msa_feature)
print("score is:", np.mean(res[:msa_feature['num_residues']]))

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
"""Models"""
from .colabdesign import COLABDESIGN, ColabDesignDataSet, colabdesign_configuration
from .deepdr import DeepDR, DeepDRDataSet, deepdr_configuration
from .deepfri import DeepFri, DeepFriDataSet, deepfri_configuration
from .esm_if1 import ESM, ESMDataSet, esm_configuration
from .esm2 import ESM2, ESM2DataSet, esm2_configuration
from .grover import Grover, GroverDataSet, grover_configuration
from .kgnn import KGNN, KGNNDataSet, kgnn_configuration
from .megaassessment import MEGAAssessment, MEGAAssessmentDataSet, megaassessment_configuration
from .megaevogen import MEGAEvoGen, MEGAEvoGenDataSet, megaevogen_configuration
from .megafold import MEGAFold, MEGAFoldDataSet, megafold_configuration
from .multimer import Multimer, MultimerDataSet, multimer_configuration
from .proteinmpnn import ProteinMpnn, ProteinMpnnDataset, proteinmpnn_configuration
from .ufold import UFold, UFoldDataSet, ufold_configuration
from .rasp import RASP, RASPDataSet, rasp_configuration

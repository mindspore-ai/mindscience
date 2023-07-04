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
Syntheisability score
"""

import os
import sys
import math
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from .. import util

module = sys.modules.get(__name__)
path = os.path.dirname(__file__)

# Calculate synthetic accessibility of molecules
# Code adapted from RDKit
# https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py


def read_fragment_scores():
    """_summary_

    Returns:
        _type_: _description_
    """
    url = "https://github.com/rdkit/rdkit/raw/master/Contrib/SA_Score/fpscores.pkl.gz"
    md5 = "2f80a169f9075e977154f9caec9e5c26"

    zip_file = util.download(url, path, md5=md5)
    pkl_file = util.extract(zip_file)
    with open(pkl_file, "rb") as fin:
        data = pickle.load(fin)
    out_dict = {}
    for i in data:
        for j in range(1, len(i)):
            out_dict[i[j]] = float(i[0])
    return out_dict


def num_bridge_heads_and_spiro(mol):
    """_summary_

    Args:
        mol (_type_): _description_

    Returns:
        _type_: _description_
    """
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge_head = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridge_head, n_spiro


def calc_score(m):
    """_summary_

    Args:
        m (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not hasattr(module, "fscores"):
        module.fscores = read_fragment_scores()
    fscores = module.fscores

    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bit_id, v in fps.items():
        nf += v
        sfp = bit_id
        score1 += fscores.get(sfp, -4) * v
    score1 /= nf

    n_atoms = m.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    n_bridge_heads, n_spiro = num_bridge_heads_and_spiro(m)
    n_macrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            n_macrocycles += 1

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridge_heads + 1)
    macrocycle_penalty = 0.0
    if n_macrocycles > 0:
        macrocycle_penalty = math.log10(2)

    score2 = 0.0 - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty

    score3 = 0.0
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    xmin = -4.0
    xmax = 2.5
    sascore = 11. - (sascore - xmin + 1) / (xmax - xmin) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore

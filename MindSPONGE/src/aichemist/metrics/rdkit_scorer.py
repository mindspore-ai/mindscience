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
The score metric for molecules. All of them are implemented from RDKit package.
"""

import os
import sys
import math
import pickle
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from .. import utils
from ..configs import Registry as R


module = sys.modules.get(__name__)
path = os.path.dirname(__file__)


@R.register('logp')
def logp(pred):
    """
    Logarithm of partition coefficient between octanol and water for a compound.

    Args:
        pred (MoleculeBatch): molecules to evaluate
    """
    logps = []
    for mol in pred:
        mol = mol.to_molecule()
        try:
            with utils.no_rdkit_log():
                mol.UpdatePropertyCache()
                score = Descriptors.MolLogP(mol)
        except Chem.AtomValenceException:
            score = 0
        logps.append(score)

    return np.array(logps)


@R.register('plogp')
def penalized_logp(pred):
    """
    Logarithm of partition coefficient, penalized by cycle length and synthetic accessibility.

    Args:
        pred (PackedMolecule): molecules to evaluate
    """
    # statistics from ZINC250k
    logp_mean = 2.4570953396190123
    logp_std = 1.434324401111988
    sa_mean = 3.0525811293166134
    sa_std = 0.8335207024513095
    cycle_mean = 0.0485696876403053
    cycle_std = 0.2860212110245455

    plogp = []
    for mol in pred:
        cycles = nx.cycle_basis(nx.Graph(mol.edge_list[:, :2].tolist()))
        if cycles:
            len_cycle = [len(cycle) for cycle in cycles]
            max_cycle = max(len_cycle)
            cycle = max(0, max_cycle - 6)
        else:
            cycle = 0
        mol = mol.to_molecule()
        try:
            with utils.no_rdkit_log():
                mol.UpdatePropertyCache()
                Chem.GetSymmSSSR(mol)
                logp_val = Descriptors.MolLogP(mol)
                sa_val = calc_score(mol)
            logp_val = (logp_val - logp_mean) / logp_std
            sa_val = (sa_val - sa_mean) / sa_std
            cycle = (cycle - cycle_mean) / cycle_std
            score = logp_val - sa_val - cycle
        except Chem.AtomValenceException:
            score = -30
        plogp.append(score)

    return np.array(plogp)


@R.register('sa')
def sa(pred):
    """
    Synthetic accessibility score.

    Args:
        pred (PackedMolecule): molecules to evaluate
    """
    sas = []
    for mol in pred:
        with utils.no_rdkit_log():
            score = calc_score(mol.to_molecule())
        sas.append(score)

    return np.array(sas)


@R.register('qed')
def qed(pred):
    """
    Quantitative estimation of drug-likeness.

    Args:
        pred (PackedMolecule): molecules to evaluate
    """
    qeds = []
    for mol in pred:
        try:
            score = Descriptors.qed(mol.to_molecule())
        except (RuntimeError, Chem.AtomValenceException):
            score = -1
        qeds.append(score)

    return np.array(qeds)


@R.register('validity')
def validity(pred):
    """
    Chemical validity of molecules.

    Args:
        pred (PackedMolecule): molecules to evaluate
    """
    validitis = []
    for mol in pred:
        with utils.no_rdkit_log():
            smiles = mol.to_smiles()
            mol = Chem.MolFromSmiles(smiles)
        validitis.append(1 if mol else 0)

    return np.array(validitis)

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

    zip_file = utils.download(url, path, md5=md5)
    pkl_file = utils.extract(zip_file)
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

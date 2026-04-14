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
feature
"""

import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from Bio.PDB import SASA

from ..configs import Registry as R

bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
atom2val = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
bond2val = [1, 2, 3, 1.5]
id2bond = {v: k for k, v in bond2id.items()}
empty_mol = Chem.MolFromSmiles("")
dummy_mol = Chem.MolFromSmiles("**")
dummy_atom = dummy_mol.GetAtomWithIdx(0)
dummy_bond = dummy_mol.GetBondWithIdx(0)

atom_voc = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_voc = {a: i for i, a in enumerate(atom_voc)}
degree_voc = range(7)
num_hs_voc = range(7)
formal_charge_voc = range(-5, 6)
chiral_tag_voc = range(4)
total_valence_voc = range(8)
num_radical_voc = range(8)
hybridization_voc = range(len(Chem.HybridizationType.values))

bond_type_voc = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                 Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
bond_type_voc = {b: i for i, b in enumerate(bond_type_voc)}
bond_dir_voc = range(len(Chem.BondDir.values))
bond_stereo_voc = range(len(Chem.BondStereo.values))


# Featurization for Equibind
periodic_table = GetPeriodicTable()

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

lig_feature_dims = (list(map(len, [
    allowable_features.get('possible_atomic_num_list'),
    allowable_features.get('possible_chirality_list'),
    allowable_features.get('possible_degree_list'),
    allowable_features.get('possible_formal_charge_list'),
    allowable_features.get('possible_implicit_valence_list'),
    allowable_features.get('possible_numH_list'),
    allowable_features.get('possible_number_radical_e_list'),
    allowable_features.get('possible_hybridization_list'),
    allowable_features.get('possible_is_aromatic_list'),
    allowable_features.get('possible_numring_list'),
    allowable_features.get('possible_is_in_ring3_list'),
    allowable_features.get('possible_is_in_ring4_list'),
    allowable_features.get('possible_is_in_ring5_list'),
    allowable_features.get('possible_is_in_ring6_list'),
    allowable_features.get('possible_is_in_ring7_list'),
    allowable_features.get('possible_is_in_ring8_list'),
])), 1)  # number of scalar features
rec_atom_feature_dims = (list(map(len, [
    allowable_features.get('possible_amino_acids'),
    allowable_features.get('possible_atomic_num_list'),
    allowable_features.get('possible_atom_type_2'),
    allowable_features.get('possible_atom_type_3'),
])), 2)

rec_residue_feature_dims = (list(map(len, [
    allowable_features.get('possible_amino_acids')
])), 2)


def indexing(data, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return data.index(e)
    except RuntimeError:
        return len(data) - 1


def lig_atom_featurizer(mol):
    """
    lig_atom_feature
    """
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            indexing(allowable_features.get('possible_atomic_num_list'), atom.GetAtomicNum()),
            allowable_features.get('possible_chirality_list').index(str(atom.GetChiralTag())),
            indexing(allowable_features.get('possible_degree_list'), atom.GetTotalDegree()),
            indexing(allowable_features.get('possible_formal_charge_list'), atom.GetFormalCharge()),
            indexing(allowable_features.get('possible_implicit_valence_list'), atom.GetImplicitValence()),
            indexing(allowable_features.get('possible_numH_list'), atom.GetTotalNumHs()),
            indexing(allowable_features.get('possible_number_radical_e_list'), atom.GetNumRadicalElectrons()),
            indexing(allowable_features.get('possible_hybridization_list'), str(atom.GetHybridization())),
            allowable_features.get('possible_is_aromatic_list').index(atom.GetIsAromatic()),
            indexing(allowable_features.get('possible_numring_list'), ringinfo.NumAtomRings(idx)),
            allowable_features.get('possible_is_in_ring3_list').index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features.get('possible_is_in_ring4_list').index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features.get('possible_is_in_ring5_list').index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features.get('possible_is_in_ring6_list').index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features.get('possible_is_in_ring7_list').index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features.get('possible_is_in_ring8_list').index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])
    return np.array(atom_features_list, dtype=np.int32)


# probe_radius: in A. Default is 1.40 roughly the radius of a water molecule.
# n_points: resolution of the surface of each atom. Default is 100. A higher number of points results
# in more precise measurements, but slows down the calculation.
sr = SASA.ShrakeRupley(probe_radius=1.4, n_points=100)


def rec_atom_featurizer(rec, surface_indices):
    """
    rec_atom_featurizer
    """
    surface_atom_feat = []
    c_alpha_feat = []
    sr.compute(rec, level="A")
    for i, atom in enumerate(rec.get_atoms()):
        if i in surface_indices or atom.name == 'CA':
            atom_name, element = atom.name, atom.element
            sasa = atom.sasa
            bfactor = atom.bfactor
            if element == 'CD':
                element = 'C'
            assert element != ''
            assert not np.isinf(bfactor)
            assert not np.isnan(bfactor)
            assert not np.isinf(sasa)
            assert not np.isnan(sasa)
            try:
                atomic_num = periodic_table.GetAtomicNumber(element)
            except RuntimeError:
                atomic_num = -1
            atom_feat = [indexing(allowable_features.get('possible_amino_acids'), atom.get_parent().get_resname()),
                         indexing(allowable_features.get('possible_atomic_num_list'), atomic_num),
                         indexing(allowable_features.get('possible_atom_type_2'), (atom_name + '*')[:2]),
                         indexing(allowable_features.get('possible_atom_type_3'), atom_name),
                         sasa,
                         bfactor]
            if i in surface_indices:
                surface_atom_feat.append(atom_feat)
            if atom.name == 'CA':
                c_alpha_feat.append(atom_feat)
    return np.array(c_alpha_feat, dtype=np.float32), np.array(surface_atom_feat, dtype=np.float32)


def rec_residue_featurizer(rec):
    """
    rec_residue_featurizer
    """
    feature_list = []
    sr.compute(rec, level="R")
    for residue in rec.get_residues():
        sasa = residue.sasa
        for atom in residue:
            if atom.name == 'CA':
                bfactor = atom.bfactor
        assert not np.isinf(bfactor)
        assert not np.isnan(bfactor)
        assert not np.isinf(sasa)
        assert not np.isnan(sasa)
        feature_list.append([indexing(allowable_features.get('possible_amino_acids'), residue.get_resname()),
                             sasa,
                             bfactor])
    return np.array(feature_list)  # (N_res, 1)


def one_hot(depth, indices):
    """one hot compute"""
    res = np.eye(depth)[indices.reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def onehot(x, voc, allow_unknown=False):
    """onehot"""
    if isinstance(voc, int):
        res = np.eye(voc)[x.reshape(-1)]
        return res.reshape(list(x.shape) + [x])
    if x in voc:
        if isinstance(voc, dict):
            index = voc.get(x)
        else:
            index = voc.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(voc) + 1)
        if index == -1:
            warnings.warn(f"Unknown value `{x}`")
        feature[index] = 1
    else:
        feature = [0] * len(voc)
        if index == -1:
            raise ValueError(f"Unknown value `{x}`. Available vocabulary is `{voc}`" % (x, voc))
        feature[index] = 1
    return feature


@R.register("feature.atom.default")
def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetChiralTag(): one-hot embedding for atomic chiral tag

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return onehot(atom.GetSymbol(), atom_voc, allow_unknown=True) + \
        onehot(atom.GetChiralTag(), chiral_tag_voc) + \
        onehot(atom.GetTotalDegree(), degree_voc, allow_unknown=True) + \
        onehot(atom.GetFormalCharge(), formal_charge_voc) + \
        onehot(atom.GetTotalNumHs(), num_hs_voc) + \
        onehot(atom.GetNumRadicalElectrons(), num_radical_voc) + \
        onehot(atom.GetHybridization(), hybridization_voc) + \
        [atom.GetIsAromatic(), atom.IsInRing()]


@R.register("feature.atom.center_identification")
def atom_center_identification(atom):
    """Reaction center identification atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return onehot(atom.GetSymbol(), atom_voc, allow_unknown=True) + \
        onehot(atom.GetTotalNumHs(), num_hs_voc) + \
        onehot(atom.GetTotalDegree(), degree_voc, allow_unknown=True) + \
        onehot(atom.GetTotalValence(), total_valence_voc) + \
        [atom.GetIsAromatic(), atom.IsInRing()]


@R.register("feature.atom.synthon_completion")
def atom_synthon_completion(atom):
    """Synthon completion atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        IsInRing(): whether the atom is in a ring

        IsInRingSize(3, 4, 5, 6): whether the atom is in a ring of a particular size

        IsInRing() and not IsInRingSize(3, 4, 5, 6): whether the atom is in a ring and not in a ring of 3, 4, 5, 6
    """
    return onehot(atom.GetSymbol(), atom_voc, allow_unknown=True) + \
        onehot(atom.GetTotalNumHs(), num_hs_voc) + \
        onehot(atom.GetTotalDegree(), degree_voc, allow_unknown=True) + \
        [atom.IsInRing(), atom.IsInRingSize(3), atom.IsInRingSize(4),
         atom.IsInRingSize(5), atom.IsInRingSize(6),
         atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4))
         and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]


@R.register("feature.atom.symbol")
def atom_symbol(atom):
    """Symbol atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
    """
    if not isinstance(atom, str):
        atom = atom.GetSymbol()
    return onehot(atom, atom_voc, allow_unknown=True)


@R.register("feature.bond.symbol")
def bond_symbol(bond):
    """Symbol atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
    """
    if not isinstance(bond, str):
        bond = str(bond.GetBondType())
    voc = bond2id
    return onehot(bond, voc, allow_unknown=True)


@R.register("feature.atom.explicit_property_prediction")
def atom_explicit_property_prediction(atom):
    """Explicit property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_voc, allow_unknown=True) + \
        onehot(atom.GetDegree(), degree_voc, allow_unknown=True) + \
        onehot(atom.GetTotalValence(), total_valence_voc, allow_unknown=True) + \
        onehot(atom.GetFormalCharge(), formal_charge_voc) + \
        [atom.GetIsAromatic()]


@R.register("feature.atom.property_prediction")
def atom_property_prediction(atom):
    """Property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetIsAromatic(): whether the atom is aromatic
    """
    return onehot(atom.GetSymbol(), atom_voc, allow_unknown=True) + \
        onehot(atom.GetDegree(), degree_voc, allow_unknown=True) + \
        onehot(atom.GetTotalNumHs(), num_hs_voc, allow_unknown=True) + \
        onehot(atom.GetTotalValence(), total_valence_voc, allow_unknown=True) + \
        onehot(atom.GetFormalCharge(), formal_charge_voc, allow_unknown=True) + \
        [atom.GetIsAromatic()]


@R.register("feature.atom.position")
def atom_position(atom):
    """
    Atom position in the molecular conformation.
    Return 3D position if available, otherwise 2D position is returned.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]


@R.register("feature.atom.pretrain")
def atom_pretrain(atom):
    """Atom feature for pretraining.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetChiralTag(): one-hot embedding for atomic chiral tag
    """
    return onehot(atom.GetSymbol(), atom_voc, allow_unknown=True) + \
        onehot(atom.GetChiralTag(), chiral_tag_voc)


@R.register("feature.bond.default")
def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetBondDir(): one-hot embedding for the direction of the bond

        GetStereo(): one-hot embedding for the stereo configuration of the bond

        GetIsConjugated(): whether the bond is considered to be conjugated
    """
    return onehot(bond.GetBondType(), bond_type_voc) + \
        onehot(bond.GetBondDir(), bond_dir_voc) + \
        onehot(bond.GetStereo(), bond_stereo_voc) + \
        [int(bond.GetIsConjugated())]


@R.register("feature.bond.length")
def bond_length(bond):
    """
    Bond length in the molecular conformation.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]


@R.register("feature.bond.property_prediction")
def bond_property_prediction(bond):
    """Property prediction bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetIsConjugated(): whether the bond is considered to be conjugated

        IsInRing(): whether the bond is in a ring
    """
    return onehot(bond.GetBondType(), bond_type_voc) + \
        [int(bond.GetIsConjugated()), bond.IsInRing()]


@R.register("feature.bond.pretrain")
def bond_pretrain(bond):
    """Bond feature for pretraining.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetBondDir(): one-hot embedding for the direction of the bond
    """
    return onehot(bond.GetBondType(), bond_type_voc) + \
        onehot(bond.GetBondDir(), bond_dir_voc)


@R.register("feature.mol.ecfp")
def molecule_ecfp(mol, radius=2, length=1024):
    """Extended Connectivity Fingerprint molecule feature.

    Features:
        GetMorganFingerprintAsBitVect(): a Morgan fingerprint for a molecule as a bit vector
    """
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)


def molecule_default(mol):
    """Default molecule feature."""
    return molecule_ecfp(mol)


def distance(dist_list, divisor=0.75) -> np.ndarray:
    # you want to use a divisor that is close to 4/7 times the average distance that you want to encode
    """distance"""
    length_scale_list = [1.5 ** x for x in range(15)]
    center_list = [0. for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))
                        for length_scale, center in zip(length_scale_list, center_list)]

    transformed_dist = np.array(transformed_dist).T
    transformed_dist = transformed_dist.reshape((num_edge, -1))
    return transformed_dist


ECFP = molecule_ecfp

__all__ = [
    "atom_default", "atom_center_identification", "atom_synthon_completion",
    "atom_symbol", "atom_explicit_property_prediction", "atom_property_prediction",
    "atom_position", "atom_pretrain",
    "bond_default", "bond_length", "bond_property_prediction", "bond_pretrain",
    "molecule_ecfp", "molecule_default",
    "ECFP",
    "distance"
]

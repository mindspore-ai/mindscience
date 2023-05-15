# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"analysis"
import os
import logging
import io
from typing import Optional
import pickle
import dataclasses
from Bio.PDB import PDBParser
import numpy as np

from commons.res_constants import EQUI_VARIANCE
import mindsponge.common.residue_constants as residue_constants

log = logging.getLogger()
log.setLevel(logging.ERROR)

ATOM_TYPES_WITH_H = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT', 'HNZ2', 'HD2*', 'HG12', 'HH1*', 'HH', 'HH11', 'HNZ1',
    'HNE', 'HD*', 'HB1', 'HG2', 'HG23', 'HD22', 'HN', 'HB', 'HZ3', 'HE21',
    'HZ2', 'HA1', 'HH2*', 'HE2*', 'HG2*', 'HB2', 'HG22', 'HB*', 'HN22', 'HG*',
    'HSG', 'HE*', 'HE22', 'HE2', 'HA*', 'HA3', 'HD2', 'HH22', 'HNE1', 'HOG',
    'HZ1', 'HD1', 'HD12', 'HH21', 'HH12', 'HB3', 'H', 'HG3', 'HA', 'HN21',
    'HA2', 'HNZ3', 'HOH', 'HG1*', 'HD**', 'HH2', 'HE', 'HG**', 'HG21', 'HND1',
    'HD3', 'HH**', 'HD13', 'HG1', 'HD23', 'HG11', 'HZ', 'HG13', 'HNE2', 'HG',
    'HE1', 'HD11', 'HD21', 'HZ*', 'HE3', 'HNZ*', 'HD1*'
]

ATOM_ORDER_WITH_H = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES_WITH_H)}

ATOM_TYPE_NUM_WITH_H = len(ATOM_TYPES_WITH_H)

RESNAME_TO_IDX = residue_constants.resname_to_idx

IDX_TO_RESNAME = {val: key for key, val in RESNAME_TO_IDX.items()}


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]


def from_pdb_string_with_h(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """
    Takes a PDB string and constructs a Protein object.
    WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
        A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser()
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    models = models[:1]
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        chains = chains[:1]
        if len(chains) != 1:
            raise ValueError(
                'Only single chain PDBs are supported when chain_id not specified. '
                f'Found {len(chains)} chains.')
        else:
            chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []

    atom_order_with_h = {atom_type: i for i, atom_type in enumerate(ATOM_TYPES_WITH_H)}
    atom_type_num_with_h = len(ATOM_TYPES_WITH_H)
    for res in chain:
        if res.id[2] != ' ':
            raise ValueError(
                f'PDB contains an insertion code at chain {chain.id} and residue '
                f'index {res.id[1]}. These are not supported.')
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((atom_type_num_with_h, 3))
        mask = np.zeros((atom_type_num_with_h,))
        res_b_factors = np.zeros((atom_type_num_with_h,))
        for atom in res:
            if atom.name not in ATOM_TYPES_WITH_H:
                continue
            pos[atom_order_with_h[atom.name]] = atom.coord
            mask[atom_order_with_h[atom.name]] = 1.
            res_b_factors[atom_order_with_h[atom.name]] = atom.bfactor
        if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip it.
            continue
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors))


def check_restraints(pdb_path, restraints, distance_threshold=8, pdb_align=1, return_new_restraints=False):
    '''check_restraints'''
    pdb_file_path = os.path.join(pdb_path)
    with open(pdb_file_path, 'r') as f:
        prot_pdb = from_pdb_string_with_h(f.read())
    aatype_all = prot_pdb.aatype
    atom99_positions = prot_pdb.atom_positions.astype(np.float32)
    all_atom_mask = prot_pdb.atom_mask.astype(np.float32)

    error_count = 0
    hdist_all = []
    ur_tuple_new = []
    for res1, candidates in restraints:
        try:
            i, atype1 = res1
            if i >= aatype_all.shape[0]:
                continue
            aatype1 = IDX_TO_RESNAME[aatype_all[i - pdb_align]]
            mask1 = all_atom_mask[i - pdb_align]
            pos1 = atom99_positions[i - pdb_align]
            if atype1 not in EQUI_VARIANCE.get(aatype1).keys():
                error_count += 1
                continue

            h1_atom = EQUI_VARIANCE.get(aatype1).get(atype1).get("equivariance")

            h_dist = 999.0
            for res2 in candidates:
                j, atype2 = res2
                if j >= aatype_all.shape[0]:
                    continue
                aatype2 = IDX_TO_RESNAME[aatype_all[j - pdb_align]]
                pos2 = atom99_positions[j - pdb_align]
                mask2 = all_atom_mask[j - pdb_align]
                if atype2 not in EQUI_VARIANCE.get(aatype2).keys():
                    error_count += 1
                    continue

                h2_atom = EQUI_VARIANCE.get(aatype2).get(atype2).get("equivariance")

                h1_idx = np.array([ATOM_ORDER_WITH_H[h1s] for h1s in h1_atom])
                h2_idx = np.array([ATOM_ORDER_WITH_H[h2s] for h2s in h2_atom])

                mask = mask1[None, :] * mask2[:, None]
                h_dists = np.sqrt((np.square(pos1[h1_idx, :][None] - pos2[h2_idx, :][:, None])).sum(-1) + 1e-8)

                h_dists = h_dists * mask[h2_idx][:, h1_idx]
                h_dists[mask[h2_idx][:, h1_idx] == 0] = 999.0
                h_dist = min(np.min(h_dists), h_dist)

            if h_dist < distance_threshold:
                ur_tuple_new.append((i, j))

            hdist_all.append(h_dist)
        except Exception as _:
            continue

    if not restraints:
        num_restraints, good_num, aatype_all.shape[0], hdist_all, good_rate, ur_per_res = 0, 0, aatype_all.shape[
            0], hdist_all, 1, 0
    else:
        hdist_all = np.array(hdist_all)
        ok_num = np.sum(hdist_all < distance_threshold)
        num_restraints = len(hdist_all)
        good_num = ok_num
        good_rate = round(ok_num / len(hdist_all), 4)
        ur_per_res = round(len(hdist_all) / aatype_all.shape[0], 2)
    if return_new_restraints:
        stats = num_restraints, good_num, aatype_all.shape[0], hdist_all, good_rate, ur_per_res, ur_tuple_new
    else:
        stats = num_restraints, good_num, aatype_all.shape[0], hdist_all, good_rate, ur_per_res
    return stats


def replace_q_in_atype(atype):
    '''replace_q_in_atype'''
    if "Q" in atype:
        atype = "H" + atype[1:] + "*"

    return atype


def remove_duplicates(restraints):
    '''remove_duplicates'''
    output_restraints = []
    for i, atype1, j, atype2 in restraints:
        if i > j:
            i, atype1, j, atype2 = j, atype2, i, atype1

        output_restraints.append([i, atype1, j, atype2])

    output_restraints = list(set([tuple(t) for t in output_restraints]))
    return output_restraints


def preprocess_restraints(restraints, num_gap=0, index_distance_threshold=-1):
    '''preprocess_restraints'''
    restraints = remove_duplicates(restraints)
    restraints_new = []
    for i, atype1, j, atype2 in restraints:
        atype1 = replace_q_in_atype(atype1)
        atype2 = replace_q_in_atype(atype2)
        i = i - num_gap
        j = j - num_gap
        if i < 1 or j < 1 or atype1 not in ATOM_TYPES_WITH_H or atype2 not in ATOM_TYPES_WITH_H:
            continue
        if abs(i - j) >= index_distance_threshold:
            restraints_new.append([[i, atype1], [[j, atype2]]])
    return restraints_new


def confidence(filename):
    '''confidence'''
    with open(filename, "r") as f:
        content = f.readlines()
    confidences = []
    for line in content:
        words = line.split()
        if len(words) > 3 and words[2] == "CA":
            confidences.append(float(words[-2]))
    avg_conf = sum(confidences) / len(confidences)
    return round(avg_conf, 3)


def select_pdb_by_conf(local_pdb_paths, return_conf=False):
    '''select_pdb_by_conf'''
    output_path_name = None
    max_conf = -100
    confs_all = []
    for name in local_pdb_paths:
        conf = confidence(name)
        if conf > max_conf:
            output_path_name = name
            max_conf = conf
        confs_all.append(conf)
    if return_conf:
        return output_path_name, np.max(confs_all), np.median(confs_all)
    return output_path_name


def gtur_vs_gtpdb(gtur_path, gtpdb_path, gap_nums=None, filter_names=None):
    '''gtur_vs_gtpdb'''
    prot_names = os.listdir(gtur_path)
    prot_names = [name.split(".")[0] for name in prot_names]
    if gap_nums:
        prot_names = list(set(prot_names).intersection(set(list(gap_nums.keys()))))
    if filter_names:
        prot_names = list(set(prot_names).intersection(set(list(filter_names))))
    prot_names.sort()
    outputs = []
    for prot_name in prot_names:

        if gap_nums and prot_name in gap_nums:
            num_gap = gap_nums[prot_name]
        else:
            num_gap = 0
        local_pdb_path = os.path.join(gtpdb_path, prot_name, prot_name + ".pdb")
        with open(os.path.join(gtur_path, f"{prot_name}.pkl"), "rb") as f:
            restraints = pickle.load(f)

        restraints_new = preprocess_restraints(restraints, num_gap=num_gap, index_distance_threshold=0)
        stats_0 = check_restraints(local_pdb_path, restraints_new, 6)

        restraints_new = preprocess_restraints(restraints, num_gap=num_gap, index_distance_threshold=4)
        stats_4 = check_restraints(local_pdb_path, restraints_new, 6)

        outputs.append([prot_name] + list(stats_0) + list(stats_4))

    return np.array(outputs)


def predur_vs_gtpdb(predur_path, gtpdb_path, filter_names=None):
    '''predur_vs_gtpdb'''
    prot_names = os.listdir(predur_path)
    prot_names = [name.split(".")[0] for name in prot_names]
    if filter_names:
        prot_names = list(set(prot_names).intersection(set(list(filter_names))))
    prot_names.sort()

    outputs = []
    for prot_name in prot_names:
        if prot_name in ["2K0M"]:
            continue

        local_pdb_path = os.path.join(gtpdb_path, prot_name, prot_name + ".pdb")

        local_ur_path = os.path.join(predur_path, prot_name + ".pkl")

        with open(local_ur_path, "rb") as f:
            restraints_ori = pickle.load(f)

        restraints = []
        restraints_ori.sort()

        for res1, candidates in restraints_ori:
            i, atype1 = res1
            for res2 in candidates:
                j, atype2 = res2
                restraints.append([i, atype1, j, atype2])

        restraints_new = preprocess_restraints(restraints, index_distance_threshold=0)
        stats_0 = check_restraints(local_pdb_path, restraints_new, 6)

        restraints_new = preprocess_restraints(restraints, index_distance_threshold=4)
        stats_4 = check_restraints(local_pdb_path, restraints_new, 6)

        outputs.append([prot_name] + list(stats_0) + list(stats_4))

    return np.array(outputs)


def gtur_vs_predpdb(gtur_path, predpdb_path, gap_nums=None, filter_names=None):
    '''gtur_vs_predpdb'''
    ur_path = gtur_path
    pdb_path = predpdb_path

    all_pdb_names = os.listdir(pdb_path)
    pdb_names_dict = {}
    for pdb_name in all_pdb_names:
        short_name = pdb_name.split("_")[0]
        pdb_full_path = os.path.join(pdb_path, pdb_name)
        pdb_names_dict[short_name] = pdb_names_dict.get(short_name, []) + [pdb_full_path]

    prot_names = os.listdir(ur_path)
    prot_names = [name.split(".")[0] for name in prot_names]

    prot_names = list(set(prot_names).intersection(set(list(pdb_names_dict.keys()))))
    if gap_nums:
        prot_names = list(set(prot_names).intersection(set(list(gap_nums.keys()))))
    if filter_names:
        prot_names = list(set(prot_names).intersection(set(list(filter_names))))

    prot_names.sort()

    outputs = []
    for prot_name in prot_names:

        if gap_nums and prot_name in gap_nums:
            num_gap = gap_nums[prot_name]
        else:
            num_gap = 0

        local_pdb_path, conf_max, conf_median = select_pdb_by_conf(pdb_names_dict.get(prot_name), return_conf=True)

        with open(os.path.join(ur_path, f"{prot_name}.pkl"), "rb") as f:
            restraints = pickle.load(f)

        restraints_new = preprocess_restraints(restraints, num_gap=num_gap, index_distance_threshold=0)
        stats_0 = check_restraints(local_pdb_path, restraints_new, 6)

        restraints_new = preprocess_restraints(restraints, num_gap=num_gap, index_distance_threshold=4)
        stats_4 = check_restraints(local_pdb_path, restraints_new, 6)
        outputs.append([prot_name] + list(stats_0) + list(stats_4) + [conf_max, conf_median])

    return np.array(outputs)


def predur_vs_predpdb(predur_path, predpdb_path, filter_names=None, return_conf=False):
    '''predur_vs_predpdb'''
    ur_path = predur_path
    pdb_path = predpdb_path

    all_pdb_names = os.listdir(pdb_path)
    print(all_pdb_names)
    pdb_names_dict = {}
    for pdb_name in all_pdb_names:
        short_name = pdb_name.split("_")[0]
        pdb_full_path = os.path.join(pdb_path, pdb_name)
        pdb_names_dict[short_name] = pdb_names_dict.get(short_name, []) + [pdb_full_path]

    prot_names = os.listdir(ur_path)
    prot_names = [name.split(".")[0] for name in prot_names]
    print("assign=", prot_names, filter_names, pdb_names_dict.keys())
    if filter_names:
        prot_names = list(set(prot_names).intersection(set(list(filter_names))))
    prot_names = list(set(prot_names).intersection(set(list(pdb_names_dict.keys()))))
    prot_names.sort()

    outputs = []
    conf_all = {}
    print("after===", prot_names)
    for prot_name in prot_names:
        print(prot_name)

        local_pdb_path, _, conf_median = select_pdb_by_conf(pdb_names_dict.get(prot_name), return_conf=True)
        local_ur_path = os.path.join(ur_path, prot_name + ".pkl")
        with open(local_ur_path, "rb") as f:
            restraints_ori = pickle.load(f)

        restraints = []
        restraints_ori.sort()
        for res1, candidates in restraints_ori:
            i, atype1 = res1
            for res2 in candidates:
                j, atype2 = res2
                restraints.append([i, atype1, j, atype2])

        restraints_new = preprocess_restraints(restraints, index_distance_threshold=0)
        stats_0 = check_restraints(local_pdb_path, restraints_new, 6)
        restraints_new = preprocess_restraints(restraints, index_distance_threshold=4)
        stats_4 = check_restraints(local_pdb_path, restraints_new, 6)
        outputs.append([prot_name] + list(stats_0) + list(stats_4))
        conf_all[prot_name] = conf_median
    if return_conf:
        return np.array(outputs), conf_all
    return np.array(outputs)


def filter_ur_with_pdb(restraints, pdb_path, distance_threshold=12, pdb_align=1):
    '''filter_ur_with_pdb'''
    restraints_new = []
    for res1, candidates in restraints:
        i, atype1 = res1
        for res2 in candidates:
            j, atype2 = res2
            restraints_new.append([i, atype1, j, atype2])

    restraints_new = preprocess_restraints(restraints_new, 0)
    stats = check_restraints(pdb_path, restraints_new, distance_threshold, pdb_align, return_new_restraints=True)
    ur_tuple_filtered = stats[-1]

    return ur_tuple_filtered

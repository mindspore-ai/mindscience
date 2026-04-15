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
"init_assign"
import copy
import os
import stat
import pickle
import numpy as np

from commons.res_constants import EQ_GROUPS, EQUI_VARIANCE, AA_3TO1, \
    atom_template, peak_template


def get_ur_list(peak_list, long_distance_threshold=0):
    '''get_ur_list'''
    new_peaks = copy.deepcopy(peak_list)
    ur_list_tuple = []
    ur_list = []
    for peak in new_peaks:
        if peak["active"] < 0.5:
            continue
        ori_contributions = peak.get("analysis").get("contributions")
        new_contributions = []
        for contribution in ori_contributions:
            if contribution["weight"] > 0:
                new_contributions.append(contribution)
        peak.get("analysis")["contributions"] = new_contributions
        if len(new_contributions) == 1:
            res_idx1 = new_contributions[0].get("spin_pairs")[0].get("Atom1").get("res")
            res_idx2 = new_contributions[0].get("spin_pairs")[0].get("Atom2").get("res")
            atype1 = new_contributions[0].get("spin_pairs")[0].get("Atom1").get("name")
            atype2 = new_contributions[0].get("spin_pairs")[0].get("Atom2").get("name")
            if abs(res_idx1 - res_idx2) < long_distance_threshold:
                continue
            ur_list.append([[res_idx1, atype1], [[res_idx2, atype2]]])
            if (res_idx1, res_idx2) not in ur_list and (res_idx2, res_idx1) not in ur_list_tuple:
                ur_list_tuple.append((res_idx1, res_idx2))
                ur_list.append([[res_idx1, atype1], [[res_idx2, atype2]]])
    return ur_list, ur_list_tuple


def ppm_shift(chem_shift, lb=-999.0,
              ub=999.0):
    return chem_shift - np.floor((chem_shift - lb) / (ub - lb)) * (ub - lb)


def group_by_peak_num(peak_adrs):
    '''group_by_peak_num'''
    peak_adrs_grouped = {}
    for peak in peak_adrs:
        peak2 = copy.deepcopy(peak)
        peak2[2] = list(peak2[2])[0]
        peak2[4] = list(peak2[4])[0]
        peak2[7] = list(peak2[7])[0]
        for value in peak2:
            value = str(value)
        peak_num = int(peak2[0])
        if peak_num not in peak_adrs_grouped:
            peak_adrs_grouped[peak_num] = [tuple(peak2[1:])]
        else:
            peak_adrs_grouped.get(peak_num).append(tuple(peak2[1:]))

    return peak_adrs_grouped


def init_assign(cn_noe, noe_atype, atom_types, chem_shifts, atom_names, res_types, res_idxs, windows=None):
    '''init_assign'''
    if noe_atype not in ["C", "N"]:
        raise ValueError("Only C or N are supported as noe_atype")

    if not windows:
        window_cn = [-999.0, 999.0]
        window_h1 = [-999.0, 999.0]
        window_h2 = [-999.0, 999.0]
    else:
        window_cn = windows[0]
        window_h1 = windows[1]
        window_h2 = windows[2]

    spect_window = window_h2[1] - window_h2[0]
    cn_shift_idxs = np.where([atom_type == noe_atype for atom_type in atom_types])[0]
    peak_adrs = []
    pair_num = []

    for i, _ in enumerate(cn_shift_idxs):
        cn_shift_ori = chem_shifts[cn_shift_idxs[i]]
        cn_shift = ppm_shift(cn_shift_ori, window_cn[0], window_cn[1])
        cn_name = atom_names[cn_shift_idxs[i]]
        cn_eq_idx = np.where(
            (EQ_GROUPS[:, 0] == AA_3TO1[res_types[cn_shift_idxs[i]]]) * \
            ([cn_name in eqg for eqg in EQ_GROUPS[:, 2]]))[0]
        if cn_eq_idx is None:
            continue
        cn_res_idx = res_idxs[cn_shift_idxs[i]]
        h1_eq_group = EQ_GROUPS[cn_eq_idx[0]][3]
        h1_idx = np.where((res_idxs == cn_res_idx) * ([atom_name in h1_eq_group for atom_name in atom_names]))[
            0]
        h1_shift = ppm_shift(chem_shifts[h1_idx], window_h1[0], window_h1[1])
        if h1_shift.shape[0] == 0:
            continue
        cn_h_pair_idx = np.where((np.abs(cn_noe[:, 0] - cn_shift) < 0.2) * ( \
                    np.min(np.abs(cn_noe[:, 2][None] - h1_shift[:, None]), axis=0) < 0.02))[0]
        pair_num.append(len(cn_h_pair_idx))
        for k, _ in enumerate(cn_h_pair_idx):
            h2_shift = np.array([cn_noe[cn_h_pair_idx[k], 1], cn_noe[cn_h_pair_idx[k], 1] + spect_window,
                                 cn_noe[cn_h_pair_idx[k], 1] - spect_window])
            if h2_shift.shape[0] == 0:
                continue
            h2_idxs = \
                np.where((atom_types == 'H') * (np.min(np.abs(chem_shifts[None] - h2_shift[:, None]), axis=0) < 0.02))[
                    0]
            for l, _ in enumerate(h2_idxs):
                h2_eq_group = EQ_GROUPS[(EQ_GROUPS[:, 0] == AA_3TO1[res_types[h2_idxs[l]]]) * (
                    [atom_names[h2_idxs[l]] in eqg for eqg in EQ_GROUPS[:, 3]])][0][3]
                peak_adrs.append(
                    [cn_h_pair_idx[k], cn_shift_idxs[i], EQ_GROUPS[cn_eq_idx[0]][2], h1_idx[0], h1_eq_group,
                     res_types[h1_idx[0]], h2_idxs[l], h2_eq_group, res_types[h2_idxs[l]]])

                cn_res_idx = cn_res_idx

    peak_adrs_uniq = []
    for peak_single in peak_adrs:
        if peak_single not in peak_adrs_uniq:
            peak_adrs_uniq.append(peak_single)

    peak_adrs_uniq = np.array(peak_adrs_uniq)

    peak_adrs_grouped = group_by_peak_num(peak_adrs_uniq)

    return peak_adrs_grouped


def make_atom(name, res, atom_id, hetero_name, types, restype):
    '''make_atom'''
    atom = copy.deepcopy(atom_template)
    atom["name"] = name
    atom["res"] = res
    atom["atom_id"] = atom_id
    atom["hetero_name"] = hetero_name
    atom["type"] = types
    atom["restype"] = restype
    return atom


def make_peak_list(peak_adrs, noe, res_idxs, noe_list_percentile=None):
    '''make_peak_list'''
    peak_list = []

    spin_pair_id = 0

    num_contributions = []
    for peak_num, assignments in peak_adrs.items():

        peak_chem_shifts = noe[peak_num]

        peak = copy.deepcopy(peak_template)

        peak["peak_id"] = peak_num

        proton1assignments = []
        hetero1assignments = []
        proton2assignments = []

        existed_equi_atoms = []

        for assignment in np.unique(np.array(assignments)[:, 1:5], axis=0):
            # merge equivariance
            res_idx = res_idxs[int(assignment[1])]
            aname = assignment[2]
            if [res_idx, aname] in existed_equi_atoms:
                continue
            cur_equivariance = EQUI_VARIANCE.get(assignment[3]).get(aname).get("equivariance")
            existed_equi_atoms.extend([[res_idx, aname] for aname in cur_equivariance])
            proton1 = make_atom(name=aname,
                                res=res_idx,
                                atom_id=int(assignment[1]),
                                hetero_name=assignment[0],
                                restype=assignment[3],
                                types="H")
            proton1assignments.append({
                'type': 'automatic',
                'atoms': [proton1]
            })

        existed_equi_atoms = []
        for assignment in np.unique(np.array(assignments)[:, 5:8], axis=0):
            res_idx = res_idxs[int(assignment[0])]
            aname = assignment[1]

            if [res_idx, aname] in existed_equi_atoms:
                continue
            cur_equivariance = EQUI_VARIANCE.get(assignment[2]).get(aname).get("equivariance")
            existed_equi_atoms.extend([[res_idx, aname] for aname in cur_equivariance])

            proton2 = make_atom(name=aname,
                                res=res_idxs[int(assignment[0])],
                                atom_id=int(assignment[0]),
                                hetero_name="N",
                                restype=assignment[2],
                                types="H")
            proton2assignments.append({
                'type': 'automatic',
                'atoms': [proton2]
            })
        for assignment in np.unique(np.array(assignments)[:, [0, 1, 3]], axis=0):
            aname = assignment[1]
            hetero1 = make_atom(name=aname,
                                res=res_idxs[int(assignment[0])],
                                atom_id=int(assignment[0]),
                                restype=assignment[2],
                                hetero_name=None,
                                types=aname)
            hetero1assignments.append({
                'type': 'automatic',
                'atoms': [hetero1]
            })

        peak.get("ref_peak")["proton1assignments"] = proton1assignments
        peak.get("ref_peak")["hetero1assignments"] = hetero1assignments
        peak.get("ref_peak")["proton2assignments"] = proton2assignments
        peak.get("ref_peak")['volume'] = [peak_chem_shifts[3], None]
        peak.get("ref_peak")['intensity'] = [peak_chem_shifts[3], None]

        if noe_list_percentile and peak_chem_shifts[3] > noe_list_percentile:
            # if :
            continue

        peak.get("ref_peak")["proton1ppm"] = [peak_chem_shifts[2], None]
        peak.get("ref_peak")["proton2ppm"] = [peak_chem_shifts[1], None]
        peak.get("ref_peak")["hetero1ppm"] = [peak_chem_shifts[0], None]

        contributions = []
        for proton1 in proton1assignments:
            for proton2 in proton2assignments:
                if proton1["atoms"][0]["name"] in \
                        EQUI_VARIANCE.get(proton2.get("atoms")[0].get("restype")).get( \
                                proton2.get("atoms")[0].get("name")).get("equivariance"):
                    continue
                contribution = {
                    'figure_of_merit': None,
                    'weight': 1.0,
                    'average_distance': [None, None],
                    'contribution_id': spin_pair_id,
                    'type': 'fast_exchange',
                    'spin_pairs': [{'id': spin_pair_id,
                                    'Atom1': proton1["atoms"][0],
                                    'Atom2': proton2["atoms"][0],
                                    }],
                }

                contributions.append(contribution)
                spin_pair_id += 1
        num_contributions.append(len(contributions))
        if not contributions:
            continue
        peak.get("analysis")["contributions"] = copy.deepcopy(contributions)

        peak_list.append(peak)
    return peak_list


def get_ur_list2(peak_list, long_distance_threshold=0):
    '''get_ur_list2'''
    new_peaks = copy.deepcopy(peak_list)
    ur_list_tuple = []
    ur_list = []
    for peak in new_peaks:
        if peak.get("active") < 0.5:
            continue
        ori_contributions = peak.get("analysis").get("contributions")
        new_contributions = []
        for contribution in ori_contributions:
            if contribution["weight"] > 0:
                new_contributions.append(contribution)
        peak.get("analysis")["contributions"] = new_contributions
        if len(new_contributions) == 1:
            res_idx1 = new_contributions[0].get("spin_pairs")[0].get("Atom1").get("res")
            res_idx2 = new_contributions[0].get("spin_pairs")[0].get("Atom2").get("res")
            atype1 = new_contributions[0].get("spin_pairs")[0].get("Atom1").get("name")
            atype2 = new_contributions[0].get("spin_pairs")[0].get("Atom2").get("name")
            if abs(res_idx1 - res_idx2) < long_distance_threshold:
                continue
            ur_list.append([[res_idx1, atype1], [[res_idx2, atype2]]])
            if (res_idx1, res_idx2) not in ur_list and (res_idx2, res_idx1) not in ur_list_tuple:
                ur_list_tuple.append((res_idx1, res_idx2))
                ur_list.append([[res_idx1, atype1], [[res_idx2, atype2]]])
    return ur_list, ur_list_tuple


def load_noelist_from_txt(noe_file):
    '''load_cs_from_txt'''
    with open(noe_file, "r") as f:
        data_txt_load = f.readlines()

    noelist = []
    for line in data_txt_load:
        words = line.split()
        try:
            noelist.append([float(words[0]), float(words[1]), float(words[2]), float(words[3])])
        except ValueError:
            continue
    noelist = np.array(noelist)

    return noelist


def load_cs_from_txt(cs_file):
    '''load_cs_from_txt'''
    with open(cs_file, "r") as f:
        data_txt_load = f.readlines()

    atom_names = []
    atom_types = []
    chem_shifts = []
    res_idxs = []
    res_types = []

    for line in data_txt_load:
        words = line.split()
        try:
            atom_name = words[0]
            atom_type = words[1]
            chem_shift = float(words[2])
            res_idx = int(words[3])
            res_type = words[4]
        except ValueError:
            continue
        atom_names.append(atom_name)
        atom_types.append(atom_type)
        chem_shifts.append(chem_shift)
        res_idxs.append(res_idx)
        res_types.append(res_type)

    res = [atom_names, atom_types, chem_shifts, res_idxs, res_types]
    res = [np.array(array) for array in res]
    return res


def init_assign_call(prot_path):
    '''init_assign_call'''
    prot_name = prot_path.split("/")[-1]
    file_list = os.listdir(prot_path)
    noe_file_list = []
    for file in file_list:
        if file.split("/")[-1].startswith("noelist_"):
            noe_file_list.append(file)

    cs_file_path = os.path.join(prot_path, "chemical_shift_aligned.txt")
    atom_names, atom_types, chem_shifts, res_idxs, res_types = load_cs_from_txt(cs_file_path)

    all_peak_list = []
    for file_id, noe_file in enumerate(noe_file_list):
        noe_file_path = os.path.join(prot_path, noe_file)
        noe_list = load_noelist_from_txt(noe_file_path)

        noe_list_25percentile = np.percentile(noe_list[:, -1], 100)

        windows = None
        peak_adrs_grouped_asc = init_assign(cn_noe=noe_list,
                                            noe_atype="C",
                                            atom_types=atom_types,
                                            chem_shifts=chem_shifts,
                                            atom_names=atom_names,
                                            res_types=res_types,
                                            res_idxs=res_idxs,
                                            windows=windows
                                            )

        peak_adrs_grouped_asn = init_assign(cn_noe=noe_list,
                                            noe_atype="N",
                                            atom_types=atom_types,
                                            chem_shifts=chem_shifts,
                                            atom_names=atom_names,
                                            res_types=res_types,
                                            res_idxs=res_idxs,
                                            windows=windows
                                            )
        if len(peak_adrs_grouped_asc) > len(peak_adrs_grouped_asn):
            peak_adrs_grouped = peak_adrs_grouped_asc
            noe_atype = "C"
        else:
            noe_atype = "N"
            peak_adrs_grouped = peak_adrs_grouped_asn

        spectrum = make_peak_list(peak_adrs=peak_adrs_grouped,
                                  noe=noe_list,
                                  res_idxs=res_idxs,
                                  noe_list_percentile=noe_list_25percentile)

        print(noe_atype, " " * 5, prot_name, noe_file, len(noe_list), len(peak_adrs_grouped))
        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        with os.fdopen(os.open(prot_path + f'/new_spectrum_{noe_atype}_{file_id}.pkl', os_flags, os_modes), "wb") as f:
            pickle.dump(spectrum, f)
        all_peak_list += spectrum

    ur_list, ur_list_tuple = get_ur_list2(all_peak_list, long_distance_threshold=0)

    return ur_list, ur_list_tuple

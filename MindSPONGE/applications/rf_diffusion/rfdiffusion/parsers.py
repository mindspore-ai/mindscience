# Modified from RFdiffusion (https://github.com/RosettaCommons/RFdiffusion)
# Original license: BSD License
#
# Copyright 2025 Huawei Technologies Co., Ltd
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

import os
import random

import numpy as np
import pandas as pd

from .chemical import aa2long, aa2num

to1letter = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def HLT_pdb_parser(path):
    """
    inputs:
        - path | string of path to chothia labeled pdb
        - path to summary file (optional)
    outputs:
        - out | dictionary including, xyz, seq, pdb_idx, cdr_bool
    Adapted from Joe's chothia parser

    NB this function is able to handle files with repeated pdb residue indices
    """

    cdr_names = ["H1", "H2", "H3", "L1", "L2", "L3"]

    with open(path, "r") as f:
        lines = f.readlines()

    # indices of residues observed in the structure
    res = []
    for l in lines:
        if l[:4] != "ATOM":
            continue

        i = 0 if l[11] == " " else 1
        if l[12 + i : 16 + i].strip() == "CA":
            res.append(
                (l[22 + i : 26 + i].strip(), l[17 + i : 20 + 1].strip(), l[21 + i])
            )

    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [
        (l[21:22].strip(), l[22:26].strip())
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]  # chain letter, res num

    loop_masks = {loop: np.zeros(len(res)).astype(bool) for loop in cdr_names}

    # 4 BB + 23 SC atoms
    xyz = np.full((len(res), 27, 3), np.nan, dtype=np.float32)
    for l in lines:
        # Check for lines that begin with REMARK and parse them to their loop labels
        if l[:3] == "TER":
            continue
        if l[:6] == "REMARK":
            loop = l[27:29].upper()
            if loop in cdr_names:
                resi = int(l[21:26]) - 1  # Loop residues in HLT are 1-indexed
                loop_masks[loop][resi] = True
            continue
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = (
            l[21:22].strip(),
            l[22:26].strip(),
            " " + l[12:16].strip().ljust(3),
            l[17:20],
        )
        if (chain, resNo) not in pdb_idx:
            continue
        idx = pdb_idx.index((chain, resNo))
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if (
                tgtatm is not None and tgtatm.strip() == atom.strip()
            ):  # ignore whitespace
                xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    out = {
        "xyz": xyz,  # cartesian coordinates, [Lx14]
        # mask showing which atoms are present in the PDB file, [Lx14]
        "mask": mask,
        "idx": np.array(
            [i[1] for i in pdb_idx]
        ),  # residue numbers in the PDB file, [L]
        "seq": np.array(seq),  # amino acid sequence, [L]
        # list of (chain letter, residue number) in the pdb file, [L]
        "pdb_idx": pdb_idx,
        # dict of [L] masks indicating which residue belongs to which loop
        "loop_masks": loop_masks,
    }
    return out


def chothia_pdb_parser(
    path, summary_path=None, chains=None, expanded_loop_def=False, rand_loop_ext=0
):
    """
    inputs:
        - path | string of path to chothia labeled pdb
        - path to summary file (optional)
    outputs:
        - out | dictionary including, xyz, seq, pdb_idx, cdr_bool
    Adapted from Jake's parser. cdr_bool is now for *all* Ab chains in a pdb file
    """

    with open(path, "r") as f:
        lines = f.readlines()

    if chains is None:
        if summary_path is None:
            summary_path = "/databases/antibody_2023JAN13/sabdab_summary_all.tsv"
            if not os.path.exists(summary_path):
                summary_path = "/net/databases/antibody/sabdab_summary_all.tsv"

        df = pd.read_csv(summary_path, sep="\t")
        # load current path from sabdab summary file
        tmp = df[df["pdb"] == path[-8:-4]]
        h_chains = tmp["Hchain"].tolist()
        l_chains = tmp["Lchain"].tolist()

        if h_chains == l_chains:
            print("Chains are the same")
            return None
    else:
        h_chains = chains["H"]
        l_chains = chains["L"]

    # indices of residues observed in the structure
    res = []
    for l in lines:
        i = 0 if l[11] == " " else 1
        if l[:4] == "ATOM" and l[12 + i : 16 + i].strip() == "CA":
            res.append(
                (l[22 + i : 27 + i].strip(), l[17 + i : 20 + 1].strip(), l[21 + i])
            )

    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [
        (l[21:22].strip(), l[22:27].strip())
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]  # chain letter, res num

    # 4 BB + 23 SC atoms
    xyz = np.full((len(res), 27, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = (
            l[21:22].strip(),
            l[22:27].strip(),
            " " + l[12:16].strip().ljust(3),
            l[17:20],
        )
        if (chain, resNo) not in pdb_idx:
            continue
        idx = pdb_idx.index((chain, resNo))
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if (
                tgtatm is not None and tgtatm.strip() == atom.strip()
            ):  # ignore whitespace
                xyz[idx, i_atm, :] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    cdr_chothia = {
        "L1": (24, 34),
        "L2": (50, 56),
        "L3": (89, 97),
        "H1": (26, 32),
        "H2": (52, 56),
        "H3": (95, 102),
    }
    if expanded_loop_def:
        # Expand the definition of CDR loops by 3 residues in each direction
        cdr_chothia = {
            key: (cdr_chothia[key][0] - 3, cdr_chothia[key][1] + 3)
            for key in cdr_chothia
        }

    if rand_loop_ext > 0:
        # Randomly extend each loop start and end by uniform(0, rand_loop_ext) residues
        for key in cdr_chothia:
            rand_start = random.randint(0, rand_loop_ext)
            rand_end = random.randint(0, rand_loop_ext)

            cdr_chothia[key] = (
                cdr_chothia[key][0] - rand_start,
                cdr_chothia[key][1] + rand_end,
            )

    cdrs_bool = np.zeros(len(pdb_idx))
    for i, l in enumerate(pdb_idx):
        iscdr = False
        if l[1][-1].isalpha():
            idx_i = int(l[1][:-1])
        else:
            idx_i = int(l[1])

        if l[0] in l_chains:
            if idx_i in range(cdr_chothia["L1"][0], cdr_chothia["L1"][1] + 1):
                iscdr = True
            elif idx_i in range(cdr_chothia["L2"][0], cdr_chothia["L2"][1] + 1):
                iscdr = True
            elif idx_i in range(cdr_chothia["L3"][0], cdr_chothia["L3"][1] + 1):
                iscdr = True
        elif l[0] in h_chains:
            if idx_i in range(cdr_chothia["H1"][0], cdr_chothia["H1"][1] + 1):
                iscdr = True
            elif idx_i in range(cdr_chothia["H2"][0], cdr_chothia["H2"][1] + 1):
                iscdr = True
            elif idx_i in range(cdr_chothia["H3"][0], cdr_chothia["H3"][1] + 1):
                iscdr = True
        if iscdr:
            cdrs_bool[i] = 1

    out = {
        "xyz": xyz,  # cartesian coordinates, [Lx14]
        # mask showing which atoms are present in the PDB file, [Lx14]
        "mask": mask,
        "idx": np.array(
            [i[1] for i in pdb_idx]
        ),  # residue numbers in the PDB file, [L]
        "seq": np.array(seq),  # amino acid sequence, [L]
        # list of (chain letter, residue number) in the pdb file, [L]
        "pdb_idx": pdb_idx,
        "cdr_bool": [True if i == 1 else False for i in cdrs_bool],
    }
    return out

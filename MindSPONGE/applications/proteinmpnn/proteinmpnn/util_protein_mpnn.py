# Modified from ProteinMPNN (https://github.com/dauparas/ProteinMPNN)
# Original license: MIT License
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
"""
Util functions for ProteinMPNN
"""

import os
import copy
import pickle
from dataclasses import dataclass
from typing import List, Optional

import mindspore as ms
import numpy as np
from mindspore import ops

from .protein_mpnn import ProteinMPNN, _S_to_seq, _scores, tied_featurize

num2aa = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
"LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","UNK","MAS"]

aa2num = {x: i for i, x in enumerate(num2aa)}

# full sc atom representation (Nx14)
aa2long = [
    (" N "," CA "," C  "," O  "," CB ",None,None,None,None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","3HB ",None,None,None,None,None,None,None,None,), # ala
    (" N "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",None,None,None,
     " H "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD "," HE ","1HH1","2HH1","1HH2","2HH2",), # arg
    (" N "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","1HD2","2HD2",None,None,None,None,None,None,None,), # asn
    (" N "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB ",None,None,None,None,None,None,None,None,None,), # asp
    (" N "," CA "," C  "," O  "," CB "," SG ",None,None,None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB "," HG ",None,None,None,None,None,None,None,None,), # cys
    (" N "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","1HG ","2HG ","1HE2","2HE2",None,None,None,None,None,), # gln
    (" N "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","1HG ","2HG ",None,None,None,None,None,None,None,), # glu
    (" N "," CA "," C  "," O  ",None,None,None,None,None,None,None,None,None,None,
     " H ","1HA ","2HA ",None,None,None,None,None,None,None,None,None,None,), # gly
    (" N "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",None,None,None,None,
     " H "," HA ","1HB ","2HB "," HD2"," HE1"," HE2",None,None,None,None,None,None,), # his
    (" N "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",None,None,None,None,None,None,
     " H "," HA "," HB ","1HG2","2HG2","3HG2","1HG1","2HG1","1HD1","2HD1","3HD1",None,None,), # ile
    (" N "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB "," HG ","1HD1","2HD1","3HD1","1HD2","2HD2","3HD2",None,None,), # leu
    (" N "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ","1HE ","2HE ","1HZ ","2HZ ","3HZ ",), # lys
    (" N "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","1HG ","2HG ","1HE ","2HE ","3HE ",None,None,None,None,), # met
    (" N "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",None,None,None,
     " H "," HA ","1HB ","2HB "," HD1"," HD2"," HE1"," HE2"," HZ ",None,None,None,None,), # phe
    (" N "," CA "," C  "," O  "," CB "," CG "," CD ",None,None,None,None,None,None,None,
     " HA ","1HB ","2HB ","1HG ","2HG ","1HD ","2HD ",None,None,None,None,None,None,), # pro
    (" N "," CA "," C  "," O  "," CB "," OG ",None,None,None,None,None,None,None,None,
     " H "," HG "," HA ","1HB ","2HB ",None,None,None,None,None,None,None,None,), # ser
    (" N "," CA "," C  "," O  "," CB "," OG1"," CG2",None,None,None,None,None,None,None,
     " H "," HG1"," HA "," HB ","1HG2","2HG2","3HG2",None,None,None,None,None,None,), # thr
    (" N "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2",
     " H "," HA ","1HB ","2HB "," HD1"," HE1"," HZ2"," HH2"," HZ3"," HE3",None,None,None,), # trp
    (" N "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",None,None,
     " H "," HA ","1HB ","2HB "," HD1"," HE1"," HE2"," HD2"," HH ",None,None,None,None,), # tyr
    (" N "," CA "," C  "," O  "," CB "," CG1"," CG2",None,None,None,None,None,None,None,
     " H "," HA "," HB ","1HG1","2HG1","3HG1","1HG2","2HG2","3HG2",None,None,None,None,), # val
    (" N "," CA "," C  "," O  "," CB ",None,None,None,None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","3HB ",None,None,None,None,None,None,None,None,), # unk
    (" N "," CA "," C  "," O  "," CB ",None,None,None,None,None,None,None,None,None,
     " H "," HA ","1HB ","2HB ","3HB ",None,None,None,None,None,None,None,None,), # mask
]

#################################
# Function Definitions
#################################


def my_rstrip(string, strip):
    """
    Remove the trailing strip from a string.
    
    Args:
        string (str): String.
        strip (str): Strip.
        
    Returns:
        str: String without the trailing strip.
    """
    if string.endswith(strip):
        return string[: -len(strip)]
    return string


# PDB Parse Util Functions

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
         "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","GAP"]

aa_1_N = {a: n for n, a in enumerate(alpha_1)}
aa_3_N = {a: n for n, a in enumerate(alpha_3)}
aa_N_1 = dict(enumerate(alpha_1))
aa_1_3 = dict(zip(alpha_1, alpha_3))
aa_3_1 = dict(zip(alpha_3, alpha_1))


def AA_to_N(x):
    """
    Convert a sequence of amino acids to a sequence of indices.
    
    Args:
        x (str or list of str): Sequence of amino acids.
        
    Returns:
        list of int: Sequence of indices.
    """
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x)
    if x.ndim == 0:
        x = x[None]
    return [[aa_1_N.get(a, states - 1) for a in y] for y in x]


def N_to_AA(x):
    """
    Convert a sequence of indices to a sequence of amino acids.
    
    Args:
        x (list of int): Sequence of indices.
        
    Returns:
        list of str: Sequence of amino acids.
    """
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
    return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

# End PDB Parse Util Functions

def parse_PDB_biounits(x, atoms=None, chain=None):
    """
    Parse a PDB file and extract the biounits.

    Args:
        x (str): PDB filename.
        atoms (list of str, optional): Atoms to extract. Default: ["N", "CA", "C"].
        chain (str, optional): Chain to extract. Default: None.

    Returns:
        tuple: (length, atoms, coords=(x,y,z)), sequence
    """
    if atoms is None:
        atoms = ["N", "CA", "C"]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    with open(x, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()

            if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12 : 12 + 4].strip()
                    resi = line[17 : 17 + 3]
                    resn = line[22 : 22 + 5].strip()
                    x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1
                    else:
                        resa, resn = "", int(resn) - 1

                    min_resn = min(min_resn, resn)
                    max_resn = max(max_resn, resn)
                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}
                    if resa not in seq[resn]:
                        seq[resn][resa] = resi

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except TypeError:
        return "no_chain", "no_chain"

def get_chain_alphabet(input_chain_list=None):
    """
    Get the chain alphabet.

    Args:
        input_chain_list (list of str, optional): List of chains. Default: None.

    Returns:
        list of str: Chain alphabet.
    """
    if input_chain_list:
        return input_chain_list
    init_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    return chain_alphabet

def format4(x):
    """
    Format a number to 4 decimal places.

    Args:
        x (float): Number to format.

    Returns:
        str: Formatted number.
    """
    return np.format_float_positional(np.float32(x), unique=False, precision=4)

def seq_with_slashes(seq, masked_chain_length_list, masked_list):
    """
    Add slashes to a sequence.

    Args:
        seq (str): Sequence.
        masked_chain_length_list (list of int): List of masked chain lengths.
        masked_list (list of bool): List of masked indices.

    Returns:
        str: Sequence with slashes.
    """
    start = 0
    end = 0
    list_of_AAs = []
    for mask_l in masked_chain_length_list:
        end += mask_l
        list_of_AAs.append(seq[start:end])
        start = end
    seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
    l0 = 0
    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
        l0 += mc_length
        seq = seq[:l0] + "/" + seq[l0:]
        l0 += 1
    return seq

def forward_scores(model, X, S_in, mask, chain_M, chain_M_pos, residue_idx, chain_encoding_all, randn,
                   use_input_decoding_order=False, decoding_order=None):
    """
    Forward pass of the model to get scores.

    Args:
        model (ProteinMPNN): ProteinMPNN model.
        X (Tensor): Input coordinates.
        S_in (Tensor): Input sequence.
        mask (Tensor): Mask.
        chain_M (Tensor): Chain mask.
        chain_M_pos (Tensor): Chain mask for positive examples.
        residue_idx (Tensor): Residue indices.
        chain_encoding_all (Tensor): Chain encodings.
        randn (Tensor): Random noise.
        use_input_decoding_order (bool, optional): Whether to use input decoding order. Default: False.
        decoding_order (Tensor, optional): Decoding order. Default: None.

    Returns:
        tuple: (scores, global_scores, log_probs, mask_for_loss)
    """
    if use_input_decoding_order:
        log_probs = model(
            X,
            S_in,
            mask,
            chain_M * chain_M_pos,
            residue_idx,
            chain_encoding_all,
            randn,
            use_input_decoding_order=True,
            decoding_order=decoding_order,
        )
    else:
        log_probs = model(
            X,
            S_in,
            mask,
            chain_M * chain_M_pos,
            residue_idx,
            chain_encoding_all,
            randn,
        )
    mask_for_loss = mask * chain_M * chain_M_pos
    scores_np = _scores(S_in, log_probs, mask_for_loss).data.numpy()
    global_scores_np = _scores(S_in, log_probs, mask).data.numpy()
    return scores_np, global_scores_np, log_probs, mask_for_loss

def ensure_output_dirs(base_folder, save_score=False, score_only=False,
                       conditional_probs_only=False, unconditional_probs_only=False,
                       save_probs=False):
    """
    Ensure the output directories exist.

    Args:
        base_folder (str): Base folder.
        save_score (bool, optional): Whether to save scores. Default: False.
        score_only (bool, optional): Whether to save score only. Default: False.
        conditional_probs_only (bool, optional): Whether to save conditional probabilities only. Default: False.
        unconditional_probs_only (bool, optional): Whether to save unconditional probabilities only. Default: False.
        save_probs (bool, optional): Whether to save probabilities. Default: False.
    """
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if not os.path.exists(base_folder + "seqs"):
        os.makedirs(base_folder + "seqs")
    if save_score and not os.path.exists(base_folder + "scores"):
        os.makedirs(base_folder + "scores")
    if score_only and not os.path.exists(base_folder + "score_only"):
        os.makedirs(base_folder + "score_only")
    if conditional_probs_only and not os.path.exists(base_folder + "conditional_probs_only"):
        os.makedirs(base_folder + "conditional_probs_only")
    if unconditional_probs_only and not os.path.exists(base_folder + "unconditional_probs_only"):
        os.makedirs(base_folder + "unconditional_probs_only")
    if save_probs and not os.path.exists(base_folder + "probs"):
        os.makedirs(base_folder + "probs")
    return base_folder


def parse_PDB(x, atoms=None, chain=None):
    """
    Parse a PDB file and extract the coordinates and sequence.
    
    Args:
        x (str): PDB filename.
        atoms (list of str, optional): Atoms to extract. Default: ["N", "CA", "C"].
        chain (str, optional): Chain to extract. Default: None.
        
    Returns:
        tuple: (length, atoms, coords=(x,y,z)), sequence
    """
    if not atoms:
        atoms = ["N", "CA", "C"]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    with open(x, "r", encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip()

            if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12 : 12 + 4].strip()
                    resi = line[17 : 17 + 3]
                    resn = line[22 : 22 + 5].strip()
                    x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1
                    else:
                        resa, resn = "", int(resn) - 1

                    min_resn = min(min_resn, resn)
                    max_resn = max(max_resn, resn)
                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}
                    if resa not in seq[resn]:
                        seq[resn][resa] = resi

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    for resn in range(min_resn, max_resn + 1):
        if resn in seq:
            for k in sorted(seq[resn]):
                seq_.append(aa_3_N.get(seq[resn][k], 20))
        else:
            seq_.append(20)
        if resn in xyz:
            for k in sorted(xyz[resn]):
                for atom in atoms:
                    if atom in xyz[resn][k]:
                        xyz_.append(xyz[resn][k][atom])
                    else:
                        xyz_.append(np.full(3, np.nan))
        else:
            for atom in atoms:
                xyz_.append(np.full(3, np.nan))
    return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))


def generate_seqopt_features(pdbfile, chains):  # multichain
    """
    Generate sequence optimization features from a PDB file.
    
    Args:
        pdbfile (str): PDB filename.
        chains (list of str): Chains to extract.
        
    Returns:
        dict: Sequence optimization features.
    """
    my_dict = {}
    concat_seq = ""

    for letter in chains:
        xyz, seq = parse_PDB_biounits(
            pdbfile, atoms=["N", "CA", "C", "O"], chain=letter
        )

        concat_seq += seq[0]
        my_dict["seq_chain_" + letter] = seq[0]
        coords_dict_chain = {}
        coords_dict_chain["N_chain_" + letter] = xyz[:, 0, :].tolist()
        coords_dict_chain["CA_chain_" + letter] = xyz[:, 1, :].tolist()
        coords_dict_chain["C_chain_" + letter] = xyz[:, 2, :].tolist()
        coords_dict_chain["O_chain_" + letter] = xyz[:, 3, :].tolist()
        my_dict["coords_chain_" + letter] = coords_dict_chain

    my_dict["name"] = my_rstrip(pdbfile, ".pdb")
    my_dict["num_of_chains"] = len(chains)
    my_dict["seq"] = concat_seq

    return my_dict


def get_seq_from_pdb(pdb_fn, slash_for_chainbreaks):
    """
    Get the sequence from a PDB file.
    
    Args:
        pdb_fn (str): PDB filename.
        slash_for_chainbreaks (bool): Whether to use slash for chain breaks.
        
    Returns:
        str: Sequence.
    """
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

    seq = ""
    with open(pdb_fn, "r", encoding='utf-8') as fp:
        for line in fp:
            if line.startswith("TER"):
                if not slash_for_chainbreaks:
                    continue
                seq += "/"
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            resName = line[17:20]

            seq += to1letter[resName]
    return my_rstrip(seq, "/")


def init_seq_optimize_model(
    hidden_dim, num_layers, backbone_noise, num_connections, checkpoint_path
):
    """
    Initialize the sequence optimization model.
    
    Args:
        hidden_dim (int): Hidden dimension.
        num_layers (int): Number of layers.
        backbone_noise (float): Backbone noise.
        num_connections (int): Number of connections.
        checkpoint_path (str): Checkpoint path.
        
    Returns:
        ProteinMPNN: Sequence optimization model.
    """
    model = ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=num_connections,
    )
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_train(False)

    return model


def set_default_args(seq_per_target, omit_AAs=None):
    """
    Set default arguments for sequence optimization.
    
    Args:
        seq_per_target (int): Number of sequences per target.
        omit_AAs (list of str, optional): AAs to omit. Default: ["X"].
        
    Returns:
        dict: Default arguments.
    """
    if omit_AAs is None:
        omit_AAs = ["X"]

    if "X" not in omit_AAs:
        omit_AAs.append("X")  # We don't want any unknown residue assignments

    retval = {}
    retval["BATCH_COPIES"] = min(1, seq_per_target)
    retval["NUM_BATCHES"] = seq_per_target // retval["BATCH_COPIES"]
    retval["temperature"] = 0.1

    omit_AAs_list = omit_AAs
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    retval["omit_AAs_np"] = np.array([AA in omit_AAs_list for AA in alphabet]).astype(
        np.float32
    )

    retval["omit_AA_dict"] = None
    retval["pssm_dict"] = None
    retval["bias_AA_dict"] = None
    retval["tied_positions_dict"] = None
    retval["bias_by_res_dict"] = None
    retval["bias_AAs_np"] = np.zeros(len(alphabet))

    return retval


def generate_sequences(
    model,
    feature_dict,
    arg_dict,
    masked_chains,
    visible_chains,
    fixed_positions_dict=None,
):
    """
    Generate sequences using the sequence optimization model.
    
    Args:
        model (ProteinMPNN): Sequence optimization model.
        feature_dict (dict): Feature dictionary.
        arg_dict (dict): Argument dictionary.
        masked_chains (list of str): Masked chains.
        visible_chains (list of str): Visible chains.
        fixed_positions_dict (dict, optional): Fixed positions dictionary. Default: None.
        
    Returns:
        list of tuple: List of sequences and scores.
    """
    seqs_scores = []

    batch_clones = [
        copy.deepcopy(feature_dict) for i in range(arg_dict["BATCH_COPIES"])
    ]
    chain_id_dict = {
        feature_dict["name"]: (masked_chains, visible_chains)
    }  # Masked, visible is the order, I think - Nate

    (
        X,
        S,
        mask,
        _,
        chain_M,
        chain_encoding_all,
        _,
        _,
        _,
        _,
        chain_M_pos,
        omit_AA_mask,
        residue_idx,
        _,
        _,
        pssm_coef,
        pssm_bias,
        pssm_log_odds_all,
        bias_by_res_all,
        _,
    ) = tied_featurize(
        batch_clones,
        chain_id_dict,
        fixed_positions_dict,
        arg_dict["omit_AA_dict"],
        arg_dict["tied_positions_dict"],
        arg_dict["pssm_dict"],
        arg_dict["bias_by_res_dict"],
    )

    pssm_threshold = 0  # Nate is hardcoding this
    pssm_log_odds_mask = (
        pssm_log_odds_all > pssm_threshold
    ).float()  # 1.0 for true, 0.0 for false

    randn_1 = ms.mint.randn(chain_M.shape)
    log_probs = model(
        X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_1
    )
    mask_for_loss = mask * chain_M * chain_M_pos
    scores = _scores(S, log_probs, mask_for_loss)

    for _ in range(arg_dict["NUM_BATCHES"]):
        randn_2 = ms.mint.randn(chain_M.shape)

        sample_dict = model.sample(
            X,
            randn_2,
            S,
            chain_M,
            chain_encoding_all,
            residue_idx,
            mask=mask,
            temperature=arg_dict["temperature"],
            omit_AAs_np=arg_dict["omit_AAs_np"],
            bias_AAs_np=arg_dict["bias_AAs_np"],
            chain_M_pos=chain_M_pos,
            omit_AA_mask=omit_AA_mask,
            pssm_coef=pssm_coef,
            pssm_bias=pssm_bias,
            pssm_multi=0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=pssm_log_odds_mask,
            pssm_bias_flag=False,
            bias_by_res=bias_by_res_all,
        )

        S_sample = sample_dict["S"]

        # Compute scores
        log_probs = model(
            X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_2
        )
        mask_for_loss = mask * chain_M * chain_M_pos
        scores = _scores(S_sample, log_probs, mask_for_loss)
        scores = scores.data.numpy()

        for b_ix in range(arg_dict["BATCH_COPIES"]):
            seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
            score = scores[b_ix]

            seqs_scores.append((seq, score))

    return seqs_scores


@dataclass
class Pose:
    """
    A class to represent a protein pose.

    Attributes:
        atoms:
            A [L, 3, 3] tensor of backbone atom coordinates
        seq:
            A [L] tensor of amino acid residues in 3 letter format
        chain:
            A [L] tensor of chain identifiers
        cdr_dict:
            A dictionary of CDR indices, with indices starting at 1
    """

    atoms: np.ndarray  # [L, 3, 3] tensor of backbone atom coordinates
    seq: np.ndarray  # [L] tensor of amino acid residues in 3 letter format
    chain: np.ndarray  # [L] tensor of chain identifiers

    cdr_dict: dict[
        str, list[int]
    ]  # dictionary of CDR indices, with indices starting at 1

    @classmethod
    def from_pdb(cls, pdbfile: str) -> "Pose":
        """
        Load a pdb file into a Pose object

        Args:
            pdbfile:
                The path to the pdb file to load
        """

        with open(pdbfile, "r", encoding='utf-8') as f:
            pdblines = f.readlines()

        return cls.from_pdblines(pdblines)

    @classmethod
    def from_pdblines(cls, pdblines: List[str]) -> "Pose":
        """
        Create a Pose object from a list of pdb lines

        Args:
            pdblines:
                A list of pdb lines to parse
        """

        seq, pdb_idx, xyz = parse_pdblines(pdblines)

        # Parse to a backbone xyz tensor
        bb_xyz = xyz[:, :4, :]  # [L, 4, 3]

        # Convert the sequence from numbers to 3 letter amino acids
        seq = np.array([num2aa[i] for i in seq])

        # Get the chain identifiers, pdb_idx is a list of tuples (chain, resnum)
        chains = np.array([i[0] for i in pdb_idx])

        cdr_masks = get_cdr_masks_from_remarks(pdb_idx, pdblines)

        # Now turn the cdr_masks into a dict of cdr indices
        cdr_dict = {
            "H1": [],
            "H2": [],
            "H3": [],
            "L1": [],
            "L2": [],
            "L3": [],
        }

        for cdr, mask in cdr_masks.items():
            cdr_dict[cdr] = np.where(mask)[0].tolist()

        return cls(
            atoms=bb_xyz,
            seq=seq,
            chain=chains,
            cdr_dict=cdr_dict,
        )

    def assert_HLT(self) -> bool:
        """
        Check if the pose is currently in HLT order.

        Returns:
            True if the pose is in HLT order, False otherwise.
        """

        # We will collect the consecutive chains in the pose

        # Find the indices where the value changes
        change_indices = np.where(np.diff(self.chains) != 0)[0] + 1

        # Include the first element as it is always unique in this context
        unique_indices = np.insert(change_indices, 0, 0)

        # Get the consecutive unique chains
        unique_chains = self.chains[unique_indices]

        # Check two things about these chains:
        # 1. The chains must be unique ie. there are no dis-contiguous chains
        # 2. The chains must be in the order H, L, T. Here, either H or L but not both can be missing,
        #    but T must be present

        # Check 1
        if np.unique(unique_chains).size != unique_chains.size:
            return False

        # Check 2
        if "T" not in unique_chains:
            return False

        if "H" in unique_chains and "L" in unique_chains:
            return unique_chains == np.array(["H", "L", "T"])

        if "H" in unique_chains and "L" not in unique_chains:
            return unique_chains == np.array(["H", "T"])

        if "H" not in unique_chains and "L" in unique_chains:
            return unique_chains == np.array(["L", "T"])

        # If we get here something has gone wrong
        raise Exception(f"Unsupported combination of chains: {unique_chains} provided")

    def mutate_residue(
        self,
        chain: str,
        residx: int,
        newres: str,
    ) -> None:
        """
        Mutate a residue in a pose

        Args:
            chain:
                The chain identifier of the residue to mutate

            residx:
                The zero-indexed residue index of the residue to mutate

            newres:
                The new 3 letter residue name to assign to the specified residue
        """

        # Assert that the residue index is within the bounds of the chain
        assert self.chain[residx] == chain, (
            "Residue index is not in the specified chain"
        )

        # Assert that the new residue is a valid amino acid
        assert newres in num2aa, "Invalid amino acid"

        # Assign the new residue to the sequence
        self.seq[residx] = newres

    def dump_pdb(self, pdbfile: str) -> None:
        """
        Dump a Pose object to a pdb file

        Args:
            pdbfile:
                The path to the pdb file to write
        """

        pdblines = self.to_pdblines()

        with open(pdbfile, "w", encoding='utf-8') as f:
            f.writelines(pdblines)

    def to_pdblines(self) -> List[str]:
        """
        Convert a pose to a list of pdb lines

        Returns:
            A list of pdb lines representing the pose
        """

        # Convert the sequence back to numbers
        seq = np.array([aa2num[i] for i in self.seq])

        pdblines = ab_write_pdblines(
            atoms=self.atoms,
            seq=seq,
            chain_idx=self.chain,
            loop_map=self.cdr_dict,
        )

        return pdblines


def stamp_pdbline(
    prefix: str,
    ctr: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_idx: int,
    x_coord: float,
    y_coord: float,
    z_coord: float,
    occupancy: float,
    b_factor: float,
) -> str:
    """
    Args:
        prefix:
            The prefix to use for the pdb line, e.g. "ATOM" or "HETATM"
        ctr:
            The atom counter to use for the pdb line
        atom_name:
            The name of the atom, e.g. " CA "
        residue_name:
            The name of the residue, e.g. "ALA"
        chain:
            The chain identifier, e.g. "A"
        residue_idx:
            The zero-indexed residue index, e.g. 1
        x_coord:
            The x coordinate of the atom, e.g. 1.0
        y_coord:
            The y coordinate of the atom, e.g. 2.0
        z_coord:
            The z coordinate of the atom, e.g. 3.0
        occupancy:
            The occupancy of the atom, e.g. 1.0
        b_factor:
            The B-factor of the atom, e.g. 0.0
    """
    return f"{prefix:<6s}{ctr:>5d} {atom_name:>4s} {residue_name:<3s} {chain}{residue_idx.item():>4d}    " + \
                f"{x_coord.item():>8.3f}{y_coord.item():>8.3f}{z_coord.item():>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}\n"


def ab_write_pdblines(
    atoms: np.ndarray,
    seq: np.ndarray,
    chain_idx: np.ndarray,
    idx_pdb: Optional[np.ndarray] = None,
    bfacts: Optional[np.ndarray] = None,
    loop_map: dict[str, List[int]] = None,
) -> List[str]:
    """
    Given a set of atomic coordinates and a sequence, generate a list of PDB lines
    describing the structure.

    Args:
        atoms:
            A [L, N, 3] tensor of atomic coordinates, where N can be 1, 3, 4, 14, or 27
        seq:
            A [L] tensor of integer amino acid residues
        chain_idx:
            A [L] tensor of chain indices
        num2aa:
            The way to convert from residue numbers to amino acid residues
        idx_pdb:
            A [L] tensor of residue indices
        bfacts:
            A [L] tensor of B-factors
        loop_map:
            A dictionary mapping loop names to lists of residue indices
    """
    if not loop_map:
        loop_map = {}

    ctr = 1
    if bfacts is None:
        bfacts = ms.mint.zeros(atoms.shape[0])
    if idx_pdb is None:
        # Default to 1-indexed residue numbers
        idx_pdb = 1 + ms.mint.arange(atoms.shape[0])

    Bfacts = np.clip(
        bfacts,
        a_min=0,
        a_max=1,
    )

    pdblines = []
    for i in range(seq.shape[0]):
        chain = chain_idx[i]

        # If the input is a single set of atomic coordinates, assume it is a C-alpha trace
        if len(atoms.shape) == 2:
            pdblines.append(
                stamp_pdbline(
                    prefix="ATOM",
                    ctr=ctr,
                    atom_name=" CA ",
                    residue_name=num2aa[seq[i]],
                    chain=chain,
                    residue_idx=idx_pdb[i],
                    x_coord=atoms[i, 0],
                    y_coord=atoms[i, 1],
                    z_coord=atoms[i, 2],
                    occupancy=1.0,
                    b_factor=Bfacts[i],
                )
            )

            ctr += 1

        # If the input is a set of atomic coordinates with 3 atoms per residue,
        # assume it is a backbone trace
        elif atoms.shape[1] == 3:
            for j, atm_j in enumerate([" N  ", " CA ", " C  "]):
                pdblines.append(
                    stamp_pdbline(
                        prefix="ATOM",
                        ctr=ctr,
                        atom_name=atm_j,
                        residue_name=num2aa[seq[i]],
                        chain=chain,
                        residue_idx=idx_pdb[i],
                        x_coord=atoms[i, j, 0],
                        y_coord=atoms[i, j, 1],
                        z_coord=atoms[i, j, 2],
                        occupancy=1.0,
                        b_factor=Bfacts[i],
                    )
                )

                ctr += 1

        # If the input is a set of atomic coordinates with 4 atoms per residue,
        # assume it is a backbone trace with an oxygen atom
        elif atoms.shape[1] == 4:
            for j, atm_j in enumerate([" N  ", " CA ", " C  ", " O  "]):
                pdblines.append(
                    stamp_pdbline(
                        prefix="ATOM",
                        ctr=ctr,
                        atom_name=atm_j,
                        residue_name=num2aa[seq[i]],
                        chain=chain,
                        residue_idx=idx_pdb[i],
                        x_coord=atoms[i, j, 0],
                        y_coord=atoms[i, j, 1],
                        z_coord=atoms[i, j, 2],
                        occupancy=1.0,
                        b_factor=Bfacts[i],
                    )
                )

                ctr += 1

        # Otherwise, assume the input is a full atomic tensor with either 14 or 27 atoms per residue
        else:
            natoms = atoms.shape[1]

            assert natoms in (14, 27), (
                "Invalid number of atoms per residue, must be 14 or 27"
            )

            atms = aa2long[aa2num[seq[i]]]

            # his prot hack
            if aa2num[seq[i]] == 8 and ops.norm(atoms[i, 9, :] - atoms[i, 5, :]) < 1.7:
                atms = (
                    " N  ", " CA ", " C  ", " O  ",
                    " CB ", " CG ", " NE2", " CD2",
                    " CE1", " ND1", None, None,
                    None, None, " H  ", " HA ",
                    "1HB ", "2HB ", " HD2", " HE1",
                    " HD1", None, None, None,
                    None, None, None,
                )  # his_d

            for j, atm_j in enumerate(atms):
                if j < natoms and atm_j is not None:
                    pdblines.append(
                        stamp_pdbline(
                            prefix="ATOM",
                            ctr=ctr,
                            atom_name=atm_j,
                            residue_name=seq[i],
                            chain=chain,
                            residue_idx=idx_pdb[i],
                            x_coord=atoms[i, j, 0],
                            y_coord=atoms[i, j, 1],
                            z_coord=atoms[i, j, 2],
                            occupancy=1.0,
                            b_factor=Bfacts[i],
                        )
                    )
                    ctr += 1

    # This may or may not be necessary between the coordinates and the REMARKS
    pdblines.append("TER\n")

    # Add in labels for loop locations in the output structure
    # NB: could also add in the hotspots labels as remarks here as well
    for loop in loop_map:
        for resi in loop_map[loop]:
            pdblines.append(f"REMARK PDBinfo-LABEL:{resi:5d} {loop}\n")

    return pdblines


def parse_pdblines(lines: list[str]) -> tuple[ms.Tensor, list, ms.Tensor]:
    """
    Parses PDB lines to extract sequence, pdb_idx, and XYZ coordinates.

    Args:
        lines:
            A list of PDB lines, where each line is a string.

    Returns:
        seq:
            A tensor of shape (N,) containing the sequence indices, where N is the number of residues.
        pdb_idx:
            A list of tuples (chain, resi) for each residue, where chain is a string and resi is an integer.
        xyz:
            A tensor of shape (N, 27, 3) containing the XYZ coordinates for each atom in each residue.
    """
    res = [
        (l[22:26], l[17:20])
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]
    seq = ms.tensor([aa2num[r[1]] if r[1] in aa2num else 20 for r in res])

    # Generating pdb_idx for indexing
    pdb_idx = [
        (l[21:22].strip(), int(l[22:26].strip()))
        for l in lines
        if l[:4] == "ATOM" and l[12:16].strip() == "CA"
    ]

    # Creating a tensor for XYZ coordinates
    xyz = ms.mint.full((len(res), 27, 3), float("nan"), dtype=ms.float32)

    # A dictionary to quickly find the index in pdb_idx (for efficiency)
    pdb_idx_lookup = {k: i for i, k in enumerate(pdb_idx)}

    for l in lines:
        if l[:4] == "ATOM":
            chain, resNo, atom, aa = (
                l[21:22].strip(),
                int(l[22:26]),
                " " + l[12:16].strip().ljust(3),
                l[17:20],
            )
            if (chain, resNo) in pdb_idx_lookup:
                idx = pdb_idx_lookup[(chain, resNo)]
                if aa in aa2num:  # Ensure aa is known
                    for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
                        if (
                            tgtatm is not None and tgtatm.strip() == atom.strip()
                        ):  # Matching atom name
                            xyz[idx, i_atm, :] = ms.tensor(
                                [float(l[30:38]), float(l[38:46]), float(l[46:54])],
                                dtype=ms.float32,
                            )
                            break

    return seq, pdb_idx, xyz


def split_remark(line: str) -> tuple[str, int]:
    """
    Splits a remark line into loop name and residue index.

    Args:
        line:
            A string line from the PDB file, starting with "REMARK PDBinfo-LABEL".

    Returns:
        loop:
            A string representing the loop name, e.g. "H1", "H2", "H3", "L1", "L2", "L3".
        resi:
            An integer representing the residue index.
    """
    return line.split()[3][0], int(line.split()[2])


def get_cdr_masks_from_remarks(pdb_idx: list, lines: list[str]) -> dict:
    """
    Extracts CDR masks from PDB remarks.

    Args:
        pdb_idx:
            A list of tuples (chain, resi) for each residue, where chain is a string and resi is an integer.
        lines:
            A list of PDB lines, where each line is a string.

    Returns:
        cdr_masks:
            A dictionary with keys "H1", "H2", "H3", "L1", "L2", "L3" and values of boolean masks for each CDR loop.
    """
    cdr_pdb_idx = []
    cdr_names = ["H1", "H2", "H3", "L1", "L2", "L3"]
    cdr_masks = {loop: ms.mint.zeros(len(pdb_idx)).bool() for loop in cdr_names}
    for l in lines:
        if l.startswith("REMARK PDBinfo-LABEL"):
            l = l.strip()
            cdr_pdb_idx.append(split_remark(l))
            loop = l[27:29].upper()
            if loop in cdr_names:
                resi = int(l[21:26]) - 1  # Loop residues in HLT are 1-indexed
                cdr_masks[loop][resi] = True
    if ms.mint.any(ms.mint.stack(list(cdr_masks.values())), dim=0).sum() != len(
        cdr_pdb_idx
    ):
        raise ValueError("Not all cdr residues found in file. Remark indexing is bad")
    return cdr_masks

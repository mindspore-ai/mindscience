# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""tein"""
import io
from typing import Any, Mapping, Optional
import dataclasses

from Bio.PDB import PDBParser
import numpy as np

from . import residue_constants

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


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


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

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
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError(
                'Only single chain PDBs are supported when chain_id not specified. '
                f'Found {len(chains)} chains.')
        chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []

    for res in chain:
        if res.id[2] != ' ':
            raise ValueError(
                f'PDB contains an insertion code at chain {chain.id} and residue '
                f'index {res.id[1]}. These are not supported.')
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
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


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

  Args:
    prot: The protein to convert to PDB.

  Returns:
    PDB string.
  """
    restypes = residue_constants.restypes + ['X']
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError('Invalid aatypes.')

    pdb_lines.append('MODEL     1')
    atom_index = 1
    chain_id = 'A'
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ''
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                         f'{res_name_3:>3} {chain_id:>1}'
                         f'{residue_index[i]:>4}{insertion_code:>1}   '
                         f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                         f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                         f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the chain.
    chain_end = 'TER'
    chain_termination_line = (
        f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} '
        f'{chain_id:>1}{residue_index[-1]:>4}')
    pdb_lines.append(chain_termination_line)
    pdb_lines.append('ENDMDL')

    pdb_lines.append('END')
    pdb_lines.append('')
    return '\n'.join(pdb_lines)


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

  `Protein.atom_mask` typically is defined according to the atoms that are
  reported in the PDB. This function computes a mask according to heavy atoms
  that should be present in the given sequence of amino acids.

  Args:
    prot: `Protein` whose fields are `numpy.ndarray` objects.

  Returns:
    An ideal atom mask.
  """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(final_atom_positions, final_atom_mask, aatype, residue_index, b_factors=None) -> Protein:
    """Assembles a protein from a prediction.

    Args:
        final_atom_positions: atom positions
        final_atom_mask: atom mask
        aatype: amino acid type
        residue_index: idx of the residue
    Returns:
        A protein instance.
    """
    if b_factors is None:
        b_factors = np.zeros_like(final_atom_mask)

    return Protein(
        aatype=aatype,
        atom_positions=final_atom_positions,
        atom_mask=final_atom_mask,
        residue_index=residue_index + 1,
        b_factors=b_factors)


@dataclasses.dataclass(frozen=True)
class ProteinV2:
    """ProteinV2 structure representation."""

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

    # 0-indexed number corresponding to the chain in the protein that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]


def to_pdb_v2(prot: ProteinV2) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ['X']
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError('Invalid aatypes.')

    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        if last_chain_index != chain_index[i]:
            chain_end = 'TER'
            chain_termination_line = (
                f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i - 1]):>3} '
                f'{chain_ids[chain_index[i - 1]]:>1}{residue_index[i - 1]:>4}')
            pdb_lines.append(chain_termination_line)
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
                atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ''
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                         f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                         f'{residue_index[i]:>4}{insertion_code:>1}   '
                         f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                         f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                         f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the chain.
    chain_end = 'TER'
    chain_termination_line = (
        f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} '
        f'{chain_ids[chain_index[-1]]:>1}{residue_index[-1]:>4}')
    pdb_lines.append(chain_termination_line)
    pdb_lines.append('ENDMDL')

    pdb_lines.append('END')
    pdb_lines.append('')
    return '\n'.join(pdb_lines)


def from_prediction_v2(final_atom_positions,
                       final_atom_mask,
                       aatype,
                       residue_index,
                       b_factors=None,
                       asym_id=None,
                       remove_leading_feature_dimension=True) -> ProteinV2:
    """Assembles a protein from a prediction.

    Args:
        final_atom_positions: atom positions
        final_atom_mask: atom mask
        aatype: amino acid type
        residue_index: idx of the residue
    Returns:
        A protein instance.
    """

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if asym_id is not None:
        chain_index = _maybe_remove_leading_dim(asym_id)
    else:
        chain_index = np.zeros_like(aatype)
    if b_factors is None:
        b_factors = np.zeros_like(final_atom_mask)

    return ProteinV2(
        aatype=aatype,
        atom_positions=final_atom_positions,
        atom_mask=final_atom_mask,
        residue_index=residue_index + 1,
        chain_index=chain_index,
        b_factors=b_factors)


def from_prediction_new(features) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.

    Returns:
      A protein instance.
    """
    dist_per_residue = features[4]

    return Protein(
        aatype=features[0],
        atom_positions=features[2],
        atom_mask=features[3],
        residue_index=features[1] + 1,
        b_factors=dist_per_residue)

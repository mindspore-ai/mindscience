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
"""
Read information from a pdb format file.
"""
import re
import string
import collections
import io
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple, List

from absl import logging
from Bio import PDB
from Bio.Data import SCOPData


@dataclasses.dataclass(frozen=True)
class HhrHit:
    """Class representing a hit in an hhr file."""
    index: int
    name: str
    prob_true: float
    e_value: float
    score: float
    aligned_cols: int
    identity: float
    similarity: float
    sum_probs: float
    neff: float
    query: str
    hit_sequence: str
    hit_dssp: str
    column_score_code: str
    confidence_scores: str
    indices_query: List[int]
    indices_hit: List[int]


# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PDBSTRUCTURE = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
    id: str
    num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
      file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
        files being processed.
      header: Biopython header.
      structure: Biopython structure.
      chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
        {'A': 'ABCDEFG'}
      seqres_to_structure: Dict; for each chain_id contains a mapping between
        SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                          1: ResidueAtPosition,
                                                          ...}}
      raw_string: The raw string used to construct the MmcifObject.
    """
    file_id: str
    header: PdbHeader
    structure: PDBSTRUCTURE
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    raw_string: Any


@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """
    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


def _update_hhr_residue_indices_list(
        sequence, start_index, indices_list):
    """Computes the relative indices for each residue with respect to the original sequence."""
    counter = start_index
    for symbol in sequence:
        if symbol == '-':
            indices_list.append(-1)
        else:
            indices_list.append(counter)
            counter += 1


def _get_hhr_line_regex_groups(
        regex_pattern: str, line: str):
    match = re.match(regex_pattern, line)
    if match is None:
        raise RuntimeError(f'Could not parse query line {line}')
    return match.groups()


def parse_fasta(fasta_string: str):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def _parse_hhr_hit(detailed_lines):
    """Parses the detailed HMM HMM comparison section for a single Hit.

    This works on .hhr files generated from both HHBlits and HHSearch.

    Args:
      detailed_lines: A list of lines from a single comparison section between 2
        sequences (which each have their own HMM's)

    Returns:
      A dictionary with the information from that detailed comparison section

    Raises:
      RuntimeError: If a certain line cannot be processed
    """
    # Parse first 2 lines.
    number_of_hit = int(detailed_lines[0].split()[-1])
    name_hit = detailed_lines[1][1:]

    # Parse the summary line.
    pattern = (
        'Probab=(.*)[\t ]*E-value=(.*)[\t ]*Score=(.*)[\t ]*Aligned_cols=(.*)[\t'
        ' ]*Identities=(.*)%[\t ]*Similarity=(.*)[\t ]*Sum_probs=(.*)[\t '
        ']*Template_Neff=(.*)')
    match = re.match(pattern, detailed_lines[2])
    if match is None:
        raise RuntimeError(
            'Could not parse section: %s. Expected this: \n%s to contain summary.' %
            (detailed_lines, detailed_lines[2]))
    (prob_true, e_value, score, aligned_cols, identity, similarity, sum_probs,
     neff) = [float(x) for x in match.groups()]

    # The next section reads the detailed comparisons. These are in a 'human
    # readable' format which has a fixed length. The strategy employed is to
    # assume that each block starts with the query sequence line, and to parse
    # that with a regexp in order to deduce the fixed length used for that
    # block.
    query = ''
    hit_sequence = ''
    hit_dssp = ''
    column_score_code = ''
    confidence_scores = ''
    indices_query = []
    indices_hit = []
    length_block = None

    for line in detailed_lines[3:]:
        # Parse the query sequence line
        if (line.startswith('Q ') and not line.startswith('Q ss_dssp') and not line.startswith('Q ss_pred') \
                and not line.startswith('Q Consensus')):
            # Thus the first 17 characters must be 'Q <query_name> ', and we can parse
            # everything after that.
            # start    sequence       end       total_sequence_length
            patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)'
            groups = _get_hhr_line_regex_groups(patt, line[17:])

            # Get the length of the parsed block using the start and finish indices,
            # and ensure it is the same as the actual block length.
            start = int(groups[0]) - 1  # Make index zero based.
            delta_query = groups[1]
            end = int(groups[2])
            num_insertions = len([x for x in delta_query if x == '-'])
            length_block = end - start + num_insertions
            assert length_block == len(delta_query)

            # Update the query sequence and indices list.
            query += delta_query
            _update_hhr_residue_indices_list(delta_query, start, indices_query)

        elif line.startswith('T '):
            # Parse the hit dssp line.
            if line.startswith('T ss_dssp'):
                #        T ss_dssp      hit_dssp
                patt = r'T ss_dssp[\t ]*([A-Z-]*)'
                groups = _get_hhr_line_regex_groups(patt, line)
                assert len(groups[0]) == length_block
                hit_dssp += groups[0]

            # Parse the hit sequence.
            elif (not line.startswith('T ss_pred') and
                  not line.startswith('T Consensus')):
                # Thus the first 17 characters must be 'T <hit_name> ', and we can
                # parse everything after that.
                # start    sequence       end     total_sequence_length
                patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*[0-9]* \([0-9]*\)'
                groups = _get_hhr_line_regex_groups(patt, line[17:])
                start = int(groups[0]) - 1  # Make index zero based.
                delta_hit_sequence = groups[1]
                assert length_block == len(delta_hit_sequence)

                # Update the hit sequence and indices list.
                hit_sequence += delta_hit_sequence
                _update_hhr_residue_indices_list(
                    delta_hit_sequence, start, indices_hit)

        # Parse the column score line.
        elif line.startswith(' ' * 22):
            assert length_block
            column_score_code += line[22:length_block + 22]

        # Update confidence score.
        elif line.startswith('Confidence'):
            assert length_block
            confidence_scores += line[22:length_block + 22]

    return HhrHit(
        index=number_of_hit,
        name=name_hit,
        prob_true=prob_true,
        e_value=e_value,
        score=score,
        aligned_cols=int(aligned_cols),
        identity=identity,
        similarity=similarity,
        sum_probs=sum_probs,
        neff=neff,
        query=query,
        hit_sequence=hit_sequence,
        hit_dssp=hit_dssp,
        column_score_code=column_score_code,
        confidence_scores=confidence_scores,
        indices_query=indices_query,
        indices_hit=indices_hit,
    )


def parse_hhr(hhr_string: str):
    """Parses the content of an entire HHR file."""
    lines = hhr_string.splitlines()

    # Each .hhr file starts with a results table, then has a sequence of hit
    # "paragraphs", each paragraph starting with a line 'No <hit number>'. We
    # iterate through each paragraph to parse each hit.

    block_starts = [i for i, line in enumerate(lines) if line.startswith('No ')]

    hits = []
    if block_starts:
        block_starts.append(len(lines))  # Add the end of the final block.
        for i in range(len(block_starts) - 1):
            hits.append(_parse_hhr_hit(lines[block_starts[i]:block_starts[i + 1]]))
    return hits


def parse_a3m(a3m_string: str):
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
      a3m_string: The string contents of a a3m file. The first sequence in the
        file should be the query sequence.

    Returns:
      A tuple of:
        * A list of sequences that have been aligned to the query. These
          might contain duplicates.
        * The deletion matrix for the alignment as a list of lists. The element
          at `deletion_matrix[i][j]` is the number of residues deleted from
          the aligned sequence i at residue position j.
    """
    sequences, _ = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans('', '', string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix


def mmcif_loop_to_list(prefix, parsed_info):
    """Extracts loop associated with a prefix from mmCIF data as a list.

    Reference for loop_ in mmCIF:
      http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
    """
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), ('mmCIF error: Not all loops are the same length: %s' % cols)

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(prefix, index, parsed_info):
    """Extracts loop associated with a prefix from mmCIF data as a dictionary.

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      index: Which item of loop data should serve as the key.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
      indexed by the index column.
    """
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


def parse_mmcif(*,
                file_id: str,
                mmcif_string: str,
                catch_all_errors: bool = True):
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    Returns:
      A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure('', handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(None, {(file_id, ''): 'No protein chains found in this file.'})
        seq_start_num = {chain_id: min([monomer.num for monomer in seq]) for chain_id, seq in valid_chains.items()}

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != '1':
                # We only process the first model at the moment.
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in valid_chains:
                hetflag = ' '
                if atom.hetatm_atom == 'HETATM':
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ('HOH', 'WAT'):
                        hetflag = 'W'
                    else:
                        hetflag = 'H_' + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = ' '
                position = ResiduePosition(chain_id=atom.author_chain_id, residue_number=int(
                    atom.author_seq_num), insertion_code=insertion_code)
                seq_idx = int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                current = seq_to_structure_mappings.get(atom.author_chain_id, {})
                current[seq_idx] = ResidueAtPosition(position=position,
                                                     name=atom.residue_name,
                                                     is_missing=False,
                                                     hetflag=hetflag)
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id.get(chain_id)
            current_mapping = seq_to_structure_mappings.get(author_chain)
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(position=None,
                                                             name=monomer.id,
                                                             is_missing=True,
                                                             hetflag=' ')

        author_chain_to_sequence = {}
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id.get(chain_id)
            seq = []
            for monomer in seq_info:
                code = SCOPData.protein_letters_3to1.get(monomer.id, 'X')
                seq.append(code if len(code) == 1 else 'X')
            seq = ''.join(seq)
            author_chain_to_sequence[author_chain] = seq

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            raw_string=parsed_info)

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, '')] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)


def _get_first_model(structure: PDBSTRUCTURE) -> PDBSTRUCTURE:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())


_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info['_pdbx_audit_revision_history.revision_date']
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list('_exptl.', parsed_info)
    header['structure_method'] = ','.join([experiment['_exptl.method'].lower() for experiment in experiments])

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    if '_pdbx_audit_revision_history.revision_date' in parsed_info:
        header['release_date'] = get_release_date(parsed_info)
    else:
        logging.warning('Could not determine release_date: %s', parsed_info['_entry.id'])

    header['resolution'] = 0.00
    for res_key in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution', '_reflns.d_resolution_high'):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header['resolution'] = float(raw_resolution)
            except ValueError:
                logging.warning('Invalid resolution format: %s', parsed_info[res_key])

    return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    """Returns list of atom sites; contains data not present in the structure."""
    return [AtomSite(*site) for site in zip(  # pylint:disable=g-complex-comprehension
        parsed_info['_atom_site.label_comp_id'],
        parsed_info['_atom_site.auth_asym_id'],
        parsed_info['_atom_site.label_asym_id'],
        parsed_info['_atom_site.auth_seq_id'],
        parsed_info['_atom_site.label_seq_id'],
        parsed_info['_atom_site.pdbx_PDB_ins_code'],
        parsed_info['_atom_site.group_PDB'],
        parsed_info['_atom_site.pdbx_PDB_model_num'],
    )]


def _get_protein_chains(*, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
    """Extracts polymer information for protein chains only.

    Args:
      parsed_info: _mmcif_dict produced by the Biopython parser.

    Returns:
      A dict mapping mmcif chain id to a list of Monomers.
    """
    # Get polymer information for each entity in the structure.
    entity_poly_seqs = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)

    polymers = collections.defaultdict(list)
    for entity_poly_seq in entity_poly_seqs:
        polymers[entity_poly_seq['_entity_poly_seq.entity_id']].append(
            Monomer(id=entity_poly_seq['_entity_poly_seq.mon_id'], num=int(entity_poly_seq['_entity_poly_seq.num'])))

    # Get chemical compositions. Will allow us to identify which of these polymers
    # are proteins.
    chem_comps = mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)

    entity_to_mmcif_chains = collections.defaultdict(list)
    for struct_asym in struct_asyms:
        chain_id = struct_asym['_struct_asym.id']
        entity_id = struct_asym['_struct_asym.entity_id']
        entity_to_mmcif_chains[entity_id].append(chain_id)

    # Identify and return the valid protein chains.
    valid_chains = {}
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]

        # Reject polymers without any peptide-like components, such as DNA/RNA.
        if any(['peptide' in chem_comps[monomer.id]['_chem_comp.type'] for monomer in seq_info]):
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info
    return valid_chains


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in ('.', '?')

# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""prepare train msa."""
import json
import argparse
import os
from os.path import join as opjoin
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set, Mapping
import random
import math
from enum import Enum
from datetime import datetime
from pathlib import Path
import concurrent.futures
import multiprocessing
from functools import partial
import time
import fcntl

from tqdm import tqdm
import biotite.structure as struct
import pandas as pd
from joblib import Parallel, delayed

from protenix.data.ccd import get_one_letter_code
from protenix.data.parser import MMCIFParser
from runner.msa_search import msa_search
from utils import (
    convert_to_shared_dict,
    SharedDict,
    get_shared_dict_ids,
    release_shared_dict,
)

pd.options.mode.copy_on_write = True


def get_seqs(mmcif_file):
    """get sequences from mmcif file.
    Args:
        mmcif_file: path to mmcif file
    Returns:
        pdb_id: pdb id
        info_df: dataframe containing sequence information
    """
    mmcif_parser = MMCIFParser(mmcif_file)

    entity_poly = mmcif_parser.get_category_table("entity_poly")
    if entity_poly is None:
        pdb_id = mmcif_file.name.split(".")[0]
        return pdb_id, None
    entity_poly["mmcif_seq_old"] = entity_poly.pdbx_seq_one_letter_code_can.str.replace(
        "\n", ""
    )
    entity_poly["pdbx_type"] = entity_poly.type
    mol_type = []
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
    for i in entity_poly.type:
        if "ribonucleotide" in i:
            mol_type.append("na")
        elif "polypeptide" in i:
            mol_type.append("protein")
        else:
            mol_type.append("other")
    entity_poly["mol_type"] = mol_type

    entity = mmcif_parser.get_category_table("entity")
    info_df = pd.merge(
        entity, entity_poly, left_on="id", right_on="entity_id", how="inner"
    )
    pdb_id = mmcif_file.name.split(".")[0]
    info_df["pdb_id"] = pdb_id

    if "pdbx_audit_revision_history" in mmcif_parser.cif.block:
        history = mmcif_parser.cif.block["pdbx_audit_revision_history"]
        info_df["release_date"] = history["revision_date"].as_array()[0]
    else:
        # Handle non-official mmcif file which transform from pdb file
        info_df["release_date"] = datetime.now().strftime("%Y-%m-%d")

    if mmcif_parser.release_date:
        info_df["release_date_retrace_obsolete"] = mmcif_parser.release_date
    else:
        # Handle non-official mmcif file which transform from pdb file
        info_df["release_date_retrace_obsolete"] = datetime.now().strftime(
            "%Y-%m-%d")

    entity_poly_seq = mmcif_parser.get_category_table("entity_poly_seq")

    seq_from_resname = []
    diff_seq_mmcif_vs_atom_site = []
    has_alt_res = []
    diff_alt_res_seq_vs_atom_site = []
    for entity_id, mmcif_seq_old in zip(info_df.entity_id, info_df.mmcif_seq_old):
        chain_mask = entity_poly_seq.entity_id == entity_id
        res_names = entity_poly_seq.mon_id[chain_mask].to_numpy(dtype=str)
        res_ids = entity_poly_seq.num[chain_mask].to_numpy(dtype=int)

        seq = ""
        pre_res_id = 0
        id_2_name = {}
        for res_id, res_name in zip(res_ids, res_names):
            if res_id == pre_res_id:
                continue
            id_2_name[res_id] = res_name
            one = get_one_letter_code(res_name)
            if one is None:
                one = "X"
            if len(one) > 1:
                one = "X"
            seq += one
            pre_res_id = res_id
        assert len(seq) == max(res_ids)

        diff_seq_mmcif_vs_atom_site.append(seq != mmcif_seq_old)
        has_alt = False
        mismatch_res_name = False
        if len(seq) < len(res_ids):  # has altloc residue in same res_id
            has_alt = True
            atom_array = mmcif_parser.get_structure()
            res_starts = struct.get_residue_starts(atom_array)
            for start in res_starts:
                if atom_array.label_entity_id[start] == entity_id:
                    first_res_in_seq = id_2_name[atom_array.res_id[start]]
                    first_res_in_atom = atom_array.res_name[start]
                    if first_res_in_seq != first_res_in_atom:
                        mismatch_res_name = True
                        break

        has_alt_res.append(has_alt)
        diff_alt_res_seq_vs_atom_site.append(mismatch_res_name)

        seq_from_resname.append(seq)
    info_df["seq"] = seq_from_resname
    info_df["length"] = [len(s) for s in info_df.seq]
    info_df["diff_seq_mmcif_vs_atom_site"] = diff_seq_mmcif_vs_atom_site
    info_df["has_alt_res"] = has_alt_res
    info_df["diff_alt_res_seq_vs_atom_site"] = diff_alt_res_seq_vs_atom_site
    info_df["auth_asym_id"] = info_df["pdbx_strand_id"]

    columns = [
        "pdb_id",
        "entity_id",
        "mol_type",
        "pdbx_type",
        "length",
        "mmcif_seq_old",
        "seq",
        "diff_seq_mmcif_vs_atom_site",
        "has_alt_res",
        "diff_alt_res_seq_vs_atom_site",
        "pdbx_description",
        "auth_asym_id",
        "release_date",
        "release_date_retrace_obsolete",
    ]
    info_df = info_df[columns]
    return pdb_id, info_df


def try_get_seqs(cif_file):
    pdb_id = cif_file.name.split(".")[0]
    try:
        return get_seqs(cif_file)
    except Exception as e:
        print("skip", pdb_id, e)
        return pdb_id, "Error:" + str(e)


def export_to_fasta(df, filename):
    df_protein = df[df["mol_type"] == "protein"]
    # drop duplicates sequence for avoiding duplicate msa search
    df_protein = df_protein.drop_duplicates(subset=["seq"])
    with open(filename, "w", encoding='utf-8') as fasta_file:
        for _, row in df_protein.iterrows():
            header = f">{row['pdb_id']}_{row['entity_id']}\n"
            sequence = f"{row['seq']}\n"
            fasta_file.write(header)
            fasta_file.write(sequence)


def mapping_seqs_to_pdb_entity_id(df, output_json_file):
    """mapping seqs to pdb id and entity id.
    Args:
        df: dataframe containing sequence information
        output_json_file: path to output json file
    Returns:
        sequence_mapping: mapping of sequence to pdb id and entity id
    """
    df_protein = df[df["mol_type"] == "protein"]
    sequence_mapping = {}

    for _, row in df_protein.iterrows():
        seq = row["seq"]
        key = row["pdb_id"]
        value = row["entity_id"]

        if seq not in sequence_mapping:
            sequence_mapping[seq] = []
        sequence_mapping[seq].append([key, value])

    with open(output_json_file, "w", encoding='utf-8') as json_file:
        json.dump(sequence_mapping, json_file, indent=4)
    return sequence_mapping


def mapping_seqs_to_integer_identifiers(
    sequence_mapping,
    pdb_index_to_seq_path,
    seq_to_pdb_index_path,
):
    """mapping seqs to integer identifiers.
    Args:
        sequence_mapping: mapping of sequence to pdb id and entity id
        pdb_index_to_seq_path: path to pdb index to sequence mapping
        seq_to_pdb_index_path: path to sequence to pdb index mapping
    """
    seq_to_pdb_index = {}
    for idx, seq in enumerate(sorted(sequence_mapping.keys())):
        seq_to_pdb_index[seq] = idx
    pdb_index_to_seq = {v: k for k, v in seq_to_pdb_index.items()}
    with open(pdb_index_to_seq_path, "w", encoding='utf-8') as f:
        json.dump(pdb_index_to_seq, f, indent=4)
    with open(seq_to_pdb_index_path, "w", encoding='utf-8') as f:
        json.dump(seq_to_pdb_index, f, indent=4)


class MolType(Enum):
    """moltype enum."""
    RNA = ("sequence", "rna")
    DNA = ("sequence", "dna")
    CCD = ("ccdCodes", "ligand")
    SMILES = ("smiles", "ligand")

    def __init__(self, af3code, upperclass):
        self.af3code = af3code
        self.upperclass = upperclass

    @classmethod
    def get_moltype(cls, moltype: str):
        """get moltype from string.
        Args:
            moltype: string of moltype
        Returns:
            MolType: MolType object
        """
        if moltype == "RNA":
            return cls.RNA
        if moltype == "DNA":
            return cls.DNA
        if moltype == "SMILES":
            return cls.SMILES
        if moltype == "CCD":
            return cls.CCD
        raise ValueError(
            "Only dna, rna, ccd, smiles are allowed as molecule types.")


def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
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
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        if not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def classify_molecules(query_sequence: str) -> Tuple[List[str], Optional[List[Tuple[MolType, str, int]]]]:
    """Classifies the sequences in the query sequence string into protein and non-protein sequences.

    Returns a tuple of two lists:
    * A list of protein sequences.
    * A list of tuples, each containing a molecule type, a sequence, and number of copies.
    """
    sequences = query_sequence.upper().split(":")
    protein_queries = []
    other_queries = []
    for seq in sequences:
        if seq.count("|") == 0:
            protein_queries.append(seq)
        else:
            parts = seq.split("|")
            moltype, sequence, *rest = parts
            moltype = MolType.get_moltype(moltype)
            if moltype == MolType.SMILES:
                sequence = sequence.replace(";", ":")
            copies = int(rest[0]) if rest else 1
            # (molecule type, sequence, copies)
            other_queries.append((moltype, sequence, copies))

    if len(other_queries) == 0:
        other_queries = None

    return protein_queries, other_queries


def _parse_csv_queries(input_path: Path):
    """Parse queries from CSV/TSV file.
    
    Args:
        input_path: Path to CSV/TSV file
        
    Returns:
        Tuple of (queries, sequences, headers)
    """
    sep = "\t" if input_path.suffix == ".tsv" else ","
    df = pandas.read_csv(input_path, sep=sep, dtype=str)
    assert "id" in df.columns and "sequence" in df.columns
    queries = [
        (seq_id, sequence.upper().split(":"), None, None)
        for seq_id, sequence in df[["id", "sequence"]].itertuples(index=False)
    ]
    for _, query in enumerate(queries):
        if len(query[1]) == 1:
            query = (query[0], query[1][0], None, None)
    # For CSV, we don't have separate sequences/headers in the same format
    return queries, None, None


def _parse_fasta_queries(input_path: Path):
    """Parse queries from FASTA file.
    
    Args:
        input_path: Path to FASTA file
        
    Returns:
        Tuple of (queries, sequences, headers)
    """
    (sequences, headers) = parse_fasta(input_path.read_text(encoding='utf-8'))
    queries = []
    for sequence, header in zip(sequences, headers):
        sequence = sequence.upper()
        if sequence.count(":") == 0:
            # Single sequence
            queries.append((header, sequence, None, None))
        else:
            # Complex mode
            protein_queries, other_queries = classify_molecules(sequence)
            queries.append((header, protein_queries, None, other_queries))
    return queries, sequences, headers


def _check_is_complex(queries):
    """Check if queries contain complex structures.
    
    Args:
        queries: List of query tuples
        
    Returns:
        Boolean indicating if queries are complex
    """
    for _, (_, query_sequence, a3m_lines, _) in enumerate(queries):
        if isinstance(query_sequence, list):
            return True
        if a3m_lines is not None and a3m_lines[0].startswith("#"):
            a3m_line = a3m_lines[0].splitlines()[0]
            tab_sep_entries = a3m_line[1:].split("\t")
            if len(tab_sep_entries) == 2:
                query_seq_len = list(map(int, tab_sep_entries[0].split(",")))
                query_seqs_cardinality = list(map(int, tab_sep_entries[1].split(",")))
                is_single_protein = bool(len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1)
                if not is_single_protein:
                    return True
    return False


def get_queries(
    input_path: Union[str, Path], sort_queries_by: str = "length"
) -> Tuple[List[Tuple[str, str, Optional[List[str]], Optional[List[Tuple[MolType, str, int]]]]], bool]:
    """Reads a directory of fasta files, a single fasta file or a csv file and returns a tuple
    of job name, sequence, optional a3m lines, and the optional non-protein sequences."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    if not input_path.is_file():
        raise ValueError(f"Only support file input but got {input_path}")

    if input_path.suffix in (".csv", ".tsv"):
        queries, sequences, headers = _parse_csv_queries(input_path)
    elif input_path.suffix in [".fasta", ".faa", ".fa"]:
        queries, sequences, headers = _parse_fasta_queries(input_path)
    else:
        raise ValueError(f"Unknown file format {input_path.suffix}")

    # sort by seq. len
    if sort_queries_by == "length":
        queries.sort(key=lambda t: len("".join(t[1])))
    elif sort_queries_by == "random":
        random.shuffle(queries)

    is_complex = _check_is_complex(queries)

    return sequences, is_complex, headers


def process_block_binary(block_info):
    """Process a range of blocks with binary file reading for better performance

    Args:
        block_info (tuple): (start_block, end_block, file_path, block_size, file_size, num_blocks)

    Returns:
        dict: Dictionary of results from these blocks
    """
    start_block, end_block, file_path, block_size, file_size, num_blocks = block_info
    local_dict = {}

    # Buffer size for reading across block boundaries
    boundary_buffer_size = 8192  # 8KB should be enough for even very long lines

    with open(file_path, 'rb') as f:
        full_line = None
        for block_num in range(start_block, end_block):
            # Calculate block bounds
            block_offset = block_num * block_size

            # Determine the start position for reading this block
            if block_num == 0:
                # First block always starts at the beginning of the file
                start_pos = 0
            else:
                # For subsequent blocks, we need to find where the first complete line starts
                # First, check if the previous block ended with a newline
                prev_block_end = block_offset - 1
                f.seek(prev_block_end)
                last_char_prev_block = f.read(1)

                # If the previous block ended with a newline, start at the beginning of this block
                if last_char_prev_block == b'\n':
                    start_pos = block_offset
                else:
                    # Previous block didn't end with a newline, find the first newline in this block
                    f.seek(block_offset)
                    chunk = f.read(
                        min(boundary_buffer_size, file_size - block_offset))
                    newline_pos = chunk.find(b'\n')

                    # If no newline found, this entire block is part of a line from previous block
                    if newline_pos == -1:
                        continue

                    # Start after the first newline
                    start_pos = block_offset + newline_pos + 1

            # Calculate how much data to read from the start position
            read_length = min(
                block_size - (start_pos - block_offset), file_size - start_pos)

            # Skip if nothing to read after adjustments
            if read_length <= 0:
                continue

            # Read the data block
            f.seek(start_pos)
            data = f.read(read_length)

            # Skip if no data
            if not data:
                continue

            # Split into lines and process
            lines = data.split(b'\n')

            # Process all lines except possibly the last one if it's incomplete
            for i, line in enumerate(lines):
                # Special handling for the last line in the block (if not the last block)
                if i == len(lines) - 1 and block_num < num_blocks - 1:
                    end_pos = start_pos + len(data)
                    if end_pos < file_size and not data.endswith(b'\n'):
                        # Last line is incomplete, need to read more to complete it
                        incomplete_line = line

                        # Read ahead to find the rest of the line
                        f.seek(end_pos)
                        extra_data = f.read(
                            min(boundary_buffer_size, file_size - end_pos))

                        # Find the end of the line
                        newline_pos = extra_data.find(b'\n')
                        if newline_pos != -1:
                            # Found the end of the line
                            line_remainder = extra_data[:newline_pos]
                            full_line = incomplete_line + line_remainder
                    else:
                        full_line = line
                else:
                    full_line = line

                # Process complete lines
                if full_line:  # Skip empty lines
                    try:
                        line_str = full_line.decode('utf-8')
                        line_list = line_str.split('\t')
                        hit_name = line_list[1]
                        ncbi_taxid = line_list[2]
                        local_dict[hit_name] = ncbi_taxid
                    except Exception:
                        # Skip problematic lines
                        continue

    return local_dict


def read_a3m(a3m_file: str) -> Tuple[List[str], List[str], int]:
    """read a3m file from output of mmseqs

    Args:
        a3m_file (str): the a3m file searched by mmseqs(colabfold search)

    Returns:
        Tuple[List[str], List[str], int]: the header, seqs of a3m files, and uniref index
    """
    heads = []
    seqs = []
    # Record the row index. The index before this index is the MSA of Uniref30 DB,
    # and the index after this index is the MSA of ColabfoldDB.
    uniref_index = 0
    with open(a3m_file, "r", encoding='utf-8') as infile:
        for idx, line in enumerate(infile):
            if line.startswith(">"):
                heads.append(line)
                if idx == 0:
                    query_name = line
                elif idx > 0 and line == query_name:
                    uniref_index = idx
            else:
                seqs.append(line)
    return heads, seqs, uniref_index


def read_m8(
    m8_file: str,
    max_workers: Optional[int] = None,
    block_size_mb: int = 64
) -> Dict[str, str]:
    """Read the uniref_tax.m8 file from output of mmseqs using optimized block processing.

    This implementation automatically selects the best processing approach based on file size:
    1. Simple sequential processing for small to medium files
    2. Block-wise processing with multiprocessing for large files when beneficial

    Args:
        m8_file (str): the uniref_tax.m8 from output of mmseqs(colabfold search)
        max_workers (Optional[int]): maximum number of worker processes to use (defaults to CPU count - 1)
        block_size_mb (int): size of each processing block in MB. Do not set too small, otherwise the 
            overhead of process creation and task management will be too high and unfound bugs will be introduced.

    Returns:
        Dict[str, str]: the dict mapping uniref hit_name to NCBI TaxID
    """
    # Get file size to report progress
    file_size = os.path.getsize(m8_file)
    print(f"Reading m8 file ({file_size/(1024*1024):.1f} MB)...")

    # Calculate block size and number of blocks based on input parameter
    block_size = block_size_mb * 1024 * 1024  # Convert MB to bytes
    num_blocks = math.ceil(file_size / block_size)

    # Calculate available CPU resources
    available_cpus = multiprocessing.cpu_count() - 1 or 1  # At least 1

    # Determine if multiprocessing would be beneficial
    # We use multiprocessing if:
    # 1. We have more than 1 block
    # 2. We have at least 2 CPUs available
    use_multiprocessing = (
        num_blocks > 1 and
        available_cpus > 1
    )

    # Set up number of workers if we're using multiprocessing
    if use_multiprocessing:
        if max_workers is None:
            # Use a reasonable number of workers based on blocks and CPUs
            max_workers = min(available_cpus, num_blocks, 16)
        else:
            # Ensure we don't use more workers than blocks or available CPUs
            max_workers = min(max_workers, available_cpus, num_blocks)

        # If we only have 1 worker, fall back to sequential processing
        if max_workers <= 1:
            use_multiprocessing = False

    uniref_to_ncbi_taxid = {}

    # For multiprocessing approach (large files with sufficient CPU resources)
    if use_multiprocessing:
        print(
            f"File is large, using multiprocessing with {max_workers} workers for {num_blocks} blocks")

        # Create batches of blocks
        batches = []
        for i in range(0, num_blocks):
            batches.append(
                (i, i+1, m8_file, block_size, file_size, num_blocks))

        # Process batches with progress tracking
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_block_binary, batch)
                       for batch in batches]

            with tqdm(total=len(batches), desc="Processing file blocks") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_dict = future.result()
                        # Merge results
                        uniref_to_ncbi_taxid.update(batch_dict)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        pbar.update(1)

    # Sequential processing approach (for small to medium files)
    else:
        print(
            f"Using sequential processing for {file_size/(1024*1024):.1f} MB file")

        # Original single-threaded line-by-line processing
        with open(m8_file, "r", encoding='utf-8') as infile:
            for line in tqdm(infile, desc="Reading m8 file", unit="lines"):
                line_list = line.rstrip().split("\t")
                hit_name = line_list[1]
                ncbi_taxid = line_list[2]
                uniref_to_ncbi_taxid[hit_name] = ncbi_taxid

    print(f"Processed {len(uniref_to_ncbi_taxid):,} unique entries")

    return uniref_to_ncbi_taxid


def update_a3m(
    a3m_path: str,
    uniref_to_ncbi_taxid: Dict[str, str],
    save_root: str,
) -> None:
    """add NCBI TaxID to header if "UniRef" in header

    Args:
        a3m_path (str): the original a3m path returned by mmseqs(colabfold search)
        uniref_to_ncbi_taxid (Dict): the dict mapping uniref hit_name to NCBI TaxID
        save_root (str): the updated a3m
    """
    heads, seqs, uniref_index = read_a3m(a3m_path)
    fname = a3m_path.split("/")[-1]
    out_a3m_path = opjoin(save_root, fname)
    with open(out_a3m_path, "w", encoding='utf-8') as ofile:
        for idx, (head, seq) in enumerate(zip(heads, seqs)):
            uniref_id = head.split("\t")[0][1:]
            ncbi_taxid = uniref_to_ncbi_taxid.get(uniref_id, None)
            if (ncbi_taxid is not None) and (idx < (uniref_index // 2)):
                if not uniref_id.startswith("UniRef100_"):
                    head = head.replace(
                        uniref_id, f"UniRef100_{uniref_id}_{ncbi_taxid}/"
                    )
                else:
                    head = head.replace(
                        uniref_id, f"{uniref_id}_{ncbi_taxid}/")
            ofile.write(f"{head}{seq}")


def update_a3m_batch(batch_paths: List[str], uniref_to_ncbi_taxid: Dict[str, str], save_root: str) -> int:
    """Process a batch of a3m files.

    Args:
        batch_paths (List[str]): List of paths to a3m files to process
        uniref_to_ncbi_taxid (Dict[str, str]): Dictionary mapping UniRef IDs to NCBI TaxIDs
        save_root (str): Directory to save processed files

    Returns:
        int: Number of files processed
    """
    for a3m_path in batch_paths:
        update_a3m(
            a3m_path=a3m_path,
            uniref_to_ncbi_taxid=uniref_to_ncbi_taxid,
            save_root=save_root
        )
    return len(batch_paths)


def process_files(
    a3m_paths: List[str],
    uniref_to_ncbi_taxid: Union[Dict[str, str], Any],
    output_msa_dir: str,
    num_workers: Optional[int] = None,
    batch_size: Optional[int] = None
) -> None:
    """Process multiple a3m files with optimized performance.

    This function uses a more efficient approach for multiprocessing by using batched 
    processing to reduce the overhead of process creation and task management.
    Works with both regular dictionaries and shared dictionaries.

    Args:
        a3m_paths (List[str]): List of a3m file paths to process
        uniref_to_ncbi_taxid (Union[Dict[str, str], Any]): 
            Dictionary mapping UniRef IDs to NCBI TaxIDs, can be either a regular dict or a shared dict
        output_msa_dir (str): Directory to save processed files
        num_workers (int, optional): Number of worker processes. Defaults to None (uses CPU count).
        batch_size (int, optional): Size of batches for processing. Defaults to None (auto-calculated).
    """
    if num_workers is None:
        # Use a smaller number of workers to avoid excessive overhead
        num_workers = max(1, min(multiprocessing.cpu_count() - 1, 16))

    total_files = len(a3m_paths)

    if batch_size is None:
        # Calculate an optimal batch size based on number of files and workers
        # Aim for each worker to get 2-5 batches for good load balancing
        target_batches_per_worker = 3
        batch_size = max(1, math.ceil(
            total_files / (num_workers * target_batches_per_worker)))

    # Create batches
    batches = [a3m_paths[i:i + batch_size]
               for i in range(0, len(a3m_paths), batch_size)]

    # Process in single-threaded mode if we have very few files or only one worker
    if total_files < 10 or num_workers == 1:
        for a3m_path in tqdm(a3m_paths, desc="Processing a3m files"):
            update_a3m(
                a3m_path=a3m_path,
                uniref_to_ncbi_taxid=uniref_to_ncbi_taxid,
                save_root=output_msa_dir,
            )
        return
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch tasks instead of individual files
        futures = []
        for batch in batches:
            future = executor.submit(
                update_a3m_batch,
                batch,
                uniref_to_ncbi_taxid,
                output_msa_dir
            )
            futures.append(future)

        # Track progress across all batches
        with tqdm(total=total_files, desc="Processing a3m files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Each result is a list of file paths processed in the batch
                    batch_size = future.result()
                    pbar.update(batch_size)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Estimate how many files might have been in this failed batch
                    avg_batch_size = total_files / len(batches)
                    pbar.update(int(avg_batch_size))


# Type alias for dictionary-like objects (regular dict or Manager.dict)
DictLike = Union[Dict[str, Any], Mapping[str, Any], SharedDict]


def load_mapping_data(seq_to_pdb_id_path: str, seq_to_pdb_index_path: str, use_shared_memory: bool = False):
    """
    Load mapping data from JSON files.

    Args:
        seq_to_pdb_id_path: Path to the seq_to_pdb_id_entity_id.json file
        seq_to_pdb_index_path: Path to the seq_to_pdb_index.json file
        use_shared_memory: Whether to use shared memory for dictionaries

    Returns:
        Tuple containing (seq_to_pdbid, first_pdbid_to_seq, seq_to_pdb_index) dictionaries
    """
    # Load sequence to PDB ID mapping
    with open(seq_to_pdb_id_path, "r", encoding='utf-8') as f:
        seq_to_pdbid: Dict[str, Any] = json.load(f)

    # Create reverse mapping for easy lookup
    first_pdbid_to_seq_data = {
        "_".join(v[0]): k for k, v in seq_to_pdbid.items()}

    # Load sequence to PDB index mapping
    with open(seq_to_pdb_index_path, "r", encoding='utf-8') as f:
        seq_to_pdb_index_data = json.load(f)

    # If using shared memory, convert the dictionaries to shared objects
    if use_shared_memory:
        # Create shared dictionaries
        first_pdbid_to_seq = convert_to_shared_dict(first_pdbid_to_seq_data)
        seq_to_pdb_index = convert_to_shared_dict(seq_to_pdb_index_data)

        print(
            f"Created shared memory dictionaries: {len(first_pdbid_to_seq)} PDB IDs,",
            f"{len(seq_to_pdb_index)} index mappings")
    else:
        first_pdbid_to_seq = first_pdbid_to_seq_data
        seq_to_pdb_index = seq_to_pdb_index_data

    return seq_to_pdbid, first_pdbid_to_seq, seq_to_pdb_index


def rematch(pdb_line: str, first_pdbid_to_seq: DictLike, seq_to_pdb_index: DictLike) -> Tuple[str, str]:
    """
    Match a PDB line to its corresponding sequence and index.

    Args:
        pdb_line: PDB header line
        first_pdbid_to_seq: Dictionary mapping PDB IDs to sequences
        seq_to_pdb_index: Dictionary mapping sequences to PDB indices

    Returns:
        Tuple of (pdb_index, origin_query_seq)
    """
    pdb_id = pdb_line[1:-1]
    origin_query_seq = first_pdbid_to_seq[pdb_id]
    pdb_index = seq_to_pdb_index[origin_query_seq]
    return pdb_index, origin_query_seq


def write_log(
    msg: str,
    fname: str,
    log_root: str,
) -> None:
    """
    Write a log message to a file with proper file locking to handle concurrency.

    Args:
        msg: Message to log
        fname: File name associated with the log
        log_root: Root directory for log files
    """
    basename = fname.split(".")[0]
    log_path = opjoin(log_root, f"{basename}-{msg}")

    # Create a directory for lock files
    lock_dir = opjoin(log_root, "locks")
    os.makedirs(lock_dir, exist_ok=True)

    # Use a separate lock file for each log file
    lock_path = opjoin(lock_dir, f"{basename}-{msg}.lock")

    try:
        # Open (or create) the lock file
        with open(lock_path, 'w', encoding='utf-8') as lock_file:
            # Acquire an exclusive lock (blocking)
            fcntl.flock(lock_file, fcntl.LOCK_EX)

            # Now safely create the log file
            with open(log_path, "w", encoding='utf-8') as f:
                f.write(msg)

            # The lock is automatically released when the file is closed
    except Exception as e:
        # If something goes wrong, log it but don't crash
        print(f"Warning: Failed to write log for {fname}: {e}")


def process_one_file(
    fname: str,
    msa_root: str,
    save_root: str,
    logger: Callable,
    first_pdbid_to_seq: DictLike,
    seq_to_pdb_index: DictLike
) -> None:
    """
    Process a single MSA file.

    Args:
        fname: Filename of the MSA file to process
        msa_root: Root directory containing MSA files
        save_root: Root directory to save processed files
        logger: Function to log events
        first_pdbid_to_seq: Dictionary mapping PDB IDs to sequences
        seq_to_pdb_index: Dictionary mapping sequences to PDB indices
    """
    pdb_line = None
    with open(file_path := opjoin(msa_root, fname), "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                pdb_line = line
            if i == 1:
                if len(line) == 1:
                    logger("empty_query_seq", fname)
                    return
                break

    save_fname, origin_query_seq = rematch(
        pdb_line, first_pdbid_to_seq, seq_to_pdb_index)

    os.makedirs(sub_dir_path := opjoin(
        save_root, f"{save_fname}"), exist_ok=True)
    uniref100_lines = [">query\n", f"{origin_query_seq}\n"]
    other_lines = [">query\n", f"{origin_query_seq}\n"]

    with open(file_path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if i < 2:
            continue
        if i % 2 == 0:
            # header
            if not line.startswith(">"):
                logger(f"bad_header_{i}", fname)
                return
            seq = lines[i + 1]

            if line.startswith(">UniRef100"):
                uniref100_lines.extend([line, seq])
            else:
                other_lines.extend([line, seq])

    assert len(other_lines) + len(uniref100_lines) - 2 == len(lines)

    other_lines = other_lines[0:2] + other_lines[4:]
    for i, line in enumerate(other_lines):
        if i > 0 and i % 2 == 0:
            assert "\t" in line
    with open(opjoin(
        sub_dir_path, "uniref100_hits.a3m"
    ), "w", encoding='utf-8') as f:
        for line in uniref100_lines:
            f.write(line)
    with open(opjoin(
        sub_dir_path, "mmseqs_other_hits.a3m"
    ), "w", encoding='utf-8') as f:
        for line in other_lines:
            f.write(line)


def process_file_batch(
    file_batch: List[str],
    msa_root: str,
    save_root: str,
    log_root: str,
    first_pdbid_to_seq: DictLike,
    seq_to_pdb_index: DictLike
) -> Set[str]:
    """
    Process a batch of MSA files.

    Args:
        file_batch: List of filenames to process in this batch
        msa_root: Root directory containing MSA files
        save_root: Root directory to save processed files
        log_root: Root directory for log files
        first_pdbid_to_seq: Dictionary mapping PDB IDs to sequences
        seq_to_pdb_index: Dictionary mapping sequences to PDB indices

    Returns:
        Set of files that were processed successfully
    """
    # Create a logger for this batch
    batch_logger = partial(write_log, log_root=log_root)

    # Track completed files
    completed_files = set()

    # Process each file in the batch
    for fname in file_batch:
        try:
            process_one_file(
                fname=fname,
                msa_root=msa_root,
                save_root=save_root,
                logger=batch_logger,
                first_pdbid_to_seq=first_pdbid_to_seq,
                seq_to_pdb_index=seq_to_pdb_index
            )
            completed_files.add(fname)
        except Exception as e:
            # Log any exceptions but continue processing the batch
            basename = fname.split(".")[0]
            error_path = opjoin(log_root, f"{basename}-exception")
            lock_dir = opjoin(log_root, "locks")
            lock_path = opjoin(lock_dir, f"{basename}-exception.lock")

            try:
                # Ensure lock directory exists
                os.makedirs(lock_dir, exist_ok=True)

                # Use file locking for the error log too
                with open(lock_path, 'w', encoding='utf-8') as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    with open(error_path, "w", encoding='utf-8') as f:
                        f.write(str(e))
                    # Lock is released when file is closed
            except Exception as log_error:
                # If locking fails, try direct write as fallback
                try:
                    with open(error_path, "w", encoding='utf-8') as f:
                        f.write(
                            f"{str(e)}\nAdditional error during logging: {str(log_error)}")
                except Exception:
                    # Last resort - print to stdout
                    print(
                        f"Error processing {fname} and failed to log: {str(e)}")

    return completed_files


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunked lists
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def process_files_batched(
    file_list: List[str],
    msa_root: str,
    save_root: str,
    log_root: str,
    first_pdbid_to_seq: DictLike,
    seq_to_pdb_index: DictLike,
    num_workers: int,
    batch_size: Optional[int],
) -> None:
    """
    Process files in batches using parallel processing.

    Args:
        file_list: List of files to process
        msa_root: Root directory containing MSA files
        save_root: Root directory to save processed files
        log_root: Root directory for log files
        first_pdbid_to_seq: Dictionary mapping PDB IDs to sequences (possibly shared)
        seq_to_pdb_index: Dictionary mapping sequences to PDB indices (possibly shared)
        num_workers: Number of parallel workers to use
        batch_size: Number of files to process in each batch
    """
    if num_workers is None:
        # Use a smaller number of workers to avoid excessive overhead
        num_workers = max(1, min(multiprocessing.cpu_count() - 1, 16))

    if batch_size is None:
        batch_size = max(1, len(file_list) // num_workers)

    # Split files into batches
    batches = chunk_list(file_list, batch_size)
    print(
        f"Split {len(file_list)} files into {len(batches)} batches of size ~{batch_size}")

    # Create a partial function with fixed arguments
    batch_processor = partial(
        process_file_batch,
        msa_root=msa_root,
        save_root=save_root,
        log_root=log_root,
        first_pdbid_to_seq=first_pdbid_to_seq,
        seq_to_pdb_index=seq_to_pdb_index
    )

    # Track progress and timing
    start_time = time.time()
    total_processed = 0

    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process batches with progress tracking
        with tqdm(total=len(file_list), desc="Processing MSA files") as pbar:
            futures = []

            # Submit all batches to the executor
            for batch in batches:
                futures.append(executor.submit(batch_processor, batch))

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    # Get the set of completed files
                    completed_files = future.result()
                    batch_count = len(completed_files)
                    total_processed += batch_count
                    pbar.update(batch_count)

                    # Calculate and display statistics
                    elapsed = time.time() - start_time
                    files_per_second = total_processed / elapsed if elapsed > 0 else 0
                    pbar.set_postfix(
                        {"processed": total_processed,
                            "files/sec": f"{files_per_second:.2f}"}
                    )
                except Exception as e:
                    print(f"Error processing batch: {e}")


def _process_cif_files(cif_files):
    """Process CIF files and extract sequence information.
    
    Args:
        cif_files: List of CIF file paths
        
    Returns:
        DataFrame with sequence information
    """
    info_dfs = []
    none_list = []
    error_list = []
    with Parallel(n_jobs=-2, verbose=10) as parallel:
        for pdb_id, info_df in parallel([delayed(try_get_seqs)(f) for f in cif_files]):
            if info_df is None:
                none_list.append(pdb_id)
            elif isinstance(info_df, str) and info_df.startswith("Error:"):
                error_list.append((pdb_id, info_df))
            else:
                info_dfs.append(info_df)

    out_df = pd.concat(info_dfs)
    return out_df.sort_values(["pdb_id", "entity_id"])


def _prepare_sequence_data(out_df, msa_out_dir):
    """Prepare sequence data and mappings.
    
    Args:
        out_df: DataFrame with sequence information
        msa_out_dir: Output directory for MSA
        
    Returns:
        Tuple of (sequence_mapping, fasta_file)
    """
    out_dir = Path(msa_out_dir + "/pdb_seqs")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # 1. extract pdb sequence info
    seq_file = out_dir / "pdb_seq.csv"
    seq_df = out_df[out_df.mol_type != "other"]
    seq_df.to_csv(seq_file, index=False)

    # 2. generate protein fasta file as MSA input
    fasta_file = out_dir / "pdb_seq.fasta"
    export_to_fasta(seq_df, fasta_file)

    # 3. get seq_to_pdb_id_entity_id mapping
    seq_to_pdb_id_entity_id_json = out_dir / "seq_to_pdb_id_entity_id.json"
    sequence_mapping = mapping_seqs_to_pdb_entity_id(
        seq_df, seq_to_pdb_id_entity_id_json
    )

    # 4. mapping sequence with integers identifiers for saving MSA.
    pdb_index_to_seq_path = out_dir / "pdb_index_to_seq.json"
    seq_to_pdb_index_path = out_dir / "seq_to_pdb_index.json"
    mapping_seqs_to_integer_identifiers(
        sequence_mapping, pdb_index_to_seq_path, seq_to_pdb_index_path
    )

    return sequence_mapping, fasta_file


def _convert_to_shared_memory_safe(uniref_to_ncbi_taxid, shared_memory):
    """Convert dictionary to shared memory if requested.
    
    Args:
        uniref_to_ncbi_taxid: Dictionary to convert
        shared_memory: Whether to use shared memory
        
    Returns:
        Original or shared dictionary
    """
    if not shared_memory:
        return uniref_to_ncbi_taxid

    try:
        print("Converting dictionary to shared memory for a3m processing...")
        result = convert_to_shared_dict(uniref_to_ncbi_taxid)
        print("Successfully converted dictionary to shared memory")
        return result
    except Exception as e:
        print(f"⚠️ Error converting to shared memory: {e}")
        return uniref_to_ncbi_taxid


def _release_shared_memory_safe(shared_memory):
    """Release all shared dictionaries safely.
    
    Args:
        shared_memory: Whether shared memory was used
    """
    if not shared_memory:
        return

    for dict_id in get_shared_dict_ids():
        try:
            release_shared_dict(dict_id)
        except Exception as e:
            print(f"Warning: Failed to release shared dict {dict_id}: {e}")


def get_msa(
    input_cif_dir,
    msa_out_dir,
    num_workers=None,
    batch_size=None,
    shared_memory=False,
    mp_read_workers=None,
    block_size_mb=64
):
    """
    get msa for training data preparation.
    Args:
        input_cif_dir: input cif directory
        msa_out_dir: output msa directory
        num_workers: number of workers
        batch_size: batch size
        shared_memory: whether to use shared memory
        mp_read_workers: number of worker processes for reading m8 file
        block_size_mb: size of each processing block in MB
    """
    cif_dir = Path(input_cif_dir)
    cif_files = [x for x in cif_dir.iterdir() if x.is_file()]

    out_df = _process_cif_files(cif_files)
    _, fasta_file = _prepare_sequence_data(out_df, msa_out_dir)

    seqs, _, pdb_name = get_queries(input_path=fasta_file)
    msa_search(
        seqs=seqs, msa_res_dir=f"{msa_out_dir}/mmcif_msa_initial", pdb_name=pdb_name)

    # Set up directories
    input_msa_dir = f"{msa_out_dir}/mmcif_msa_initial"
    output_msa_dir = f"{msa_out_dir}/mmcif_msa_with_taxid"
    os.makedirs(output_msa_dir, exist_ok=True)

    # Find input files
    a3m_paths = os.listdir(input_msa_dir)
    a3m_paths = [opjoin(input_msa_dir, x)
                 for x in a3m_paths if x.endswith(".a3m")]
    m8_file = f"{input_msa_dir}/uniref_tax.m8"

    # Read m8 file with improved parameters
    print(f"Reading m8 file with block size: {block_size_mb}MB")

    uniref_to_ncbi_taxid = read_m8(
        m8_file=m8_file,
        max_workers=mp_read_workers,
        block_size_mb=block_size_mb
    )

    print(
        f"Successfully read m8 file with {len(uniref_to_ncbi_taxid):,} entries")

    # Convert to shared memory if needed for a3m processing
    uniref_to_ncbi_taxid = _convert_to_shared_memory_safe(uniref_to_ncbi_taxid, shared_memory)

    # Process the a3m files
    print(f"Processing {len(a3m_paths)} a3m files...")
    process_files(
        a3m_paths=a3m_paths,
        uniref_to_ncbi_taxid=uniref_to_ncbi_taxid,
        output_msa_dir=output_msa_dir,
        num_workers=num_workers,
        batch_size=batch_size
    )

    # Release all shared dictionaries if necessary
    _release_shared_memory_safe(shared_memory)

    # Set start method to spawn to ensure compatibility with shared memory
    multiprocessing.set_start_method('spawn', force=True)

    msa_root = f"{msa_out_dir}/mmcif_msa_with_taxid"
    save_root = f"{msa_out_dir}/mmcif_msa"
    log_root = f"{msa_out_dir}/mmcif_msa_log"

    # Load mapping data
    print("Loading mapping data...")
    _, first_pdbid_to_seq, seq_to_pdb_index = load_mapping_data(
        f"{msa_out_dir}/pdb_seqs/seq_to_pdb_id_entity_id.json",
        f"{msa_out_dir}/pdb_seqs/seq_to_pdb_index.json",
        use_shared_memory=shared_memory
    )
    print("Mapping data loaded successfully")

    os.makedirs(log_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    print("Loading file names...")
    file_list = os.listdir(msa_root)
    print(f"Found {len(file_list)} MSA files to process")

    # Process files in batches
    process_files_batched(
        file_list=file_list,
        msa_root=msa_root,
        save_root=save_root,
        log_root=log_root,
        first_pdbid_to_seq=first_pdbid_to_seq,
        seq_to_pdb_index=seq_to_pdb_index,
        num_workers=num_workers,
        batch_size=batch_size
    )

    # Release all shared dictionaries if necessary
    _release_shared_memory_safe(shared_memory)

    print("Processing complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_dir", type=str,
                        default="./scripts/msa/data/mmcif")
    parser.add_argument("--out_dir", type=str, default="./scripts/msa/data")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes for a3m processing. Defaults to auto.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Number of a3m files per batch. Defaults to auto.")
    parser.add_argument("--shared_memory", action="store_true",
                        help="Use shared memory for dictionary to reduce memory usage.")
    parser.add_argument("--mp_read_workers", type=int, default=None,
                        help="Number of worker processes for reading m8 file. Defaults to auto.")
    parser.add_argument("--block_size_mb", type=int, default=64,
                        help="Size of each processing block in MB (smaller blocks can improve parallelism).")
    args = parser.parse_args()

    get_msa(
        input_cif_dir=args.cif_dir,
        msa_out_dir=args.out_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shared_memory=args.shared_memory,
        mp_read_workers=args.mp_read_workers,
        block_size_mb=args.block_size_mb
    )

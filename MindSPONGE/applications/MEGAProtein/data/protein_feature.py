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
protein feature generation module.
"""

import numpy as np
from absl import logging

from mindsponge.data.data_transform import convert_monomer_features, convert_unnecessary_leading_dim_feats
from mindsponge.common import residue_constants
from data.templates import TemplateHitFeaturizer
from data.hhsearch import HHSearch
from data.msa_query import MmseqQuery
from data.multimer_pipeline import add_assembly_features, pair_and_merge, pad_msa
from data.parsers import parse_fasta, parse_hhr, parse_a3m


def make_msa_features(msas, deletion_matrices):
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError('At least one MSA must be provided.')

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)
    features = {'deletion_matrix_int': np.array(deletion_matrix, dtype=np.int32),
                'deletion_matrix_int_all_seq': np.array(deletion_matrix, dtype=np.int32),
                'msa': np.array(int_msa, dtype=np.int32),
                'msa_all_seq': np.array(int_msa, dtype=np.int32),
                'num_alignments': np.array([num_alignments] * num_res, dtype=np.int32),
                'msa_species_identifiers_all_seq': np.array([b''] * num_alignments)}
    return features


def make_sequence_features(sequence: str, description: str, num_res: int):
    """Constructs a feature dict of sequence features."""
    features = {'aatype': residue_constants.sequence_to_onehot(sequence=sequence,
                                                               mapping=residue_constants.restype_order_with_x,
                                                               map_unknown_to_x=True),
                'between_segment_residues': np.zeros((num_res,), dtype=np.int32),
                'domain_name': np.array([description.encode('utf-8')], dtype=np.object_),
                'residue_index': np.array(range(num_res), dtype=np.int32),
                'seq_length': np.array([num_res] * num_res, dtype=np.int32),
                'sequence': np.array([sequence.encode('utf-8')], dtype=np.object_)}
    return features


class RawFeatureGenerator:
    """Runs the alignment tools"""

    def __init__(self,
                 template_mmcif_dir,
                 max_template_date,
                 kalign_binary_path,
                 obsolete_pdbs_path,
                 hhsearch_binary_path,
                 pdb70_database_path,
                 database_envdb_dir,
                 mmseqs_binary,
                 uniref30_path,
                 a3m_result_path,
                 max_hits=20,
                 msa_length=512):
        """Search the a3m info for a given FASTA file."""

        self.template_mmcif_dir = template_mmcif_dir
        self.max_template_date = max_template_date
        self.kalign_binary_path = kalign_binary_path
        self.obsolete_pdbs_path = obsolete_pdbs_path
        self.hhsearch_binary_path = hhsearch_binary_path
        self.pdb70_database_path = pdb70_database_path
        self.a3m_result_path = a3m_result_path
        self.max_hits = max_hits
        self.msa_length = msa_length
        self.msa_query = MmseqQuery(database_envdb_dir=database_envdb_dir, mmseqs_binary=mmseqs_binary,
                                    uniref30_path=uniref30_path, result_path=a3m_result_path)
        self.hhsearch_pdb70_runner = HHSearch(binary_path=hhsearch_binary_path, databases=[pdb70_database_path])


    def monomer_feature_generate(self, fasta_path):
        """protein raw feature generation"""
        template_featurizer = TemplateHitFeaturizer(mmcif_dir=self.template_mmcif_dir,
                                                    max_template_date=self.max_template_date,
                                                    max_hits=self.max_hits,
                                                    kalign_binary_path=self.kalign_binary_path,
                                                    release_dates_path=None,
                                                    obsolete_pdbs_path=self.obsolete_pdbs_path)
        with open(fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parse_fasta(input_fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(f'More than one input sequence found in {fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]

        num_res = len(input_sequence)
        a3m_lines = self.msa_query.aligned_a3m_files(fasta_path, self.a3m_result_path)

        hhsearch_result = self.hhsearch_pdb70_runner.query(a3m_lines)
        hhsearch_hits = parse_hhr(hhsearch_result)

        msas, deletion_matrices = parse_a3m(a3m_lines)
        templates_result = template_featurizer.get_templates(
            query_sequence=input_sequence,
            query_pdb_code=None,
            query_release_date=None,
            hhr_hits=hhsearch_hits)
        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res)
        msa_features = make_msa_features(msas=(msas,), deletion_matrices=(deletion_matrices,))

        feature_dict = {**sequence_features, **msa_features, **templates_result.features}
        return feature_dict

    def multimer_feature_generate(self, fasta_paths: list):
        """ multimer feature preprocess.

        Args:
            fasta_paths: a list path of fasta, each fasta for one chain fasta sequence file

        Return:
            multimer_feature: a combined feature for multi_chain protein

        """
        if len(fasta_paths) == 1:
            logging.error("get only one fasta, will return monomer feature")
            return self.monomer_feature_generate(fasta_paths[0])
        all_chain_features = {}
        for id_, fasta_path_ in enumerate(fasta_paths):
            chain_feature = self.monomer_feature_generate(fasta_path_)
            chain_feature["chain_id"], chain_feature["aatype"], chain_feature["template_aatype"] = \
                convert_monomer_features(str(id_), chain_feature["aatype"], chain_feature["template_aatype"])
            sequence, domain_name, num_alignments, seq_length = \
                convert_unnecessary_leading_dim_feats(chain_feature["sequence"], chain_feature["domain_name"],
                                                      chain_feature["num_alignments"], chain_feature["seq_length"])
            chain_feature["sequence"] = sequence
            chain_feature["domain_name"] = domain_name
            chain_feature["num_alignments"] = num_alignments
            chain_feature["seq_length"] = seq_length

            all_chain_features[str(id_)] = chain_feature
        all_chain_features = add_assembly_features(all_chain_features)
        combined_features = pair_and_merge(all_chain_features)
        combined_features = pad_msa(combined_features, self.msa_length)

        return combined_features

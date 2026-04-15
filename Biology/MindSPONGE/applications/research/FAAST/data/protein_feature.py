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
import os
import stat
import pickle
import numpy as np
from data.hhsearch import HHSearch
from data.msa_query import MmseqQuery
from data.parsers import parse_fasta, parse_hhr, parse_a3m
from data.templates import TemplateHitFeaturizer
from mindsponge.common import residue_constants
from search import colabsearch


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

    def __init__(self, database_search_config, a3m_path, templatepath, use_custom, use_template, max_hits=20,
                 msa_length=512):
        """Search the a3m info for a given FASTA file."""

        self.template_path = templatepath
        self.use_template = use_template
        self.template_mmcif_dir = f"{self.template_path}/"
        self.max_template_date = database_search_config.max_template_date
        self.kalign_binary_path = database_search_config.kalign_binary_path
        self.hhsearch_binary_path = database_search_config.hhsearch_binary_path
        self.pdb70_database_path = f"{self.template_path}/pdb70"
        self.a3m_result_path = a3m_path
        self.database_envdb_dir = database_search_config.database_envdb_dir
        self.mmseqs_binary = database_search_config.mmseqs_binary
        self.uniref30_path = database_search_config.uniref30_path
        self.max_hits = max_hits
        self.msa_length = msa_length

        self.msa_query = MmseqQuery(database_envdb_dir=self.database_envdb_dir,
                                    mmseqs_binary=self.mmseqs_binary,
                                    uniref30_path=self.uniref30_path,
                                    result_path=self.a3m_result_path)
        self.use_custom = use_custom

    def monomer_feature_generate(self, fasta_path, prot_name):
        """protein raw feature generation"""
        with open(fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parse_fasta(input_fasta_str)
        if not self.use_custom:
            colabsearch(input_seqs, self.a3m_result_path, self.template_path)
        if self.use_template:
            hhsearch_pdb70_runner = HHSearch(binary_path=self.hhsearch_binary_path,
                                             databases=[self.pdb70_database_path])
            template_featurizer = TemplateHitFeaturizer(mmcif_dir=self.template_mmcif_dir,
                                                        max_template_date=self.max_template_date,
                                                        max_hits=self.max_hits,
                                                        kalign_binary_path=self.kalign_binary_path,
                                                        release_dates_path=None)
        if len(input_seqs) != 1:
            raise ValueError(f'More than one input sequence found in {fasta_path}.')
        input_sequence = input_seqs[0]
        input_description = input_descs[0]

        num_res = len(input_sequence)

        a3m_lines = self.msa_query.aligned_a3m_files(self.a3m_result_path)

        if self.use_template:
            hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
            hhsearch_hits = parse_hhr(hhsearch_result)
            templates_result = template_featurizer.get_templates(
                query_sequence=input_sequence,
                query_pdb_code=None,
                query_release_date=None,
                hhr_hits=hhsearch_hits)

        msas, deletion_matrices = parse_a3m(a3m_lines)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res)
        msa_features = make_msa_features(msas=(msas,), deletion_matrices=(deletion_matrices,))
        features = {}
        shape0 = 20
        shape1 = 22
        shape2 = 37
        shape4 = 3
        shape5 = 1
        features["template_aatype"] = np.zeros((shape0, num_res, shape1))
        features["template_all_atom_masks"] = np.zeros((shape0, num_res, shape2))
        features["template_all_atom_positions"] = np.zeros((shape0, num_res, shape2, shape4))
        features["template_domain_names"] = np.zeros((shape0,))
        features["template_e_value"] = np.zeros((shape0, shape5))
        features["template_neff"] = np.zeros((shape0, shape5))
        features["template_prob_true"] = np.zeros((shape0, shape5))
        features["template_similarity"] = np.zeros((shape0, shape5))
        features["template_sequence"] = np.zeros((shape0, shape5))
        features["template_sum_probs"] = np.zeros((shape0, shape5))
        features["template_confidence_scores"] = np.zeros((shape0, num_res))
        if self.use_template:
            features = templates_result.features

        feature_dict = {**sequence_features, **msa_features, **features}
        os.makedirs("./pkl_file/", exist_ok=True)
        os_flags = os.O_RDWR | os.O_CREAT
        os_modes = stat.S_IRWXU
        with os.fdopen(os.open(f"./pkl_file/{prot_name}.pkl", os_flags, os_modes), "wb") as fout:
            pickle.dump(feature_dict, fout)
            f.close()
        return feature_dict

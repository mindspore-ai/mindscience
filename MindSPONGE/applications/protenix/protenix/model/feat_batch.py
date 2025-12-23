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

"""Batch dataclass."""

import dataclasses
import mindspore as ms
from protenix.model.features import MSA, TokenFeatures, RefStructure, AtomCrossAtt


@dataclasses.dataclass
class Batch:
    """Dataclass containing batch."""
    msa: MSA=None
    token_features: TokenFeatures=None
    ref_structure: RefStructure=None
    atom_cross_att: AtomCrossAtt=None

    def load_from_dict(self, data: dict):
        """Load batch from dictionary."""
        self.msa = MSA()
        self.token_features = TokenFeatures()
        self.ref_structure = RefStructure()
        self.atom_cross_att = AtomCrossAtt()
        self.msa.rows = data["input_feature_dict"]["msa"].astype(ms.float32)
        self.msa.mask = data["input_feature_dict"]["has_deletion"].astype(ms.float32)
        self.msa.deletion_matrix = data["input_feature_dict"]["deletion_value"].astype(ms.float32)
        self.token_bonds = data["input_feature_dict"]["token_bonds"].astype(ms.float32)
        self.token_features.asym_id = data["input_feature_dict"]["asym_id"].astype(ms.float32)
        self.token_features.residue_index = data["input_feature_dict"]["residue_index"].astype(ms.float32)
        self.token_features.entity_id = data["input_feature_dict"]["entity_id"].astype(ms.float32)
        self.token_features.sym_id = data["input_feature_dict"]["sym_id"].astype(ms.float32)
        self.token_features.token_index = data["input_feature_dict"]["token_index"].astype(ms.float32)
        self.token_features.mask = None
        self.ref_structure.positions = data["input_feature_dict"]["ref_pos"].astype(ms.float32)
        self.ref_structure.charge = data["input_feature_dict"]["ref_charge"].astype(ms.float32)
        self.ref_structure.mask = data["input_feature_dict"]["ref_mask"].astype(ms.float32)
        self.ref_structure.ref_space_uid = data["input_feature_dict"]["ref_space_uid"].astype(ms.float32)
        self.ref_structure.element = data["input_feature_dict"]["ref_element"].astype(ms.float32)
        self.ref_structure.atom_name_chars = data["input_feature_dict"]["ref_atom_name_chars"].astype(ms.float32)
        self.ref_structure.restype = data["input_feature_dict"]["restype"].astype(ms.float32)
        self.ref_structure.profile = data["input_feature_dict"]["profile"].astype(ms.float32)
        self.ref_structure.deletion_mean = data["input_feature_dict"]["deletion_mean"].astype(ms.float32)
        self.atom_cross_att.token_atoms_to_queries = data["input_feature_dict"]["atom_to_token_idx"].astype(ms.int32)
        self.distogram_rep_atom_mask = data["input_feature_dict"]["distogram_rep_atom_mask"].astype(ms.bool_)
        self.atom_to_token_idx = data["input_feature_dict"]["atom_to_token_idx"].astype(ms.int32)
        self.atom_to_tokatom_idx = data["input_feature_dict"]["atom_to_tokatom_idx"].astype(ms.int32)
        self.num_tokens = len(data["input_feature_dict"]["token_index"])
        self.has_frame = data["input_feature_dict"]["has_frame"].astype(ms.int32)
        self.is_ligand = data["input_feature_dict"]["is_ligand"].astype(ms.int32)
        self.mol_id = data["input_feature_dict"]["mol_id"].astype(ms.int32)
        self.entity_mol_id = data["input_feature_dict"]["entity_mol_id"].astype(ms.int32)
        self.mol_atom_index = data["input_feature_dict"]["mol_atom_index"].astype(ms.int32)
        self.frames_mask = data["input_feature_dict"]["has_frame"]
        self.frame_atom_index = data["input_feature_dict"]["frame_atom_index"].astype(ms.int32)
        self.bond_mask = data["input_feature_dict"]["bond_mask"]
        self.is_dna = data["input_feature_dict"]["is_dna"]
        self.is_rna = data["input_feature_dict"]["is_rna"]
        self.is_ligand = data["input_feature_dict"]["is_ligand"]
        self.pae_rep_atom_mask = data["input_feature_dict"]["pae_rep_atom_mask"]
        self.resolution = data["input_feature_dict"]["resolution"]
        self.plddt_m_rep_atom_mask = data["input_feature_dict"]["plddt_m_rep_atom_mask"]

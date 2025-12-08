# Copyright 2025 Huawei Technologies Co., Ltd
#
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Batch dataclass."""

import dataclasses
from typing import Self
import mindspore as ms
from mindspore import Tensor
from alphafold3.model import features


@dataclasses.dataclass
class Batch:
    """Dataclass containing batch."""

    msa: features.MSA
    templates: features.Templates
    token_features: features.TokenFeatures
    ref_structure: features.RefStructure
    predicted_structure_info: features.PredictedStructureInfo
    polymer_ligand_bond_info: features.PolymerLigandBondInfo
    ligand_ligand_bond_info: features.LigandLigandBondInfo
    pseudo_beta_info: features.PseudoBetaInfo
    atom_cross_att: features.AtomCrossAtt
    convert_model_output: features.ConvertModelOutput
    frames: features.Frames

    @property
    def num_res(self) -> int:
        """Number of residues."""
        return self.token_features.aatype.shape[-1]

    @staticmethod
    def gather_to_tensor(input_feat):
        """Convert gather indices to tensor."""
        input_feat.gather_idxs = Tensor(input_feat.gather_idxs)
        input_feat.gather_mask = Tensor(input_feat.gather_mask)
        input_feat.input_shape = Tensor(input_feat.input_shape)

    @classmethod
    def from_data_dict(cls, batch: features.BatchDict) -> Self:
        """Construct batch object from dictionary."""
        return cls(
            msa=features.MSA.from_data_dict(batch),
            templates=features.Templates.from_data_dict(batch),
            token_features=features.TokenFeatures.from_data_dict(batch),
            ref_structure=features.RefStructure.from_data_dict(batch),
            predicted_structure_info=features.PredictedStructureInfo.from_data_dict(
                batch
            ),
            polymer_ligand_bond_info=features.PolymerLigandBondInfo.from_data_dict(
                batch
            ),
            ligand_ligand_bond_info=features.LigandLigandBondInfo.from_data_dict(
                batch
            ),
            pseudo_beta_info=features.PseudoBetaInfo.from_data_dict(batch),
            atom_cross_att=features.AtomCrossAtt.from_data_dict(batch),
            convert_model_output=features.ConvertModelOutput.from_data_dict(
                batch),
            frames=features.Frames.from_data_dict(batch),
        )

    def as_data_dict(self) -> features.BatchDict:
        """Converts batch object to dictionary."""
        output = {
            **self.msa.as_data_dict(),
            **self.templates.as_data_dict(),
            **self.token_features.as_data_dict(),
            **self.ref_structure.as_data_dict(),
            **self.predicted_structure_info.as_data_dict(),
            **self.polymer_ligand_bond_info.as_data_dict(),
            **self.ligand_ligand_bond_info.as_data_dict(),
            **self.pseudo_beta_info.as_data_dict(),
            **self.atom_cross_att.as_data_dict(),
            **self.convert_model_output.as_data_dict(),
            **self.frames.as_data_dict(),
        }
        return output

    def convert_to_tensor(self, dtype=ms.float32):
        """Convert all fields to tensor."""
        # msa: features.MSA
        self.msa.rows = Tensor(self.msa.rows, dtype=ms.int32)
        self.msa.mask = Tensor(self.msa.mask, dtype=ms.int32)
        self.msa.deletion_matrix = Tensor(
            self.msa.deletion_matrix, dtype=dtype)
        self.msa.deletion_mean = Tensor(self.msa.deletion_mean, dtype=dtype)
        self.msa.profile = Tensor(self.msa.profile, dtype=dtype)
        self.msa.num_alignments = Tensor(
            self.msa.num_alignments, dtype=ms.int32)
        # templates: features.Templates
        self.templates.aatype = Tensor(self.templates.aatype, dtype=ms.int32)
        self.templates.atom_mask = Tensor(
            self.templates.atom_mask, dtype=ms.int32)
        self.templates.atom_positions = Tensor(
            self.templates.atom_positions, dtype=dtype)
        # token_features: features.TokenFeatures
        self.token_features.mask = Tensor(
            self.token_features.mask, dtype=ms.int32)
        self.token_features.token_index = Tensor(
            self.token_features.mask, dtype=ms.int32)
        self.token_features.asym_id = Tensor(
            self.token_features.asym_id, dtype=ms.int32)
        self.token_features.aatype = Tensor(
            self.token_features.aatype, dtype=ms.int32)
        self.token_features.residue_index = Tensor(
            self.token_features.residue_index, dtype=ms.int32)
        self.token_features.entity_id = Tensor(
            self.token_features.entity_id, dtype=ms.int32)
        self.token_features.sym_id = Tensor(
            self.token_features.sym_id, dtype=ms.int32)
        # ref_structure: features.RefStructure
        self.ref_structure.positions = Tensor(
            self.ref_structure.positions, dtype=dtype)
        self.ref_structure.mask = Tensor(self.ref_structure.mask, dtype=dtype)
        self.ref_structure.element = Tensor(
            self.ref_structure.element, dtype=ms.int32)
        self.ref_structure.charge = Tensor(
            self.ref_structure.charge, dtype=dtype)
        self.ref_structure.atom_name_chars = Tensor(
            self.ref_structure.atom_name_chars, dtype=ms.int32)
        self.ref_structure.ref_space_uid = Tensor(
            self.ref_structure.ref_space_uid, dtype=dtype)

        # predicted_structure_info: features.PredictedStructureInfo
        self.predicted_structure_info.atom_mask = Tensor(
            self.predicted_structure_info.atom_mask, dtype=dtype)

        # polymer_ligand_bond_info: features.PolymerLigandBondInfo
        self.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs = Tensor(
            self.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_idxs, dtype=ms.int32
        )
        self.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask = Tensor(
            self.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds.gather_mask, dtype=ms.int32
        )
        # ligand_ligand_bond_info: features.LigandLigandBondInfo
        self.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs = Tensor(
            self.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_idxs, dtype=ms.int32
        )
        self.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask = Tensor(
            self.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds.gather_mask, dtype=ms.int32
        )

        self.gather_to_tensor(self.pseudo_beta_info.token_atoms_to_pseudo_beta)
        self.gather_to_tensor(self.atom_cross_att.queries_to_keys)
        self.gather_to_tensor(self.atom_cross_att.tokens_to_queries)
        self.gather_to_tensor(self.atom_cross_att.tokens_to_keys)
        self.gather_to_tensor(self.atom_cross_att.token_atoms_to_queries)
        self.gather_to_tensor(self.atom_cross_att.queries_to_token_atoms)

        # frames: features.Frames

    def astype(self, dtype=ms.float32):
        """Change dtype of float."""
        # change dtype of float
        # msa: features.MSA
        self.msa.deletion_matrix = self.msa.deletion_matrix.astype(dtype)
        self.msa.deletion_mean = self.msa.deletion_mean.astype(dtype)
        self.msa.profile = self.msa.profile.astype(dtype)
        # templates: features.Templates
        self.templates.atom_positions = self.templates.atom_positions.astype(
            dtype)
        # ref_structure: features.RefStructure
        self.ref_structure.positions = self.ref_structure.positions.astype(
            dtype)
        self.ref_structure.mask = self.ref_structure.mask.astype(dtype)
        self.ref_structure.charge = self.ref_structure.charge.astype(dtype)
        self.ref_structure.ref_space_uid = self.ref_structure.ref_space_uid.astype(
            dtype)

        # predicted_structure_info: features.PredictedStructureInfo
        self.predicted_structure_info.atom_mask = self.predicted_structure_info.atom_mask.astype(
            dtype)

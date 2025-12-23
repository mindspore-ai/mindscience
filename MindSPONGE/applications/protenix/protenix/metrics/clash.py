# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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
Clash detection module.
"""

import logging
from typing import Optional

import mindspore as ms
from mindspore import nn, mint

from protenix.data.constants import rdkit_vdws

RDKIT_VDWS = ms.Tensor(rdkit_vdws)
ID2TYPE = {0: "UNK", 1: "lig", 2: "prot", 3: "dna", 4: "rna"}


def get_vdw_radii(elements_one_hot):
    """get vdw radius for each atom according to their elements"""
    element_order = elements_one_hot.argmax(dim=1)
    return RDKIT_VDWS[element_order]


class Clash(nn.Cell):
    """
    Clash detection class.
    """
    def __init__(
        self,
        af3_clash_threshold=1.1,
        vdw_clash_threshold=0.75,
        compute_af3_clash=True,
        compute_vdw_clash=True,
    ):
        super().__init__()
        self.af3_clash_threshold = af3_clash_threshold
        self.vdw_clash_threshold = vdw_clash_threshold
        self.compute_af3_clash = compute_af3_clash
        self.compute_vdw_clash = compute_vdw_clash

    def construct(
        self,
        pred_coordinate,
        asym_id,
        atom_to_token_idx,
        is_ligand,
        is_protein,
        is_dna,
        is_rna,
        mol_id: Optional[ms.Tensor] = None,
        elements_one_hot: Optional[ms.Tensor] = None,
    ):
        """
        Calculate clash statistics.
        """
        chain_info = self.get_chain_info(
            asym_id=asym_id,
            atom_to_token_idx=atom_to_token_idx,
            is_ligand=is_ligand,
            is_protein=is_protein,
            is_dna=is_dna,
            is_rna=is_rna,
            mol_id=mol_id,
            elements_one_hot=elements_one_hot,
        )
        return self._check_clash_per_chain_pairs(
            pred_coordinate=pred_coordinate, **chain_info
        )

    def get_chain_info(
        self,
        asym_id,
        atom_to_token_idx,
        is_ligand,
        is_protein,
        is_dna,
        is_rna,
        mol_id: Optional[ms.Tensor] = None,
        elements_one_hot: Optional[ms.Tensor] = None,
    ):
        """
        Get chain information.
        """
        # Get chain info
        asym_id = asym_id.long()
        asym_id_to_asym_mask = {
            aid.item(): asym_id == aid for aid in mint.unique(asym_id)
        }
        n_chains = len(asym_id_to_asym_mask)
        # Make sure it is from 0 to n_chains-1
        if n_chains != asym_id.max() + 1:
            raise ValueError("n_chains must be equal to asym_id.max() + 1")

        # Check and compute chain_types
        chain_types = []
        mol_id_to_asym_ids, asym_id_to_mol_id = {}, {}
        atom_type = (1 * is_ligand + 2 * is_protein + 3 * is_dna + 4 * is_rna).long()
        if self.compute_vdw_clash:
            if mol_id is None:
                raise ValueError("mol_id must be provided")
            if elements_one_hot is None:
                raise ValueError("elements_one_hot must be provided")

        for aid in range(n_chains):
            atom_chain_mask = asym_id_to_asym_mask[aid][atom_to_token_idx]
            atom_type_i = atom_type[atom_chain_mask]
            if len(atom_type_i.unique()) != 1:
                raise ValueError("atom_type_i must be unique")
            if atom_type_i[0].item() == 0:
                logging.warning(
                    "Unknown asym_id type: not in ligand / protein / dna / rna"
                )
            chain_types.append(ID2TYPE[atom_type_i[0].item()])
            if self.compute_vdw_clash:
                # Check if all atoms in a chain are from the same molecule
                mol_id_i = mol_id[atom_chain_mask].unique().item()
                mol_id_to_asym_ids.setdefault(mol_id_i, []).append(aid)
                asym_id_to_mol_id[aid] = mol_id_i

        chain_info = {
            "n_chains": n_chains,
            "atom_to_token_idx": atom_to_token_idx,
            "asym_id_to_asym_mask": asym_id_to_asym_mask,
            "atom_type": atom_type,
            "mol_id": mol_id,
            "elements_one_hot": elements_one_hot,
            "chain_types": chain_types,
        }

        if self.compute_vdw_clash:
            chain_info.update({"asym_id_to_mol_id": asym_id_to_mol_id})

        return chain_info

    def get_chain_pair_violations(
        self,
        pred_coordinate,
        violation_type,
        chain_1_mask,
        chain_2_mask,
        elements_one_hot: Optional[ms.Tensor] = None,
    ):
        """
        Get chain pair violations.
        """
        chain_1_coords = pred_coordinate[chain_1_mask, :]
        chain_2_coords = pred_coordinate[chain_2_mask, :]
        pred_dist = mint.cdist(chain_1_coords, chain_2_coords)
        # pred_dist = mint.sum(mint.square(chain_1_coords- chain_2_coords),
        #             dim=-1, keepdim=False).sqrt().astype(chain_1_coords.dtype)
        if violation_type == "af3":
            clash_per_atom_pair = (
                pred_dist < self.af3_clash_threshold
            )  # [ N_atom_chain_1, N_atom_chain_2]
            clashed_col, clashed_row = mint.where(clash_per_atom_pair)
            clash_atom_pairs = mint.stack((clashed_col, clashed_row), dim=-1)
        else:
            if elements_one_hot is None:
                raise ValueError("elements_one_hot must be provided")
            vdw_radii_i, vdw_radii_j = get_vdw_radii(
                elements_one_hot[chain_1_mask, :]
            ), get_vdw_radii(elements_one_hot[chain_2_mask, :])
            vdw_sum_pair = (
                vdw_radii_i[:, None] + vdw_radii_j[None, :]
            )  # [N_atom_chain_1, N_atom_chain_2]
            relative_vdw_distance = pred_dist / vdw_sum_pair
            clash_per_atom_pair = (
                relative_vdw_distance < self.vdw_clash_threshold
            )  # [N_atom_chain_1, N_atom_chain_2]
            clashed_col, clashed_row = mint.where(clash_per_atom_pair)
            clash_rel_dist = relative_vdw_distance[clashed_col, clashed_row]
            clashed_global_col = mint.where(chain_1_mask)[0][clashed_col]
            clashed_global_row = mint.where(chain_2_mask)[0][clashed_row]
            clash_atom_pairs = mint.stack(
                (clashed_global_col, clashed_global_row, clash_rel_dist), dim=-1
            )
        return clash_atom_pairs

    def _init_clash_results(self, n_sample, n_chains):
        """
        Initialize clash results.
        """
        results = {}
        if self.compute_af3_clash:
            results["has_af3_clash_flag"] = mint.zeros(
                (n_sample, n_chains, n_chains), dtype=ms.bool_
            )
            results["af3_clash_details"] = mint.zeros(
                (n_sample, n_chains, n_chains, 2), dtype=ms.bool_
            )
        else:
            results["has_af3_clash_flag"] = None
            results["af3_clash_details"] = None

        if self.compute_vdw_clash:
            results["has_vdw_clash_flag"] = mint.zeros(
                (n_sample, n_chains, n_chains), dtype=ms.bool_
            )
            results["vdw_clash_details"] = {}
        else:
            results["has_vdw_clash_flag"] = None
            results["vdw_clash_details"] = None

        results["skipped_pairs"] = []
        return results

    def _process_pair(
        self,
        sample_id,
        i,
        j,
        pred_coordinate,
        atom_to_token_idx,
        chain_types,
        asym_id_to_asym_mask,
        asym_id_to_mol_id,
        elements_one_hot,
        n_chain_i,
        atom_chain_mask_i,
        results,
    ):
        """
        Process a pair of chains.
        """
        if chain_types[j] == "UNK":
            return

        chain_pair_type = set([chain_types[i], chain_types[j]])
        # Skip potential bonded ligand to polymers
        skip_bonded_ligand = False
        if (
            self.compute_vdw_clash
            and "lig" in chain_pair_type
            and len(chain_pair_type) > 1
            and asym_id_to_mol_id[i] == asym_id_to_mol_id[j]
        ):
            common_mol_id = asym_id_to_mol_id[i]
            logging.warning(
                "mol_id %s may contain bonded ligand to polymers", common_mol_id
            )
            skip_bonded_ligand = True
            results["skipped_pairs"].append((i, j))

        atom_chain_mask_j = asym_id_to_asym_mask[j][atom_to_token_idx]
        n_chain_j = mint.sum(atom_chain_mask_j).item()

        if self.compute_vdw_clash and not skip_bonded_ligand:
            vdw_clash_pairs = self.get_chain_pair_violations(
                pred_coordinate=pred_coordinate[sample_id, :, :],
                violation_type="vdw",
                chain_1_mask=atom_chain_mask_i,
                chain_2_mask=atom_chain_mask_j,
                elements_one_hot=elements_one_hot,
            )
            if vdw_clash_pairs.shape[0] > 0:
                results["vdw_clash_details"][(sample_id, i, j)] = vdw_clash_pairs
                results["has_vdw_clash_flag"][sample_id, i, j] = True
                results["has_vdw_clash_flag"][sample_id, j, i] = True

        if chain_types[i] == "lig" or chain_types[j] == "lig":
            # AF3 clash only consider polymer chains
            return

        if self.compute_af3_clash:
            af3_clash_pairs = self.get_chain_pair_violations(
                pred_coordinate=pred_coordinate[sample_id, :, :],
                violation_type="af3",
                chain_1_mask=atom_chain_mask_i,
                chain_2_mask=atom_chain_mask_j,
            )
            total_clash = af3_clash_pairs.shape[0]
            relative_clash = total_clash / min(n_chain_i, n_chain_j)
            results["af3_clash_details"][sample_id, i, j, 0] = total_clash
            results["af3_clash_details"][sample_id, i, j, 1] = relative_clash
            results["has_af3_clash_flag"][sample_id, i, j] = (
                total_clash > 100 or relative_clash > 0.5
            )
            results["af3_clash_details"][sample_id, j, i, :] = results[
                "af3_clash_details"
            ][sample_id, i, j, :]
            results["has_af3_clash_flag"][sample_id, j, i] = results[
                "has_af3_clash_flag"
            ][sample_id, i, j]

    def _check_clash_per_chain_pairs(
        self,
        pred_coordinate,
        atom_to_token_idx,
        n_chains,
        chain_types,
        elements_one_hot,
        asym_id_to_asym_mask,
        asym_id_to_mol_id: Optional[ms.Tensor] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Check clash per chain pairs.
        """
        n_sample = pred_coordinate.shape[0]

        # initialize results
        results = self._init_clash_results(n_sample, n_chains)

        for sample_id in range(n_sample):
            for i in range(n_chains):
                if chain_types[i] == "UNK":
                    continue
                atom_chain_mask_i = asym_id_to_asym_mask[i][atom_to_token_idx]
                n_chain_i = mint.sum(atom_chain_mask_i).item()
                for j in range(i + 1, n_chains):
                    self._process_pair(
                        sample_id,
                        i,
                        j,
                        pred_coordinate,
                        atom_to_token_idx,
                        chain_types,
                        asym_id_to_asym_mask,
                        asym_id_to_mol_id,
                        elements_one_hot,
                        n_chain_i,
                        atom_chain_mask_i,
                        results,
                    )

        return {
            "summary": {
                "af3_clash": results["has_af3_clash_flag"],
                "vdw_clash": results["has_vdw_clash_flag"],
                "chain_types": chain_types,
                "skipped_pairs": results["skipped_pairs"],
            },
            "details": {
                "af3_clash": results["af3_clash_details"],
                "vdw_clash": results["vdw_clash_details"],
            },
        }

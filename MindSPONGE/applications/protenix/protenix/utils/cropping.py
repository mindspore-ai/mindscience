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

"""cropping"""

import copy
import random
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import mindspore as ms
from mindspore import mint
from biotite.structure import AtomArray
from scipy.spatial.distance import cdist

from protenix.data.tokenizer import TokenArray


def identify_mol_type(
    ref_space_uid: ms.Tensor,
    atom_sums: ms.Tensor,
    chain_id: ms.Tensor,
    chain_lengths: ms.Tensor,
) -> ms.Tensor:
    """
    Generate mol_type masks based on the given rules.

    Args:
        ref_space_uid (ms.Tensor): A tensor of unique ids, shape (N,).
        atom_sums (ms.Tensor): A tensor of atom sums corresponding to each unique id, shape (N,).
        chain_id (ms.Tensor): A tensor of chain IDs corresponding to each unique id, shape (N,).
        chain_lengths (ms.Tensor): A tensor of chain lengths, shape (num_chains,).

    Returns:
        is_metal (ms.Tensor): A mask indicating metals.
        first_indices (ms.Tensor): A tensor of first indices for each unique id, shape (N,).
        last_indices (ms.Tensor): A tensor of last indices for each unique id, shape (N,).
    """

    if (
        ref_space_uid.shape != atom_sums.shape
    ):
        raise ValueError(
            "ref_space_uid and atom_sums must have the same shape.")
    # Initialize masks
    is_metal = mint.zeros(ref_space_uid.shape, dtype=ms.bool_)
    first_indices = mint.zeros(ref_space_uid.shape, dtype=ms.int32)
    last_indices = mint.zeros(ref_space_uid.shape, dtype=ms.int32)

    # Count occurrences of each ref_space_uid
    unique_ids, counts = mint.unique(ref_space_uid, return_counts=True)
    for unique_id, count in zip(unique_ids, counts):
        mask = ref_space_uid == unique_id
        first_index = mask.nonzero(as_tuple=False)[0].item()
        last_index = mask.nonzero(as_tuple=False)[-1].item()
        first_indices[mask] = first_index
        last_indices[mask] = last_index
        atom_sum = atom_sums[mask]

        if count == 1 and chain_lengths[chain_id[mask].long()] == 1:
            is_metal[mask] = atom_sum == 1

    return (
        is_metal,
        first_indices,
        last_indices,
    )


def get_interface_token(
    chain_id: ms.Tensor,
    reference_chain_id: ms.Tensor,
    token_distance: ms.Tensor,
    token_distance_mask: ms.Tensor,
    interface_minimal_distance: int = 15,
) -> ms.Tensor:
    """
    Get tokens in contact with the other chain.
    Args:
        chain_id:           [all_token_length, ], chain ID of each token
        reference_chain_id: [1] or [2], the reference atom is selected within the reference chains
        token_distance:     [chain/interface_token_length, all_token_length], 
        distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask:[chain/interface_token_length, all_token_length], indicates valid distance
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        interface_token_indices: indices of tokens of interface
    """
    # expand reference_chain_id to chain_id shape
    expand_reference_chain_id = mint.zeros(chain_id.shape[0], dtype=ms.int32)
    for chain_id_value in reference_chain_id:
        expand_reference_chain_id += chain_id == int(chain_id_value)

    # get distance mask, difference chain mask
    mask_distance = token_distance < interface_minimal_distance
    mask_diff_chain = (chain_id[None, :] != chain_id[:, None])[
        expand_reference_chain_id.nonzero(as_tuple=True)[0]
    ]

    mask = mask_distance * mask_diff_chain * token_distance_mask
    mask_interface = mint.sum(mask, dim=-1)
    interface_token_indices = mint.nonzero(mask_interface, as_tuple=True)[0]
    return interface_token_indices


def get_spatial_crop_index(
    tokens: ms.Tensor,
    chain_id: ms.Tensor,
    token_distance: ms.Tensor,
    token_distance_mask: ms.Tensor,
    reference_chain_id: ms.Tensor,
    ref_space_uid_token: ms.Tensor,
    crop_size: int,
    crop_complete_ligand_unstd_res: bool = False,
    interface_crop: bool = False,
    interface_minimal_distance: int = 15,
) -> ms.Tensor:
    """
    Crop sequences continuesly across chains.
    Args:
        tokens:   [all_token_length,], all tokens within an assembly
        chain_id: [all_token_length,], all tokens' chain ID within an assembly
        token_distance: [chain/interface_token_length, all_token_length], 
        distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask: [chain/interface_token_length, all_token_length], indicates valid distance
        reference_chain_id:  [1] or [2],the reference atom is selected within the reference_chains ID
        crop_size: total crop size of the whole assembly
        interface_crop: whether use interface tokens as referenced token
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        selected_token_indices: ms.Tensor, shape=(min(crop_size, tokens.shape[0]), )
    """

    # interface spatial cropping: select reference tokens with contact to the other
    if interface_crop and interface_minimal_distance is not None:
        reference_token_indices = get_interface_token(
            chain_id=chain_id,
            reference_chain_id=reference_chain_id,
            token_distance=token_distance,
            token_distance_mask=token_distance_mask,
            interface_minimal_distance=interface_minimal_distance,
        )
        if len(reference_token_indices) < 1 and len(reference_chain_id) == 1:
            # If a chain does not contain any interfacial atoms, use all resolved tokens.
            reference_token_indices = mint.nonzero(
                token_distance_mask.bool().any(-1), as_tuple=True
            )[0]
    else:
        # select reference tokens within the given chain or interface
        reference_token_indices = mint.nonzero(
            token_distance_mask.bool().any(-1), as_tuple=True
        )[0]

    # random select one token from reference_token_indices
    if len(reference_token_indices) <= 0:
        raise ValueError("No resolved atoms in reference tokens!")

    random_idx = np.random.randint(
        0, reference_token_indices.shape[0], (1,)).item()
    reference_token_idx = reference_token_indices[random_idx].item()

    if not token_distance_mask[reference_token_idx].bool().any():
        raise ValueError("Select a unresolved reference token")
    distance_to_reference = token_distance[reference_token_idx]
    # add noise to break tie
    noise_break_tie = mint.arange(
        0, distance_to_reference.shape[0]).float() * 1e-3

    distance_to_reference_mask = token_distance_mask[reference_token_idx]
    distance_to_reference = mint.where(
        distance_to_reference_mask.bool(), distance_to_reference, 1e10
    )

    # find k nearest tokens
    nearest_k = min(crop_size, tokens.shape[0])
    selected_token_indices = (
        mint.topk(distance_to_reference + noise_break_tie,
                  nearest_k, largest=False)[1]
        .sort()[0]
    )

    def drop_uncompleted_mol(selected_token_indices):
        selected_uid = ref_space_uid_token[selected_token_indices]
        mask = mint.ones_like(ref_space_uid_token, dtype=ms.bool_)
        mask[selected_token_indices] = False
        unselected_uid = ref_space_uid_token[mask]

        # Find overlap elements
        overlap_uid = ms.Tensor(np.intersect1d(selected_uid, unselected_uid))

        # Remove overlap elements from elements_B
        remain_indices = selected_token_indices[
            ~ms.numpy.isin(selected_uid, overlap_uid)
        ].long()
        return remain_indices

    selected_token_indices = mint.flatten(selected_token_indices)
    if crop_complete_ligand_unstd_res is True:
        selected_token_indices = drop_uncompleted_mol(selected_token_indices)
    if selected_token_indices.shape[0] > crop_size:
        raise ValueError(
            f"Spatial cropping crop {selected_token_indices.shape[0]}, more than {crop_size} tokens!!")
    return selected_token_indices, reference_token_idx


def _qualify_crop_size(cur_crop_size, crop_size_min, n_added, crop_size):
    if cur_crop_size < crop_size_min:
        return False
    if cur_crop_size + n_added > crop_size:
        return False
    return True


def _get_boundary_points(
    start_idx, end_idx, uid_first_indices, uid_last_indices
):
    """Get left and right boundary points for start and end indices."""
    left_start_point = right_start_point = start_idx
    start_in_middle = False
    if uid_first_indices[start_idx] != start_idx:
        start_in_middle = True
        left_start_point = uid_first_indices[start_idx]
        right_start_point = uid_last_indices[start_idx] + 1

    left_end_point = right_end_point = end_idx
    end_in_middle = False
    if end_idx > 0 and uid_last_indices[end_idx - 1] != end_idx - 1:
        end_in_middle = True
        left_end_point = uid_first_indices[end_idx - 1]
        right_end_point = uid_last_indices[end_idx - 1] + 1

    return (
        left_start_point,
        right_start_point,
        left_end_point,
        right_end_point,
        start_in_middle,
        end_in_middle,
    )


def _select_end_point_with_middle_end(
    start_idx,
    left_end_point,
    right_end_point,
    crop_size_min,
    n_added,
    crop_size,
    drop_last,
    uid_first_indices,
):
    """Select end point when end is in middle of uid."""
    left_crop_size = left_end_point - start_idx
    right_crop_size = right_end_point - start_idx
    is_left_ok = _qualify_crop_size(
        left_crop_size, crop_size_min, n_added, crop_size)
    is_right_ok = _qualify_crop_size(
        right_crop_size, crop_size_min, n_added, crop_size
    )

    if is_left_ok and is_right_ok:
        return (
            left_end_point
            if np.random.randint(low=0, high=2, size=(1,)).item() == 0
            else right_end_point
        )
    if is_left_ok:
        return left_end_point
    if is_right_ok:
        return right_end_point
    if drop_last is True:
        end_point = left_end_point
        while end_point - start_idx + n_added > crop_size:
            if end_point > start_idx:
                end_point = uid_first_indices[end_point - 1]
            else:
                break
        return end_point

    cur_crop_size = min(right_end_point - start_idx, crop_size - n_added)
    return start_idx + cur_crop_size


def _select_start_point_with_middle_start(
    end_idx,
    left_start_point,
    right_start_point,
    crop_size_min,
    n_added,
    crop_size,
    drop_last,
):
    """Select start point when start is in middle of uid."""
    left_crop_size = end_idx - left_start_point
    right_crop_size = end_idx - right_start_point
    is_left_ok = _qualify_crop_size(
        left_crop_size, crop_size_min, n_added, crop_size)
    is_right_ok = _qualify_crop_size(
        right_crop_size, crop_size_min, n_added, crop_size
    )

    if is_left_ok and is_right_ok:
        return (
            left_start_point
            if np.random.randint(low=0, high=2, size=(1,)).item() == 0
            else right_start_point
        )
    if is_left_ok:
        return left_start_point
    if is_right_ok:
        return right_start_point
    if drop_last is True:
        return right_start_point
    return left_start_point


def _determine_start_end_point(
    start_idx,
    end_idx,
    crop_size_min,
    n_added,
    crop_size,
    drop_last,
    uid_first_indices,
    uid_last_indices,
):
    """Determine the start and end points for the crop."""
    if start_idx == end_idx:
        return start_idx, end_idx

    (
        left_start_point,
        right_start_point,
        left_end_point,
        right_end_point,
        start_in_middle,
        end_in_middle,
    ) = _get_boundary_points(
        start_idx, end_idx, uid_first_indices, uid_last_indices
    )

    if not start_in_middle and not end_in_middle:
        return start_idx, end_idx

    if start_in_middle and end_in_middle:
        # Always use left edge
        start_in_middle = False
        start_idx = left_start_point

    if not start_in_middle and end_in_middle:
        end_idx = _select_end_point_with_middle_end(
            start_idx,
            left_end_point,
            right_end_point,
            crop_size_min,
            n_added,
            crop_size,
            drop_last,
            uid_first_indices,
        )
        return start_idx, end_idx
    if start_in_middle and not end_in_middle:
        start_idx = _select_start_point_with_middle_start(
            end_idx,
            left_start_point,
            right_start_point,
            crop_size_min,
            n_added,
            crop_size,
            drop_last,
        )
        return start_idx, end_idx

    return start_idx, end_idx


def get_continues_crop_index(
    tokens: ms.Tensor,
    chain_id: ms.Tensor,
    ref_space_uid_token: ms.Tensor,
    atom_sums: ms.Tensor,
    crop_size: int,
    crop_complete_ligand_unstd_res: Optional[bool] = False,
    drop_last: Optional[bool] = False,
    remove_metal: Optional[bool] = False,
) -> ms.Tensor:
    """
    Crop sequences continuesly across chains. Reference: AF-multimer Algorithm 1.
    Args:
        tokens:    [all_token_length,], flatten tokens
        chain_id:  [all_token_length,], all tokens' chain ID within an assembly
        atom_sums: [all_token_length,] sum of atoms within one ref_space_uid
        ref_space_uid: [all_atom_length,] unique chain-residue id
        crop_size: total crop size of the whole assembly
        crop_complete_ligand_unstdRes: Whether to crop the complete ligand or unstandard residues.
                              If False, the ligand is usually fragmented during sequential cropping.
        drop_last: whether to ensure all ligands or unstandard residues to be cropped completely,
                    if not, we will ignore the completion of the last one to meet the crop_size quota.
        remove_metal: whether remove all metal/ions
    Returns:
        selected_token_indices: ms.Tensor, shape=(crop_size, )
    """
    # get chain counts info
    unique_chain_id = mint.unique(chain_id)
    chain_lengths = mint.bincount(chain_id.long())
    chain_offset_list = ms.Tensor(
        [mint.where(chain_id == chain_idx)[0][0]
         for chain_idx in unique_chain_id],
    )

    # identify the mol type
    (
        is_metal, uid_first_indices, uid_last_indices,
    ) = identify_mol_type(ref_space_uid_token, atom_sums, chain_id, chain_lengths)

    # shuffle the list of chains
    chain_shuffle_index = ms.Tensor(
        np.random.permutation(len(unique_chain_id)))

    # crop over chains iteratively
    selected_token_indices = []
    n_added = 0  # number of tokens already selected
    n_remaining = len(tokens)  # number of tokens in remaining chains
    if remove_metal is True:
        n_remaining -= sum(is_metal).item()
    for idx in chain_shuffle_index:
        if n_added >= crop_size:
            break

        # get chain type: whether it is metal/ions
        curr_is_metal = is_metal[chain_offset_list[idx]]
        # whether remove metal chain
        if remove_metal is True and curr_is_metal:
            # skip if it is metal/ions
            continue

        chain_length = chain_lengths[unique_chain_id[idx].int()]
        n_remaining -= chain_length

        # determine the crop size
        crop_size_min = min(chain_length, max(
            0, crop_size - (n_added + n_remaining)))
        crop_size_max = min(crop_size - n_added, chain_length)
        if crop_size_min > crop_size_max:
            print(f"error crop_size: {crop_size_min} > {crop_size_max}")

        chain_crop_size = np.random.randint(
            int(crop_size_min), int(crop_size_max + 1), size=(1,),
        ).item()

        chain_crop_start = np.random.randint(
            low=0,
            high=int(chain_length - chain_crop_size + 1),
            size=(1,),
        ).item()

        chain_offset = chain_offset_list[idx]
        start_token_index = chain_offset + chain_crop_start
        end_token_index = chain_offset + chain_crop_start + chain_crop_size
        if crop_complete_ligand_unstd_res is True:
            start_token_index, end_token_index = _determine_start_end_point(
                start_token_index, end_token_index,
                crop_size_min, n_added,
                crop_size, drop_last,
                uid_first_indices, uid_last_indices,
            )
            if end_token_index < start_token_index:
                raise ValueError(
                    f"invalid crop indices!! {start_token_index}, {end_token_index}")
            chain_crop_size = end_token_index - start_token_index

        selected_token_indices.append(
            mint.arange(
                int(start_token_index),
                int(end_token_index),
            )
        )
        n_added += chain_crop_size
        if crop_complete_ligand_unstd_res is True and drop_last is True:
            if start_token_index < end_token_index:
                if uid_first_indices[start_token_index] != start_token_index:
                    raise ValueError(
                        "uid_first_indices[start_token_index] must be equal to start_token_index")
                if uid_last_indices[end_token_index - 1] != end_token_index - 1:
                    raise ValueError(
                        "uid_last_indices[end_token_index - 1] must be equal to end_token_index - 1")

    selected_token_indices = mint.concat(selected_token_indices).sort()[0]
    selected_token_indices = mint.flatten(selected_token_indices)
    if drop_last is True:
        if selected_token_indices.shape[0] > crop_size:
            raise ValueError(f"Continuous cropping crop {selected_token_indices.shape[0]}",
                             f"more than {crop_size} tokens!!")
    return selected_token_indices


class CropData:
    """
    Crop the data based on the given crop size and reference chain indices (asym_id).
    """

    def __init__(
        self,
        crop_size: int,
        ref_chain_indices: list[int],
        token_array: TokenArray,
        atom_array: AtomArray,
        method_weights: list[float] = None,
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
    ) -> None:
        """
        Args:
            crop_size (int): The size of the crop to be sampled.
            ref_chain_indices (list[int]): The "asym_id_int" of the reference chains.
            token_array (TokenArray): The token array.
            atom_array (AtomArray): The atom array.
            method_weights (list[float]): The weights corresponding to these three cropping methods:
                                          ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"].
            contiguous_crop_complete_lig: Whether to crop the complete ligand in ContiguousCropping method.

        """
        self.crop_size = crop_size
        self.ref_chain_indices = ref_chain_indices
        self.token_array = token_array
        self.atom_array = atom_array
        self.method_weights = method_weights if method_weights is not None else [
            0.2, 0.4, 0.4]

        self.cand_crop_methods = [
            "ContiguousCropping",
            "SpatialCropping",
            "SpatialInterfaceCropping",
        ]
        self.contiguous_crop_complete_lig = contiguous_crop_complete_lig
        self.spatial_crop_complete_lig = spatial_crop_complete_lig
        self.drop_last = drop_last
        self.remove_metal = remove_metal

    def random_crop_method(self) -> str:
        """
        Choose a random cropping method based on the given weights.

        Returns:
            str: The name of the randomly selected cropping method.
        """
        return random.choices(self.cand_crop_methods, k=1, weights=self.method_weights)[
            0
        ]

    def get_token_dist_mat(self, token_indices_in_ref: np.ndarray) -> np.ndarray:
        """
        Get the distance matrix of the tokens in the reference chain.

        Args:
            token_indices_in_ref (list): The indices of the tokens in the reference chain.

        Returns:
            numpy.ndarray: The distance matrix of the tokens in the reference chain,
                           shape=(len(tokens_in_ref_chain), len(tokens)).
        """
        centre_atom_indices = self.token_array.get_annotation(
            "centre_atom_index")
        centre_atom_coords = self.atom_array.coord[centre_atom_indices]

        partial_token_dist_matrix = cdist(
            centre_atom_coords[token_indices_in_ref],
            centre_atom_coords,
            "euclidean",
        )

        if partial_token_dist_matrix.shape != (
            len(token_indices_in_ref),
            len(self.token_array),
        ):
            raise ValueError("partial_token_dist_matrix.shape must be",
                             f"{len(token_indices_in_ref)}, {len(self.token_array)})")
        return partial_token_dist_matrix

    def extract_info(
        self,
    ) -> tuple[ms.Tensor, ms.Tensor, ms.Tensor, list[int]]:
        """
        Extract information from the token array and atom array.

        Returns:
            tuple: A tuple containing the following elements:
                - tokens (ms.Tensor): The token array.
                - chain_id (ms.Tensor): The chain IDs of the atoms.
                - token_dist_mask_1d (ms.Tensor): The distance mask of the tokens.
                - token_indices_in_ref (list[int]): The indices of the tokens in the reference chain.
                - is_ligand (ms.Tensor): Whether chain type is ligand.
        """
        tokens = self.token_array.get_values()
        chain_id = []
        token_dist_mask_1d = []
        token_indices_in_ref = []

        token_centre_atom_indices = self.token_array.get_annotation(
            "centre_atom_index")
        centre_atoms = self.atom_array[token_centre_atom_indices]
        chain_id = centre_atoms.asym_id_int
        token_dist_mask_1d = centre_atoms.is_resolved
        token_indices_in_ref = np.where(
            np.isin(centre_atoms.asym_id_int, self.ref_chain_indices)
        )[0]
        is_ligand = centre_atoms.is_ligand

        tokens = ms.Tensor(tokens)
        chain_id = ms.Tensor(chain_id)
        token_dist_mask_1d = ms.Tensor(token_dist_mask_1d)
        is_ligand = ms.Tensor(is_ligand)
        return tokens, chain_id, token_dist_mask_1d, token_indices_in_ref, is_ligand

    def crop_by_indices(
        self,
        selected_token_indices: ms.Tensor,
        msa_features: dict[str, np.ndarray] = None,
        template_features: dict[str, np.ndarray] = None,
    ) -> tuple[TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop the token array, atom array, msa features and template features based on the selected token indices.
        """
        return self.select_by_token_indices(
            token_array=self.token_array,
            atom_array=self.atom_array,
            selected_token_indices=selected_token_indices,
            msa_features=msa_features,
            template_features=template_features,
        )

    @staticmethod
    def select_by_token_indices(
        token_array: TokenArray,
        atom_array: AtomArray,
        selected_token_indices: ms.Tensor,
        msa_features: dict[str, np.ndarray] = None,
        template_features: dict[str, np.ndarray] = None,
    ) -> tuple[TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop the token array, atom array, msa features and template features based on the selected token indices.

        Args:
            token_array (TokenArray): the input token array
            atom_array (AtomArray): the input atom array
            selected_token_indices (ms.Tensor): The indices of the tokens to be cropped.
            msa_feature (dict[str, np.ndarray]): The MSA features.
            template_feature (dict[str, np.ndarray]): The Template features.

        Returns:
            cropped_token_array (TokenArray): The cropped token array.
            cropped_atom_array (AtomArray): The cropped atom array.
            cropped_msa_features (dict[str, np.ndarray]): The cropped msa features.
            cropped_template_features (dict[str, np.ndarray]): The cropped template features.
        """
        cropped_token_array = copy.deepcopy(
            token_array[selected_token_indices])

        cropped_atom_indices = []
        totol_atom_num = 0
        for _, token in enumerate(cropped_token_array):
            cropped_atom_indices.extend(token.atom_indices)
            centre_idx_in_token_atoms = token.atom_indices.index(
                token.centre_atom_index
            )
            token_atom_num = len(token.atom_indices)
            token.atom_indices = list(
                range(totol_atom_num, totol_atom_num + token_atom_num)
            )
            token.centre_atom_index = token.atom_indices[centre_idx_in_token_atoms]
            totol_atom_num += token_atom_num

        cropped_atom_array = copy.deepcopy(atom_array[cropped_atom_indices])
        if len(cropped_token_array) != selected_token_indices.shape[0]:
            raise ValueError(
                "len(cropped_token_array) must be equal to selected_token_indices.shape[0]")

        selected_token_indices_list = selected_token_indices.tolist()
        # crop msa
        cropped_msa_features = {}
        if msa_features is not None:
            for k, v in msa_features.items():
                if k in ["profile", "deletion_mean"]:
                    cropped_msa_features[k] = v[selected_token_indices_list]
                elif k in ["msa", "has_deletion", "deletion_value"]:
                    cropped_msa_features[k] = v[:, selected_token_indices]
                elif k in [
                    "prot_pair_num_alignments",
                    "prot_unpair_num_alignments",
                    "rna_pair_num_alignments",
                    "rna_unpair_num_alignments",
                ]:
                    # keep the feature that do not need crop
                    cropped_msa_features[k] = v
        # crop template
        cropped_template_features = {}
        if template_features is not None:
            for k, v in template_features.items():
                if k == "template_restype":
                    cropped_template_features[k] = v[:,
                                                     selected_token_indices_list]
                elif k == "template_all_atom_mask":
                    cropped_template_features[k] = v[:,
                                                     selected_token_indices_list, :]
                elif k == "template_all_atom_positions":
                    cropped_template_features[k] = v[:,
                                                     selected_token_indices_list, :, :]
                else:
                    raise ValueError(
                        f"Cropping for {k} has not been implemented yet")

        return (
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
        )

    def get_crop_indices(self, crop_method: str = None) -> ms.Tensor:
        """
        Get selected indices based on the selected crop method.

        Args:
            crop_method (str): The cropping method to be used. Default is None.
        Returns:
            selected_indices : ms.Tensor, shape=(N_selected, )
        """
        tokens, chain_id, token_dist_mask_1d, token_indices_in_ref, _ = (
            self.extract_info()
        )

        if crop_method not in self.cand_crop_methods:
            raise ValueError(f"Unknown crop method: {crop_method}")

        # add token level ref_space_uid
        ref_space_uid_token = self.atom_array.ref_space_uid[
            self.token_array.get_annotation("centre_atom_index")
        ]

        atom_num_in_tokens = []
        for token in self.token_array:
            atom_num_in_tokens.append(len(token.atom_indices))

        uid_num_dict = defaultdict(int)
        for idx, uid in enumerate(ref_space_uid_token):
            uid_num_dict[uid] += atom_num_in_tokens[idx]
        atom_sums = ms.Tensor(
            [uid_num_dict[uid] for idx, uid in enumerate(ref_space_uid_token)]
        )
        if not (atom_sums > 0).all().item():
            raise ValueError("zero atoms")

        ref_space_uid_token = ms.Tensor(ref_space_uid_token)

        if crop_method == "ContiguousCropping":
            selected_token_indices = get_continues_crop_index(
                tokens=tokens,
                chain_id=chain_id,
                ref_space_uid_token=ref_space_uid_token,
                atom_sums=atom_sums,
                crop_size=self.crop_size,
                crop_complete_ligand_unstd_res=self.contiguous_crop_complete_lig,
                drop_last=self.drop_last,
                remove_metal=self.remove_metal,
            )
            reference_token_index = -1

        else:
            interface_crop = crop_method == "SpatialInterfaceCropping"
            token_distance = self.get_token_dist_mat(
                token_indices_in_ref=token_indices_in_ref
            )
            token_indices_in_ref = ms.Tensor(token_indices_in_ref)
            token_distance_mask = (
                token_dist_mask_1d[token_indices_in_ref][:, None]
                * token_dist_mask_1d[None, :]
            )
            selected_token_indices, reference_token_index = get_spatial_crop_index(
                tokens=tokens,
                chain_id=chain_id,
                token_distance=ms.Tensor(token_distance),
                token_distance_mask=ms.Tensor(token_distance_mask),
                reference_chain_id=self.ref_chain_indices,
                ref_space_uid_token=ref_space_uid_token,
                crop_size=self.crop_size,
                crop_complete_ligand_unstd_res=self.spatial_crop_complete_lig,
                interface_crop=interface_crop,
            )
        return (
            selected_token_indices,
            token_indices_in_ref[reference_token_index].item(),
        )

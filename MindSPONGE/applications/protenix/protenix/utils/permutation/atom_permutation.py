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

"""atom permutation"""

import mindspore as ms
from mindspore import _no_grad, mint

from protenix.metrics.rmsd import rmsd, self_aligned_rmsd
from protenix.model.loss import expand_at_dim
from protenix.model.modules.utils import pad_at_dim
from protenix.utils.logger import get_logger


logger = get_logger(__name__)


def run(
    pred_coord: ms.Tensor,
    true_coord: ms.Tensor,
    true_coord_mask: ms.Tensor,
    ref_space_uid: ms.Tensor,
    atom_perm_list: ms.Tensor,
    permute_label: bool = True,
    alignment_mask: ms.Tensor = None,
    dataset_name: str = None,
    pdb_id: str = None,
    global_align_wo_symmetric_atom: bool = False,
):
    """apply a permutation to correct symmetric atoms in residues.

    Args:
        Please refer to the args of `correct_symmetric_atoms`.

    Returns:
        if permute_label = True,
            output_dict: a dictionary in the following form, recording the permuted label.
                {
                    "coordinate": permuted_coord,
                    "coordinate_mask": permuted_mask,
                }
            info_dict: a dictionary of logging.
        if permute_label = False,
            output_dict: a dictionary in the following form, recording the permuted prediction.
                {
                    "coordinate": permuted_coord,
                }
            info_dict: a dictionary of logging.
    """
    try:
        permuted_coord, permuted_mask, info_dict, indices_permutation = (
            correct_symmetric_atoms(
                pred_coord=pred_coord,
                true_coord=true_coord,
                true_coord_mask=true_coord_mask,
                ref_space_uid=ref_space_uid,
                atom_perm_list=atom_perm_list,
                permute_label=permute_label,
                alignment_mask=alignment_mask,
                global_align_wo_symmetric_atom=global_align_wo_symmetric_atom,
            )
        )
        if permute_label:
            return (
                {
                    "coordinate": permuted_coord,
                    "coordinate_mask": permuted_mask,
                },
                info_dict,
                indices_permutation,
            )
        return {"coordinate": permuted_coord}, info_dict, indices_permutation

    except Exception as e:
        error_message = str(e)
        if dataset_name:
            logger.warning(f"dataset: {dataset_name}")
        if pdb_id:
            logger.warning(f"pdb id: {pdb_id}")
        logger.warning(error_message)

        return {}, {}, None


def collect_residues_with_symmetric_atoms(
    coord: ms.Tensor,
    coord_mask: ms.Tensor,
    ref_space_uid: ms.Tensor,
    atom_perm_list: list[list],
) -> tuple[list]:
    """Convert atom-level permutation attributes to residue-level attributes.
    Only residues that require symmetric corrections are returned.

    Args:
        coord (ms.Tensor): Coordinates of atoms.
            [num_atom, 3]
        coord_mask (ms.Tensor): The mask indicating whether the atom is resolved in GT.
            [num_atom]
        ref_space_uid (ms.Tensor): Each (chain id, residue index) tuple has a unique ID.
            [num_atom]
        atom_perm_list (list[list]): The atom permutation list, where each sublist contains
                                   the permutation information of the corresponding residue.
            len(atom_perm_list) = num_atom.
            len(atom_perm_list[i]) = num_perm for the residue of atom i.

    """

    # Find start & end positions of each residue
    diff = ms.Tensor([True] + (ref_space_uid[1:] !=
                     ref_space_uid[:-1]).tolist())
    start_positions = mint.cat(
        (mint.nonzero(diff, as_tuple=True)[0], ms.Tensor([len(ref_space_uid)]))
    )
    res_start_end = list(
        zip(start_positions[:-1].tolist(), start_positions[1:].tolist())
    )  # [num_residue, 2]
    num_residue = len(res_start_end)
    if num_residue != len(mint.unique(ref_space_uid)):
        raise ValueError("num_residue must be equal to len(mint.unique(ref_space_uid))")

    position_list = []
    perm_list = []
    coord_list = []
    coord_mask_list = []

    # Traverse residues and store the corresponding data
    for start, end in res_start_end:

        if len(mint.unique(ref_space_uid[start:end])) != 1:
            raise ValueError("len(mint.unique(ref_space_uid[start:end])) must be 1")

        # Skip if this residue contains < 3 resolved atoms.
        # Alignment requires at least 3 atoms to obtain a reasonable result.
        res_coord_mask = coord_mask[start:end].bool()  # [num_residue_atom]
        if res_coord_mask.sum() < 3:
            continue

        # Drop duplicated permutations
        perm = ms.Tensor(atom_perm_list[start:end], dtype=ms.int32)
        perm = mint.unique(perm, dim=-1)  # [num_residue_atom, num_perm]
        num_residue_atom, _ = perm.shape

        # Basic checks
        if perm.min().item() != 0:
            raise ValueError("perm.min().item() must be 0")
        if perm.max().item() != num_residue_atom - 1:
            raise ValueError("perm.max().item() must be num_residue_atom - 1")

        # If all symmetric atoms are unresolved, drop the permutation
        identity_perm = mint.arange(len(perm)).unsqueeze(
            dim=-1
        ).astype(ms.int32)  # [num_residue_atom, 1]
        is_sym_atom = perm != identity_perm  # [num_residue_atom, num_perm]
        is_sym_atom_resolved = is_sym_atom * res_coord_mask.unsqueeze(dim=-1)
        is_valid_perm = is_sym_atom_resolved.any(dim=0)
        if not is_valid_perm.any():
            # Skip if no valid permutation (other than identity) exists
            continue
        perm = perm[..., is_valid_perm]
        # Put identity to the first
        perm = mint.cat([identity_perm, perm], dim=-1)
        perm = perm.transpose(-1, -2)  # [num_perm, num_residue_atom]

        position_list.append((start, end))
        perm_list.append(perm)
        coord_mask_list.append(res_coord_mask)
        coord_list.append(coord[start:end, :])

    return position_list, coord_list, coord_mask_list, perm_list, num_residue


def collect_permuted_coords(
    coord_list: list[ms.Tensor],
    coord_mask_list: list[ms.Tensor],
    perm_list: list[ms.Tensor],
) -> tuple[ms.Tensor]:
    """Apply permutations to coordinates and coordinate masks

    Args:
        coord_list (list[ms.Tensor]): A list of coordinates.
            Each element is a tensor of shape [num_residue_atom, 3]. The value num_residue_atom can
            vary across different residues.
        coord_mask_list (list[ms.Tensor]): A list of coordinate masks.
            Each element is a tensor of shape [num_residue_atom].
        perm_list (list[ms.Tensor]): list of permutations.
            Each element is a long tensor of shape [num_perm, num_residue_atom]. The value num_perm
            can vary across different residues.

    Returns:
        ms.Tensor:
            [num_total_perm, max_num_residue_atom, 3]
            [num_total_perm, max_num_residue_atom]
    """

    max_num_residue_atom = max(perm.shape[-1] for perm in perm_list)
    perm_coord = []  # [num_total_perm, num_residue_atom, 3]
    perm_coord_mask = []  # [num_total_perm, num_residue_atom]

    num_total_perm = 0
    for perm, res_coord, res_coord_mask in zip(perm_list, coord_list, coord_mask_list):

        # Basic shape checks
        num_perm, num_residue_atom = perm.shape
        if res_coord.shape[-1] != 3:
            raise ValueError("res_coord.shape[-1] must be 3")
        if res_coord.shape[0] != res_coord_mask.shape[0] != perm.shape[-1]:
            raise ValueError("res_coord.shape[0] must be equal to res_coord_mask.shape[0]",
                             "must be equal to perm.shape[-1]")

        # Permute coordinates & masks
        res_coord_permuted = res_coord[perm]  # [num_perm, num_residue_atom, 3]
        res_coord_mask_permuted = res_coord_mask[perm].astype(
            ms.int32)  # [num_perm, num_residue_atom]
        if res_coord_permuted.shape != (num_perm, num_residue_atom, 3):
            raise ValueError("res_coord_permuted.shape must be (num_perm, num_residue_atom, 3)")
        if res_coord_mask_permuted.shape != (num_perm, num_residue_atom):
            raise ValueError("res_coord_mask_permuted.shape must be (num_perm, num_residue_atom)")

        # Pad to MAX_num_residue_atom
        num_residue_atom = perm.size(dim=-1)
        if num_residue_atom < max_num_residue_atom:
            pad_length = (0, max_num_residue_atom - num_residue_atom)
            res_coord_permuted = pad_at_dim(
                res_coord_permuted, dim=-2, pad_length=pad_length
            )  # [num_perm, MAX_num_residue_atom, 3]
            res_coord_mask_permuted = pad_at_dim(
                res_coord_mask_permuted, dim=-1, pad_length=pad_length
            )

        num_total_perm += num_perm
        perm_coord.append(res_coord_permuted)
        perm_coord_mask.append(res_coord_mask_permuted)

    perm_coord = mint.cat(perm_coord, dim=0)
    perm_coord_mask = mint.cat(perm_coord_mask, dim=0)

    # Shape check
    if perm_coord.shape != (num_total_perm, max_num_residue_atom, 3):
        raise ValueError("perm_coord.shape must be (num_total_perm, max_num_residue_atom, 3)")
    if perm_coord_mask.shape != (num_total_perm, max_num_residue_atom):
        raise ValueError("perm_coord_mask.shape must be (num_total_perm, max_num_residue_atom)")

    return perm_coord, perm_coord_mask


class AtomPermutation:
    """
    Class for assigning the optimal permutations of true coordinates/pred coordinates and coordinate masks.
    Args:
        eps (float): A small number used in alignment.
        run_checker (bool): If true, it applies more checkers to ensure the correctness.
        global_align_wo_symmetric_atom (bool):  If true, the global alignment
        before AtomPermutation will not consider atoms with permutation.
    Returns:
        None
    """

    def __init__(
        self,
        eps: float = 1e-8,
        run_checker: bool = False,
        global_align_wo_symmetric_atom: bool = False,
    ):
        """Class for assigning the optimal permutations of true coordinates/pred coordinates and coordinate masks.

        Args:
            eps (float): A small number used in alignment.
            run_checker (bool): If true, it applies more checkers to ensure the correctness.
            global_align_wo_symmetric_atom (bool):  If true, the global alignment
            before AtomPermutation will not consider atoms with permutation.
        """

        self.eps = eps
        self.run_checker = run_checker
        self.global_align_wo_symmetric_atom = global_align_wo_symmetric_atom

    @staticmethod
    def check_input_shape(
        pred_coord: ms.Tensor,
        true_coord: ms.Tensor,
        true_coord_mask: ms.Tensor,
        ref_space_uid: ms.Tensor,
        atom_perm_list: list[list],
    ):
        """
        Check if the input shapes are valid.

        Args:
            pred_coord (ms.Tensor): Predicted coordinates of atoms.
            true_coord (ms.Tensor): True coordinates of atoms.
            true_coord_mask (ms.Tensor): True coordinate masks.
            ref_space_uid (ms.Tensor): Reference space UIDs.
            atom_perm_list (list[list]): List of atom permutations.
        """

        num_atom = len(true_coord)
        if true_coord.dim() != 2:
            raise ValueError("true_coord.dim() must be 2")
        if true_coord_mask.dim() != 1:
            raise ValueError("true_coord_mask.dim() must be 1")
        if ref_space_uid.dim() != 1:
            raise ValueError("ref_space_uid.dim() must be 1")

        if true_coord.shape[-1] != 3:
            raise ValueError("true_coord.shape[-1] must be 3")
        if true_coord.shape[-2] != num_atom:
            raise ValueError("true_coord.shape[-2] must be num_atom")
        if ref_space_uid.shape[-1] != num_atom:
            raise ValueError("ref_space_uid.shape[-1] must be num_atom")
        if len(atom_perm_list) != num_atom:
            raise ValueError("len(atom_perm_list) must be num_atom")

        if pred_coord.dim() not in [2, 3]:  # for simplicity
            raise ValueError("pred_coord.dim() must be in [2, 3]")
        if pred_coord.shape[-2:] != (num_atom, 3):
            raise ValueError("pred_coord.shape[-2:] must be (num_atom, 3)")

    @staticmethod
    def global_align_pred_to_true(
        pred_coord: ms.Tensor,
        true_coord: ms.Tensor,
        true_coord_mask: ms.Tensor,
        eps: float = 1e-8,
    ) -> tuple[ms.Tensor]:
        """Align the predicted coordinates to true coordinates

        Args:
            pred_coord (ms.Tensor):
                [batch_size, num_atom, 3] or [num_atom, 3]
            true_coord (ms.Tensor):
                [num_atom, 3]
            true_coord_mask (ms.Tensor):
                [num_atom, 3]

        Returns:
            aligned_rmsd (ms.Tensor):
                [batch_size] or []
            transformed_pred_coord (ms.Tensor): having the same shape as pred_coord.
                [batch_size, num_atom, 3] or [num_atom, 3]
        """

        if true_coord.dim() < pred_coord.dim():
            if pred_coord.dim() != 3:  # [batch_size, num_atom, 3]
                raise ValueError("pred_coord.dim() must be 3")
            batch_size = pred_coord.shape[0]

            def expand_func(x):
                return expand_at_dim(x, dim=0, n=batch_size)
        else:
            def expand_func(x):
                return x

        # with torch.amp.autocast("cuda", enabled=False):
        aligned_rmsd, transformed_pred_coord, _, _ = self_aligned_rmsd(
            pred_pose=pred_coord.astype(ms.float32),
            true_pose=expand_func(true_coord.astype(ms.float32)),
            atom_mask=expand_func(true_coord_mask),
            allowing_reflection=False,
            reduce=False,
            eps=eps,
        )  # [batch_size], [batch_size, num_atom, 3]

        return aligned_rmsd, transformed_pred_coord

    @staticmethod
    def get_identity_permutation(batch_shape, num_atom):
        """Return identity permutation indices if no multiple-permutation exists for every residue

        Returns:
            ms.Tensor: identity permutation of indices
                [num_atom] or [batch_size, num_atom]
        """
        identity = mint.arange(num_atom)
        if len(batch_shape) == 0:
            return identity

        if len(batch_shape) != 1:
            raise ValueError("len(batch_shape) must be 1")
        return mint.stack([identity for _ in range(batch_shape[0])], dim=0)

    @staticmethod
    def _find_best_permutation_indices(
        per_residue_num_perm,
        per_residue_num_atom,
        per_residue_perm_list,
        per_res_rmsd,
        batch_shape,
        run_checker,
    ):
        """Find the best permutation indices."""
        best_permutation_list = []
        is_permuted_list = []
        original_rmsd_list = []
        optimized_rmsd_list = []
        i = 0

        # Enumerate over all residues (could be improved by scatter)
        for num_perm, num_residue_atom, perm in zip(
            per_residue_num_perm, per_residue_num_atom, per_residue_perm_list
        ):
            # [batch_shape, num_perm]
            cur_res_rmsd = per_res_rmsd[..., i: i + num_perm]
            best_rmsd, best_j = mint.min(cur_res_rmsd, dim=-1)  # [batch_shape]
            best_perm = perm[best_j]  # [batch_shape, num_residue_atom]
            best_permutation_list.append(best_perm)

            is_permuted_list.append(
                best_j > 0
            )  # The first of the perm lists is the identity

            optimized_rmsd_list.append(best_rmsd)
            original_rmsd_list.append(cur_res_rmsd[..., 0])

            i += num_perm

            if run_checker:
                if perm.shape != (num_perm, num_residue_atom):
                    raise ValueError("perm.shape must be (num_perm, num_residue_atom)")
                if cur_res_rmsd.shape != batch_shape + (num_perm,):
                    raise ValueError("cur_res_rmsd.shape must be batch_shape + (num_perm,)")
                if best_rmsd.shape != batch_shape:
                    raise ValueError("best_rmsd.shape must be batch_shape")
                if best_j.shape != batch_shape:
                    raise ValueError("best_j.shape must be batch_shape")
                if best_perm.shape != batch_shape + (num_residue_atom,):
                    raise ValueError("best_perm.shape must be batch_shape + (num_residue_atom,)")

        return (
            best_permutation_list,
            is_permuted_list,
            optimized_rmsd_list,
            original_rmsd_list,
        )

    @staticmethod
    def _optimize_per_residue_permutation_by_rmsd(
        per_residue_pred_coord_list: list[ms.Tensor],
        per_residue_coord_list: list[ms.Tensor],
        per_residue_coord_mask_list: list[ms.Tensor],
        per_residue_perm_list: list[ms.Tensor],
        eps: float = 1e-8,
        run_checker: bool = False,
    ) -> tuple[list[ms.Tensor]]:
        """Find the optimal permutations of true coordinates and coordinate masks to minimize the
        RMSD between true coordinates and predicted coordinates.

        Args:
            per_residue_pred_coord_list (ms.Tensor): List of residues. Each element records
                the predicted atom coordinates of one residue. Each element has shape
                [num_residue_atom, 3] or [batch_size, num_residue_atom, 3]
            per_residue_coord_list (list[ms.Tensor]): List of residues. Each element records
                the atom coordinates of one residue. Each element has shape [num_residue_atom, 3].
            per_residue_coord_mask_list (list[ms.Tensor]): List of residues. Each element records
                the atom coordinate masks of one residue. Each element has shape [num_residue_atom].
            per_residue_perm_list (list[ms.Tensor]): List of residues. Each element records
                the atom permutations of one residue. Each element has shape [num_perm, num_residue_atom].
            eps (float, optional): A small number, used in alignment. Defaults to 1e-8.
            run_checker (bool, optional): If True, run extensive checks.

        Returns:
            best_permutation_list (list[ms.Tensor]): List of residues. Each element records the
                optimal permutation that should apply to true coordinates for one residue. Each element
                has shape
                [num_residue_atom] or [batch_size, num_residue_atom]
            is_permuted_list (list[ms.Tensor]): List of residues. Each element records whether the
                atoms in this residue is permuted. Each element has shape
                [] or [batch_size]
            optimized_rmsd_list (list[ms.Tensor]): List of residues. Each element records the optimized
                rmsd of the residue. Each element has shape
                [] or [batch_size]
            original_rmsd_list (list[ms.Tensor]): List of residues. Each element records the original
                rmsd of the residue. Each element has shape
                [] or [batch_size]
        """

        # Find max number of per-residue atoms
        per_residue_num_perm = [perm.shape[0]
                                for perm in per_residue_perm_list]
        per_residue_num_atom = [perm.shape[1]
                                for perm in per_residue_perm_list]
        num_max_atom = max(per_residue_num_atom)

        # Permute true coordinates & masks according to the permutations in per_residue_perm_list
        permuted_coord, permuted_coord_mask = collect_permuted_coords(
            coord_list=per_residue_coord_list,
            coord_mask_list=per_residue_coord_mask_list,
            perm_list=per_residue_perm_list,
        )  # [num_total_perm, num_max_atom, 3], [num_total_perm, num_max_atom]
        if permuted_coord.shape[-2] != permuted_coord_mask.shape[-1] != num_max_atom:
            raise ValueError("permuted_coord.shape[-2] must be equal to permuted_coord_mask.shape[-1]",
                             "must be equal to num_max_atom")
        num_total_perm = permuted_coord.shape[0]

        # Pad 'pred_coord' to the same shape as 'permuted_coord'
        per_residue_pred_coord_list = [
            pad_at_dim(
                p_coord, dim=-2, pad_length=(0, num_max_atom - p_coord.shape[-2])
            )
            for p_coord in per_residue_pred_coord_list
        ]
        # Repeat num_perm times for each residue
        pred_coord = mint.stack(
            sum(
                [
                    [p_coord] * num_perm
                    for num_perm, p_coord in zip(
                        per_residue_num_perm, per_residue_pred_coord_list
                    )
                ],
                [],
            ),
            dim=-3,
        )  # [num_total_perm, num_max_atom, 3] or [batch_size, num_total_perm, num_max_atom, 3]
        if pred_coord.shape[-3:] != (num_total_perm, num_max_atom, 3):
            raise ValueError("pred_coord.shape[-3:] must be (num_total_perm, num_max_atom, 3)")

        batch_shape = pred_coord.shape[:-3]
        if len(batch_shape) not in [0, 1]:
            raise ValueError("len(batch_shape) must be in [0, 1]")
        if len(batch_shape) == 1:
            # expand true coord & mask to have the same batch size as pred coord
            batch_size = pred_coord.shape[0]
            permuted_coord = expand_at_dim(permuted_coord, dim=0, n=batch_size)
            permuted_coord_mask = expand_at_dim(
                permuted_coord_mask, dim=0, n=batch_size)

        # Compute per-residue rmsd
        # with torch.amp.autocast("cuda", enabled=False):
        per_res_rmsd = rmsd(
            pred_pose=pred_coord.astype(ms.float32),
            true_pose=permuted_coord.astype(ms.float32),
            mask=permuted_coord_mask,
            eps=eps,
            reduce=False,
        )  # [num_total_perm] or [batch_size, num_total_perm]
        if per_res_rmsd.shape != batch_shape + (num_total_perm,):
            raise ValueError("per_res_rmsd.shape must be batch_shape + (num_total_perm,)")

        # Find the best permutation
        return AtomPermutation._find_best_permutation_indices(
            per_residue_num_perm,
            per_residue_num_atom,
            per_residue_perm_list,
            per_res_rmsd,
            batch_shape,
            run_checker,
        )

    def _optimize_residues(
        self,
        grouped_indices,
        transformed_pred_coord,
        per_residue_coord_list,
        per_residue_coord_mask_list,
        per_residue_perm_list,
        per_residue_position_list,
        verbose,
    ):
        """Optimize the residues by the RMSD."""
        residue_position_list = []
        residue_best_permutation_list = []
        residue_is_permuted_list = []
        residue_optimized_rmsd_list = []
        residue_original_rmsd_list = []
        for atom_cutoff, residue_group in grouped_indices.items():

            if verbose:
                print(f"{len(residue_group)} residues have <={atom_cutoff} atoms.")

            # Enumerte permutations within each residue to minimize per-residue RMSD
            per_res_pos_list = [per_residue_position_list[i]
                                for i in residue_group]
            (
                per_res_best_permutation,
                per_res_is_permuted,
                per_res_optimized_rmsd,
                per_res_ori_rmsd,
            ) = self._optimize_per_residue_permutation_by_rmsd(
                per_residue_pred_coord_list=[
                    transformed_pred_coord[..., pos[0]: pos[1], :]
                    for pos in per_res_pos_list
                ],
                per_residue_coord_list=[
                    per_residue_coord_list[i] for i in residue_group
                ],
                per_residue_coord_mask_list=[
                    per_residue_coord_mask_list[i] for i in residue_group
                ],
                per_residue_perm_list=[per_residue_perm_list[i]
                                       for i in residue_group],
                eps=self.eps,
                run_checker=self.run_checker,
            )
            residue_position_list.extend(per_res_pos_list)
            residue_best_permutation_list.extend(per_res_best_permutation)
            residue_is_permuted_list.extend(per_res_is_permuted)
            residue_optimized_rmsd_list.extend(per_res_optimized_rmsd)
            residue_original_rmsd_list.extend(per_res_ori_rmsd)
        return (
            residue_position_list,
            residue_best_permutation_list,
            residue_is_permuted_list,
            residue_optimized_rmsd_list,
            residue_original_rmsd_list,
        )

    def __call__(
        self,
        pred_coord: ms.Tensor,
        true_coord: ms.Tensor,
        true_coord_mask: ms.Tensor,
        ref_space_uid: ms.Tensor,
        atom_perm_list: list[list],
        alignment_mask: ms.Tensor,
        verbose: bool = False,
        run_checker: bool = False,
    ):
        """

        Args:
            pred_coord (ms.Tensor): Predicted coordinates of atoms.
                [num_atom, 3] or [batch_size, atom, 3]
            true_coord (ms.Tensor): true coordinates of atoms.
                [num_atom, 3]
            true_coord_mask (ms.Tensor): The mask indicating whether the atom is resolved.
                [num_atom]
            ref_space_uid (ms.Tensor): Each (chain id, residue index) tuple has a unique ID.
                [num_atom]
            atom_perm_list (list[list]): The atom permutation list, where each sublist contains
                                          the permutation information of the corresponding residue.
                len(atom_perm_list) = num_atom.
                len(atom_perm_list[i]) = N_perm for the residue of atom i.
            permute_label (bool, optional): If true, return indices permutations of the true coordinate.
                Otherwise, return indices permutations for the predicted coordinate. Defaults to True.
            alignment_mask (ms.Tensor, optional): Defaults to None. A mask indicating which atoms to
                consider while performing the alignment.
            verbose (bool, optional): Defaults to False.
            run_checker (bool, optional): Whether running more checks for debug. Defaults to False.

        Returns:
            permutation (ms.Tensor): the optimized permutation of atoms.
                [num_atom] or [batch_size, num_atom]
            log_dict (Dict): a dictionary recording the permutation stats.
        """

        # Basic Info & Shape checker
        batch_shape = pred_coord.shape[:-2]
        num_atom = pred_coord.shape[-2]
        self.check_input_shape(
            pred_coord, true_coord, true_coord_mask, ref_space_uid, atom_perm_list
        )

        # Initialize log dict
        log_dict = {}

        # Initialize the permutation as identity
        permutation = self.get_identity_permutation(
            batch_shape, num_atom=num_atom)

        # Collect residues that require permutations
        (
            per_residue_position_list,
            per_residue_coord_list,
            per_residue_coord_mask_list,
            per_residue_perm_list,
            num_residue,
        ) = collect_residues_with_symmetric_atoms(
            coord=true_coord, coord_mask=true_coord_mask,
            ref_space_uid=ref_space_uid, atom_perm_list=atom_perm_list,
        )
        log_dict["num_residue"] = num_residue
        log_dict["N_res_with_symmetry"] = len(per_residue_coord_list)
        log_dict["N_res_permuted"] = 0.0
        log_dict["has_res_permuted"] = 0

        # If no residues contain symmetry, return now.
        if not per_residue_perm_list:
            print("No atom permutation is needed. Return the identity permutation.")
            return (permutation, log_dict)

        # no_permute_atom_mask: 1 represent this atom can not be permuted
        no_permute_atom_mask = mint.ones_like(true_coord_mask)
        for (start, end), per_residue_perm in zip(
            per_residue_position_list, per_residue_perm_list
        ):
            no_permute_atom_mask[start:end] = 1 - (
                (per_residue_perm != per_residue_perm[0]).sum(dim=0) > 0
            ).astype(ms.int32)

        # Perform a global alignment of predictions to true coordinates
        if alignment_mask is None:
            alignment_mask = true_coord_mask
        else:
            alignment_mask = true_coord_mask * alignment_mask.bool()
        if self.global_align_wo_symmetric_atom:
            alignment_mask = no_permute_atom_mask * alignment_mask

        if alignment_mask.sum().item() < 3:
            print("No atom permutation is needed. Return the identity permutation.")
            return (permutation, log_dict)

        # This is for atom permutation, use mask with different strategies
        _, transformed_pred_coord = self.global_align_pred_to_true(
            pred_coord, true_coord, alignment_mask, eps=self.eps,
        )
        # This is for unpermuted all-atom baseline calculation
        aligned_rmsd, _ = self.global_align_pred_to_true(
            pred_coord, true_coord, true_coord_mask, eps=self.eps,
        )
        log_dict["unpermuted_rmsd"] = aligned_rmsd.mean().item()  # [batch_size]

        # To efficiently optimize the residues parallelly, group the residues
        # according to the number of atoms in each residue.
        per_residue_num_atom = [coord.shape[0]
                                for coord in per_residue_coord_list]
        res_atom_cutoff = [15, 30, 50, 100, 100000]
        grouped_indices = {}
        for i, n in enumerate(per_residue_num_atom):
            for atom_cutoff in res_atom_cutoff:
                if n <= atom_cutoff:
                    break
            grouped_indices.setdefault(atom_cutoff, []).append(i)

        if len(sum(list(grouped_indices.values()), [])) != len(
            per_residue_perm_list
        ):
            raise ValueError("len(sum(list(grouped_indices.values()), [])) must be equal to len(per_residue_perm_list)")

        (
            residue_position_list, residue_best_permutation_list,
            residue_is_permuted_list, _, _,
        ) = self._optimize_residues(
            grouped_indices, transformed_pred_coord,
            per_residue_coord_list, per_residue_coord_mask_list,
            per_residue_perm_list, per_residue_position_list, verbose,
        )

        # Aggregate per_residue results
        # 1. Best permutation
        indices_list = [mint.arange(pos[0], pos[1])
                        for pos in residue_position_list]
        residue_atom_indices = mint.cat(indices_list, dim=-1)  # [N_perm_atom]
        residue_best_permutation = mint.cat(
            [
                ind[perm]
                for ind, perm in zip(indices_list, residue_best_permutation_list)
            ],
            dim=-1,
        )  # [batch_size, N_perm_atom] or [N_perm_atom]
        permutation[..., residue_atom_indices] = residue_best_permutation

        # 2. Other statistics
        is_res_permuted = mint.stack(residue_is_permuted_list, dim=-1).float()
        log_dict["N_res_permuted"] = is_res_permuted.sum(dim=-1).mean().item()
        log_dict["has_res_permuted"] = (
            (is_res_permuted.sum(dim=-1) > 0).float().mean().item()
        )

        return permutation, log_dict


def correct_symmetric_atoms(
    pred_coord: ms.Tensor,
    true_coord: ms.Tensor,
    true_coord_mask: ms.Tensor,
    ref_space_uid: ms.Tensor,
    atom_perm_list: list[list],
    permute_label: bool = True,
    alignment_mask: ms.Tensor = None,
    verbose: bool = False,
    run_checker: bool = False,
    eps: float = 1e-8,
    global_align_wo_symmetric_atom: bool = False,
):
    """
    Return optimally permuted true coordinates and masks according to the predicted coordinates
    Or, return optimalled permuted predicted coordinates if permute_label is False.

    Args:
        pred_coord (ms.Tensor): predicted atom positions
            [batch_size, num_atom, 3] or [num_atom, 3]
        true_coord (ms.Tensor): true atom positions
        true_coord_mask (ms.Tensor): a mask indicating whether the atom is resolved.
        ref_space_uid (ms.Tensor): unique residue ID for each atom.
            [num_atom]
        atom_perm_list (list[list]): The atom permutation list, where each sublist contains
                                     the permutation information of the corresponding residue.
            len(atom_perm_list) = num_atom.
            len(atom_perm_list[i]) = num_perm for the residue of atom i.
        permute_label (bool): indicates whether permuted true coordinates are returned or
            predicted coordinates are returned.
        alignment_mask (ms.Tensor, optional): a mask indicating which atoms are considered while
            performing the alignment.
            [num_atom]
        eps (float, optional): A small number used in alignment. Defaults to 1e-8.
        global_align_wo_symmetric_atom (bool):  If true, the global alignment 
        before AtomPermutation will not consider atoms has permutation.

    Returns:
        If permute_label is True, it returns
            coordinate (ms.Tensor): permuted true coordinates.
                [batch_size, num_atom, 3] or [num_atom, 3]
            coordinate_mask (ms.Tensor): permuted true coordinate masks.
                [batch_size, num_atom] or [num_atom]
        If permuted_label is False, it returns the permuted prediction.
            [batch_size, num_atom, 3] or [num_atom, 3]

        log_dict: logging info for the permutation
            percent_res_permuted (ms.Tensor): percentage of residues
            (excluding those with less than 3 atoms or identity perm only)that have been permuted
            best_aligned_rmsd_improved: rmsd improved after permutation, using self_aligned_rmsd
    """

    if pred_coord.dim() not in [2, 3]:
        raise ValueError("pred_coord.dim() must be in [2, 3]")
    if pred_coord.shape[-1] != 3:
        raise ValueError("pred_coord.shape[-1] must be 3")

    if alignment_mask is not None:
        alignment_mask = (true_coord_mask * alignment_mask).bool()
    else:
        alignment_mask = true_coord_mask.bool()

    with _no_grad():
        # Do not compute gradient while optimizing the permutation
        atom_perm = AtomPermutation(
            run_checker=run_checker,
            eps=eps,
            global_align_wo_symmetric_atom=global_align_wo_symmetric_atom,
        )
        indices_permutation, log_dict = atom_perm(
            pred_coord,
            true_coord,
            true_coord_mask,
            ref_space_uid,
            atom_perm_list,
            alignment_mask=alignment_mask,
            verbose=verbose,
        )

    # Log aligned rmsd after permutation
    if "unpermuted_rmsd" in log_dict:
        # This is the final permuted all-atom rmsd
        permuted_rmsd, _ = AtomPermutation.global_align_pred_to_true(
            pred_coord,
            true_coord[indices_permutation],
            true_coord_mask[indices_permutation],
            eps=eps,
        )
        log_dict["permuted_rmsd"] = permuted_rmsd.mean().item()
        log_dict["improved_rmsd"] = (
            log_dict["unpermuted_rmsd"] - log_dict["permuted_rmsd"]
        )

    if permute_label:
        return (
            true_coord[indices_permutation],
            true_coord_mask[indices_permutation],
            log_dict,
            indices_permutation,
        )
    # Find the permutation of the prediction
    if pred_coord.dim() == 2:
        # Inverse permutation for 1D case
        indices_permutation = mint.argsort(indices_permutation)
        pred_coord_permuted = pred_coord[indices_permutation]
    else:
        # Inverse permutation for 2D case (batch mode)
        indices_permutation = mint.argsort(indices_permutation, dim=1)
        indices_permutation_expanded = expand_at_dim(
            indices_permutation, dim=-1, n=3
        )  # [batch_size, num_atom, 3]
        pred_coord_permuted = pred_coord.gather(
            1, indices_permutation_expanded)

    return pred_coord_permuted, None, log_dict, indices_permutation

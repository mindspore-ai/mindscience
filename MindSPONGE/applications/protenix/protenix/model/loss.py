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

"""loss"""

from typing import Any, Optional

import mindspore as ms
from mindspore import nn, ops, Tensor, mint, _no_grad

from protenix.metrics.rmsd import weighted_rigid_align
from protenix.model.modules.frames import gather_frame_atom_by_indices, express_coordinates_in_frame

EPSILON = 1e-8


def expand_at_dim(x: Tensor, dim: int, n: int) -> Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        Tensor: expanded tensor of shape [..., n, ...]
    """
    x = ops.expand_dims(x, dim)
    if dim < 0:
        dim = x.ndim + dim

    # Create the target shape for expand
    target_shape = list(x.shape)
    target_shape[dim] = n

    # Use tile to expand the tensor
    tile_shape = [1] * x.ndim
    tile_shape[dim] = n

    return ops.tile(x, tuple(tile_shape))


def loss_reduction(loss: Tensor, method: str = "mean") -> Tensor:
    """reduction wrapper

    Args:
        loss (Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        Tensor: reduced loss
            [] or [...]
    """

    if method is None:
        return loss
    if method not in ["mean", "sum", "add", "max", "min"]:
        raise ValueError(f"Invalid method {method}")
    if method == "add":
        method = "sum"
    if method == "mean":
        return ops.mean(loss)
    if method == "sum":
        return ops.sum(loss)
    if method == "max":
        return ops.max(loss)
    if method == "min":
        return ops.min(loss)
    return loss


class Cdist(nn.Cell):
    def __init__(self):
        super().__init__(self)

    def construct(self, coord1, coord2):
        dist = mint.sum(mint.square(mint.unsqueeze(coord1, -2).astype(ms.bfloat16)
                                    - mint.unsqueeze(coord2, -3).astype(ms.bfloat16)) + EPSILON,
                        dim=-1, keepdim=False).sqrt().astype(coord1.dtype)
        return dist


class SmoothLDDTChunk(nn.Cell):
    """SmoothLDDTChunk"""

    def __init__(self, eps):
        super().__init__(self)
        self.eps = eps

    def construct(self, pred_distance, true_distance, c_lm=None):
        """SmoothLDDTChunk"""
        dist_diff = ops.abs(pred_distance - true_distance)
        # For save cuda memory we use inplace op
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon = dist_diff_epsilon + \
                0.25 * mint.sigmoid(threshold - dist_diff)

        # Compute mean
        if c_lm is not None:
            lddt = ops.sum(c_lm.astype(ms.int32) * dist_diff_epsilon, (-1, -2)) / (
                ops.sum(c_lm.astype(ms.int32), (-1, -2)) + self.eps
            )  # [..., n_sample]
        else:
            # It's for sparse forward mode
            lddt = ops.mean(dist_diff_epsilon, -1)
        return lddt


class SmoothLDDTLoss(nn.Cell):
    """
    Implements Algorithm 27 [SmoothLDDTLoss] in AF3
    """

    def __init__(
        self,
        eps: float = 1e-10,
        reduction: str = "mean",
    ) -> None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.chunk = SmoothLDDTChunk(self.eps)
        self.chunk.recompute()
        self.cdist = Cdist()
        self.cdist.recompute()

    def construct(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        lddt_mask: Tensor,
        diffusion_chunk_size: Optional[int] = None,
    ) -> Tensor:
        """SmoothLDDTLoss dense implementation

        Args:
            pred_coordinate (Tensor): the diffusion denoised atom coordinates
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): the ground truth atom coordinates
                [..., n_atom, 3]
            lddt_mask (Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [n_atom, n_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the n_sample dimension. Defaults to None.

        Returns:
            Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = ops.expand_dims(
            lddt_mask.bool(), -3).astype(ms.int32)  # [..., 1, n_atom, n_atom]
        true_distance = self.cdist(true_coordinate, true_coordinate)
        diffusion_chunk_size = 8

        if diffusion_chunk_size is None:
            pred_distance = self.cdist(pred_coordinate, pred_coordinate)
            lddt = self.chunk(
                pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm
            )
        else:
            lddt = []
            n_sample = pred_coordinate.shape[-3]
            no_chunks = n_sample // diffusion_chunk_size + (
                n_sample % diffusion_chunk_size != 0
            )
            for i in range(no_chunks):
                pred_distance_i = self.cdist(
                    pred_coordinate[
                        i * diffusion_chunk_size: (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                    pred_coordinate[
                        i * diffusion_chunk_size: (i + 1) * diffusion_chunk_size,
                        :,
                        :,
                    ],
                )
                lddt_i = self.chunk(
                    pred_distance_i,
                    true_distance,
                    c_lm,)

                lddt.append(lddt_i)
            lddt = mint.concat(lddt, dim=-1)

        lddt = lddt.mean(-1)  # [...]
        return 1 - loss_reduction(lddt, method=self.reduction)


class BondLoss(nn.Cell):
    """
    Implements Formula 5 [BondLoss] in AF3
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        """BondLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, bond_mask):
        # Distance squared error
        # [...,  n_sample , n_atom, n_atom]
        dist_squared_err = (
            pred_distance - ops.expand_dims(true_distance, -3)) ** 2
        bond_loss = ops.sum(dist_squared_err * bond_mask, (-1, -2)) / ops.sum(
            bond_mask + self.eps, (-1, -2)
        )  # [..., n_sample]
        return bond_loss

    def construct(
        self,
        pred_distance: Tensor,
        true_distance: Tensor,
        distance_mask: Tensor,
        bond_mask: Tensor,
        per_sample_scale: Tensor = None,
        diffusion_chunk_size: Optional[int] = None,
    ) -> Tensor:
        """BondLoss

        Args:
            pred_distance (Tensor): the diffusion denoised atom-atom distance
                [..., n_sample, n_atom, n_atom]
            true_distance (Tensor): the ground truth coordinates
                [..., n_atom, n_atom]
            distance_mask (Tensor): whether true coordinates exist.
                [n_atom, n_atom] or [..., n_atom, n_atom]
            bond_mask (Tensor): bonds considered in this loss
                [n_atom, n_atom] or [..., n_atom, n_atom]
            per_sample_scale (Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., n_sample]
            diffusion_chunk_size (Optional[int]): Chunk size over the n_sample dimension. Defaults to None.

        Returns:
            Tensor: the bond loss
                [...] if reduction is None else []
        """
        if not bond_mask:
            return ms.Tensor(0, dtype=ms.float32)
        # [1, n_atom, n_atom] or [..., 1, n_atom, n_atom]
        bond_mask = ops.expand_dims(bond_mask * distance_mask, -3)
        # Bond Loss
        if diffusion_chunk_size is None:
            bond_loss = self._chunk_forward(
                pred_distance=pred_distance,
                true_distance=true_distance,
                bond_mask=bond_mask,
            )
        else:
            raise ValueError("Wrong branch")
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale

        bond_loss = bond_loss.mean(-1)  # [...]
        return loss_reduction(bond_loss, method=self.reduction)

    def sparse_forward(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        distance_mask: Tensor,
        bond_mask: Tensor,
        per_sample_scale: Tensor = None,
    ) -> Tensor:
        """BondLoss sparse implementation

        Args:
            pred_coordinate (Tensor): the diffusion denoised atom coordinates
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): the ground truth atom coordinates
                [..., n_atom, 3]
            distance_mask (Tensor): whether true coordinates exist.
                [n_atom, n_atom] or [..., n_atom, n_atom]
            bond_mask (Tensor): bonds considered in this loss
                [n_atom, n_atom] or [..., n_atom, n_atom]
            per_sample_scale (Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., n_sample]
        Returns:
            Tensor: the bond loss
                [...] if reduction is None else []
        """
        if bond_mask is None:
            return ms.Tensor(0, dtype=ms.float32)
        bond_mask = bond_mask * distance_mask
        bond_indices = ops.nonzero(bond_mask)
        pred_coords_i = pred_coordinate.gather(bond_indices[:, 0], axis=-2)
        pred_coords_j = pred_coordinate.gather(bond_indices[:, 1], axis=-2)
        true_coords_i = true_coordinate.gather(bond_indices[:, 0], axis=-2)
        true_coords_j = true_coordinate.gather(bond_indices[:, 1], axis=-2)

        pred_distance_sparse = ops.norm(
            pred_coords_i - pred_coords_j, ord=2, dim=-1)
        true_distance_sparse = ops.norm(
            true_coords_i - true_coords_j, ord=2, dim=-1)
        dist_squared_err_sparse = (
            pred_distance_sparse - true_distance_sparse)  # ** 2
        dist_squared_err_sparse = ms.ops.pow(dist_squared_err_sparse, 2)
        # Protecting special data that has size: tensor([], size=(x, 0), grad_fn=<PowBackward0>)
        if dist_squared_err_sparse.numel() == 0:
            return Tensor(0.0, dtype=dist_squared_err_sparse.dtype)
        bond_loss = ops.mean(dist_squared_err_sparse,
                             axis=-1)  # [..., n_sample]
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale
        bond_loss = bond_loss.mean(-1)  # [...]
        return bond_loss


def compute_lddt_mask(
    true_distance: Tensor,
    distance_mask: Tensor,
    is_nucleotide: Tensor,
    is_nucleotide_threshold: float = 30.0,
    is_not_nucleotide_threshold: float = 15.0,
) -> Tensor:
    """calculate the atom pair mask with the bespoke radius

    Args:
        true_distance (Tensor): the ground truth coordinates
            [..., n_atom, n_atom]
        distance_mask (Tensor): whether true coordinates exist.
            [..., n_atom, n_atom] or [n_atom, n_atom]
        is_nucleotide (Tensor): Indicator for nucleotide atoms.
            [..., n_atom] or [n_atom]
        is_nucleotide_threshold (float): Threshold distance for nucleotide atoms. Defaults to 30.0.
        is_not_nucleotide_threshold (float): Threshold distance for non-nucleotide atoms. Defaults to 15.0.

    Returns:
        c_lm (Tensor): the atom pair mask c_lm, not symmetric
            [..., n_atom, n_atom]
    """
    # Restrict to bespoke inclusion radius
    is_nucleotide_mask = is_nucleotide.bool()
    c_lm = ((true_distance < is_nucleotide_threshold) * is_nucleotide_mask[..., None]).astype(ms.int32) + ((
        true_distance < is_not_nucleotide_threshold
    ) * (
        ~is_nucleotide_mask[..., None]
    )).astype(ms.int32)  # [..., n_atom, n_atom]

    # Zero-out diagonals of c_lm and cast to float
    c_lm = c_lm * (
        1 - ops.eye(n=c_lm.shape[-1], dtype=true_distance.dtype)
    )
    # Zero-out atom pairs without true coordinates
    # Note: the sparsity of c_lm is ~10% in 5000 atom-pairs,
    # and becomes more sparse as the number of atoms increases,
    # change to sparse implementation can reduce cuda memory
    c_lm = c_lm * distance_mask  # [..., n_atom, n_atom]
    return c_lm


def softmax_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Softmax cross entropy

    Args:
        logits (Tensor): classification logits
            [..., num_class]
        labels (Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        Tensor: softmax cross entropy
            [...]
    """
    log_softmax = ops.log_softmax(logits, -1)
    loss = -1 * ops.sum(labels * log_softmax, -1)
    return loss


class DistogramLoss(nn.Cell):
    """
    Implements DistogramLoss in AF3
    """

    def __init__(
        self,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """Distogram loss
        This head and loss are identical to AlphaFold 2, 
        where the pairwise token distances use the representative atom for each token,
        which are:
            Cβ for protein residues (Cα for glycine),
            C4 for purines and C2 for pyrimidines.
            All ligands already have a single atom per token.

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 2.3125.
            max_bin (float, optional): max boundary of bins. Defaults to 21.6875.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction
        self.cdist = Cdist()
        self.cdist.recompute()

    def calculate_label(
        self,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        rep_atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """calculate the label as bins

        Args:
            true_coordinate (Tensor): true coordinates.
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist.
                [n_atom] or [..., n_atom]
            rep_atom_mask (Tensor): representative atom mask
                [n_atom]

        Returns:
            true_bins (Tensor): distance error assigned into bins (one-hot).
                [..., N_token, N_token, no_bins]
            pair_coordinate_mask (Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """

        boundaries = ops.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins - 1,
        )

        # Compute label: the true bins
        # True distance
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[...,
                                          rep_atom_mask, :]  # [..., N_token, 3]
        gt_dist = self.cdist(true_coordinate, true_coordinate)
        # Assign distance to bins
        true_bins = ops.sum(
            ops.expand_dims(gt_dist, -1) > boundaries, -1
        )  # range in [0, no_bins-1], shape = [..., N_token, N_token]

        # Mask
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return ops.one_hot(true_bins, self.no_bins), pair_mask

    def construct(
        self,
        logits: Tensor,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        rep_atom_mask: Tensor,
    ) -> Tensor:
        """Distogram loss

        Args:
            logits (Tensor): logits.
                [..., N_token, N_token, no_bins]
            true_coordinate (Tensor): true coordinates.
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist.
                [n_atom] or [..., n_atom]
            rep_atom_mask (Tensor): representative atom mask.
                [n_atom]

        Returns:
            Tensor: the return loss.
                [...] if self.reduction is not None else []
        """
        with _no_grad():
            true_bins, pair_mask = self.calculate_label(
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )
        errors = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., N_token, N_token]

        denom = self.eps + ops.sum(pair_mask, (-1, -2))
        loss = ops.sum(errors * pair_mask, (-1, -2))
        loss = loss / denom
        return loss_reduction(loss, method=self.reduction)


class PDELoss(nn.Cell):
    """
    Implements Predicted distance loss in AF3
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 32,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """PDELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction
        self.cdist = Cdist()
        self.cdist.recompute()

    def calculate_label(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        rep_atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """calculate the label as bins

        Args:
            pred_coordinate (Tensor): predicted coordinates.
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): true coordinates.
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist.
                [n_atom] or [..., n_atom]
            rep_atom_mask (Tensor):
                [n_atom]

        Returns:
            true_bins (Tensor): distance error assigned into bins (one-hot).
                [..., n_sample, N_token, N_token, no_bins]
            pair_coordinate_mask (Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """

        boundaries = ops.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins + 1,
        )

        # Compute label: the true bins
        # True distance
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[...,
                                          rep_atom_mask, :]  # [..., N_token, 3]
        gt_dist = self.cdist(true_coordinate, true_coordinate)
        # Predicted distance
        pred_coordinate = pred_coordinate[..., rep_atom_mask, :]
        pred_dist = self.cdist(pred_coordinate, pred_coordinate)
        # Distance error
        dist_error = ops.abs(pred_dist - ops.expand_dims(gt_dist, -3))

        # Assign distance error to bins
        true_bins = ops.sum(
            ops.expand_dims(dist_error, -1) > boundaries, -1
        )  # range in [1, no_bins + 1], shape = [..., n_sample, N_token, N_token]
        true_bins = ops.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0 occurs

        # Mask
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        return ops.one_hot(true_bins - 1, self.no_bins), pair_mask

    def construct(
        self,
        logits: Tensor,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        rep_atom_mask: Tensor,
    ) -> Tensor:
        """PDELoss

        Args:
            logits (Tensor): logits
                [..., n_sample, N_token, N_token, no_bins]
            pred_coordinate: (Tensor): predict coordinates
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): true coordinates
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist
                [n_atom] or [..., n_atom]
            rep_atom_mask (Tensor): representative atom mask for this loss
                [n_atom]

        Returns:
            Tensor: the return loss
                [...] if reduction is None else []
        """
        with _no_grad():
            true_bins, pair_mask = self.calculate_label(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                rep_atom_mask=rep_atom_mask,
            )

        errors = softmax_cross_entropy(
            logits=logits[:true_bins.shape[0],
                          :true_bins.shape[1], :true_bins.shape[2]],
            labels=true_bins,
        )  # [..., n_sample, N_token, N_token]

        denom = self.eps + ops.sum(pair_mask, (-1, -2))  # [...]
        # [..., n_sample, N_token, N_token]
        loss = errors * ops.expand_dims(pair_mask, -3)
        loss = ops.sum(loss, (-1, -2))  # [..., n_sample]
        loss = loss / ops.expand_dims(denom, -1)  # [..., n_sample]
        loss = loss.mean(-1)  # [...]

        return loss_reduction(loss, method=self.reduction)


# Algorithm 30 Compute alignment error
def compute_alignment_error_squared(
    pred_coordinate: Tensor,
    true_coordinate: Tensor,
    pred_frames: Tensor,
    true_frames: Tensor,
) -> Tensor:
    """Implements Algorithm 30 Compute alignment error, but do not take the square root

    Args:
        pred_coordinate (Tensor): the predict coords [frame center]
            [..., n_sample, N_token, 3]
        true_coordinate (Tensor): the ground truth coords [frame center]
            [..., N_token, 3]
        pred_frames (Tensor): the predict frame
            [..., n_sample, N_frame, 3, 3]
        true_frames (Tensor): the ground truth frame
            [..., N_frame, 3, 3]

    Returns:
        Tensor: the computed alignment error
            [..., n_sample, N_frame, N_token]
    """
    x_transformed_pred = express_coordinates_in_frame(
        coordinate=pred_coordinate, frames=pred_frames
    )  # [..., n_sample, N_frame, N_token, 3]
    x_transformed_true = express_coordinates_in_frame(
        coordinate=true_coordinate, frames=true_frames
    )  # [..., N_frame, N_token, 3]
    squared_pae = ops.sum(
        (x_transformed_pred - ops.expand_dims(x_transformed_true, -4)) ** 2, -1
    )  # [..., n_sample, N_frame, N_token]
    return squared_pae


class PAELoss(nn.Cell):
    """
    Implements Predicted Aligned distance loss in AF3
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 32,
        no_bins: int = 64,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """PAELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        rep_atom_mask: Tensor,
        frame_atom_index: Tensor,
        has_frame: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """calculate true PAE (squared) and true bins

        Args:
            pred_coordinate: (Tensor): predict coordinates.
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): true coordinates.
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist
                [n_atom]
            rep_atom_mask (Tensor): masks of the representative atom for each token.
                [n_atom]
            frame_atom_index (Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            squared_pae (Tensor): pairwise alignment error squared
                [..., n_sample, N_frame, N_token] where N_token = rep_atom_mask.sum()
            true_bins (Tensor): the true bins
                [..., n_sample, N_frame, N_token, no_bins]
            frame_token_pair_mask (Tensor): whether frame_i token_j both have true coordinates.
                [N_frame, N_token]
        """

        coordinate_mask = coordinate_mask.bool()
        rep_atom_mask = rep_atom_mask.bool()
        has_frame = has_frame.bool()

        if len(frame_atom_index.shape) != 2:
            raise ValueError(f"frame_atom_index shape {frame_atom_index.shape} is not 2")

        # Take valid frames: N_token -> N_frame
        # [N_frame, 3[three atom]]
        frame_atom_index = frame_atom_index[has_frame, :]

        # Get predicted frames and true frames
        pred_frames = gather_frame_atom_by_indices(
            coordinate=pred_coordinate, frame_atom_index=frame_atom_index, dim=-2
        )  # [..., n_sample, N_frame, 3[three atom], 3[coordinates]]
        true_frames = gather_frame_atom_by_indices(
            coordinate=true_coordinate, frame_atom_index=frame_atom_index, dim=-2
        )  # [..., N_frame, 3[three atom], 3[coordinates]]

        # Get pair_mask for computing the loss
        true_frame_coord_mask = gather_frame_atom_by_indices(
            coordinate=coordinate_mask, frame_atom_index=frame_atom_index, dim=-1
        )  # [N_frame, 3[three atom]]
        true_frame_coord_mask = (
            true_frame_coord_mask.sum(-1) >= 3
        )  # [N_frame] whether all atoms in the frame has coordinates
        token_mask = coordinate_mask.squeeze()[rep_atom_mask]  # [N_token]
        frame_token_pair_mask = (
            true_frame_coord_mask[..., None] * token_mask[..., None, :]
        )  # [N_frame, N_token]

        squared_pae = (
            compute_alignment_error_squared(
                pred_coordinate=pred_coordinate[..., rep_atom_mask, :],
                true_coordinate=true_coordinate[..., rep_atom_mask, :],
                pred_frames=pred_frames,
                true_frames=true_frames,
            )
            * frame_token_pair_mask
        )  # [..., n_sample, N_frame, N_token]

        # Compute true bins
        boundaries = ops.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins + 1,
        )
        boundaries = boundaries**2

        true_bins = ops.sum(
            ops.expand_dims(squared_pae, -1) > boundaries, -1
        )  # range [1, no_bins + 1]
        true_bins = ops.where(
            frame_token_pair_mask,
            true_bins,
            ops.ones_like(true_bins) * self.no_bins,
        )
        true_bins = ops.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0 occurs

        return (
            squared_pae,
            ops.one_hot(true_bins - 1, self.no_bins),
            frame_token_pair_mask,
        )

    def construct(
        self,
        logits: Tensor,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        frame_atom_index: Tensor,
        rep_atom_mask: Tensor,
        has_frame: Tensor,
    ) -> Tensor:
        """PAELoss

        Args:
            logits (Tensor): logits
                [..., n_sample, N_token, N_token, no_bins]
            pred_coordinate: (Tensor): predict coordinates
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): true coordinates
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist
                [n_atom]
            rep_atom_mask (Tensor): masks of the representative atom for each token.
                [n_atom]
            frame_atom_index (Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            Tensor: the return loss
                [] if reduce
                [..., n] else
        """
        has_frame = has_frame.bool()
        rep_atom_mask = rep_atom_mask.bool()
        if len(has_frame.shape) != 1:
            raise ValueError(f"has_frame shape {has_frame.shape} is not 1")
        if len(frame_atom_index.shape) != 2:
            raise ValueError(f"frame_atom_index shape {frame_atom_index.shape} is not 2")

        # true_bins: [..., n_sample, N_frame, N_token, no_bins]
        # pair_mask: [N_frame, N_token]
        with _no_grad():
            _, true_bins, pair_mask = self.calculate_label(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                frame_atom_index=frame_atom_index,
                rep_atom_mask=rep_atom_mask,
                coordinate_mask=coordinate_mask,
                has_frame=has_frame,
            )

        loss = softmax_cross_entropy(
            logits=logits[
                ..., has_frame, :, :
            ],  # [..., n_sample, N_frame, N_token, no_bins]
            labels=true_bins,
        )  # [..., n_sample, N_frame, N_token]

        denom = self.eps + ops.sum(pair_mask, (-1, -2))  # []
        # [..., n_sample, N_token, N_token]
        loss = loss * ops.expand_dims(pair_mask, -3)
        loss = ops.sum(loss, (-1, -2))  # [..., n_sample]
        loss = loss / ops.expand_dims(denom, -1)  # [..., n_sample]
        loss = loss.mean(-1)  # [...]
        return loss_reduction(loss, self.reduction)


class ExperimentallyResolvedLoss(nn.Cell):
    """
    ExperimentallyResolvedLoss
    Args:
        eps (float, optional): avoid nan. Defaults to 1e-6.
        reduction (str, optional): reduction method for the batch dims. Defaults to mean.
    Returns:
        Tensor: the return loss
            [] if reduce
            [..., n] else
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def construct(
        self,
        logits: Tensor,
        coordinate_mask: Tensor,
        atom_mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            logits (Tensor): logits
                [..., n_sample, n_atom, no_bins:=2]
            coordinate_mask (Tensor): whether true coordinates exist
                [..., n_atom] | [n_atom]
            atom_mask (Tensor, optional): whether to conside the atom in the loss
                [..., n_atom]
        Returns:
            Tensor: the experimentally resolved loss
        """
        is_resolved = ops.one_hot(
            coordinate_mask.astype(ms.int64), 2
        ).astype(ms.int32)  # [..., n_atom, 2] or [n_atom, 2]
        errors = softmax_cross_entropy(
            logits=logits, labels=ops.expand_dims(is_resolved, -3)
        )  # [..., n_sample, n_atom]
        if atom_mask is None:
            loss = errors.mean(-1)  # [..., n_sample]
        else:
            loss = ops.sum(
                errors * atom_mask[..., None, :], -1
            )  # [..., n_sample]
            loss = loss / (
                self.eps + ops.sum(atom_mask[..., None, :], -1)
            )  # [..., n_sample]

        loss = loss.mean(-1)  # [...]

        return loss_reduction(loss, method=self.reduction)


class MSELoss(nn.Cell):
    """
    Implements Formula 2-4 [MSELoss] in AF3
    """

    def __init__(
        self,
        weight_mse: float = 1 / 3,
        weight_dna: float = 5.0,
        weight_rna=5.0,
        weight_ligand=10.0,
        eps=1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction
        self.cdist = Cdist()
        self.cdist.recompute()

    def weighted_rigid_align(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        is_dna: Tensor,
        is_rna: Tensor,
        is_ligand: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """compute weighted rigid alignment results

        Args:
            pred_coordinate (Tensor): the denoised coordinates from diffusion module
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): the ground truth coordinates
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist
                [n_atom] or [..., n_atom]
            is_dna / is_rna / is_ligand (Tensor): mol type mask
                [n_atom] or [..., n_atom]

        Returns:
            true_coordinate_aligned (Tensor): aligned coordinates for each sample
                [..., n_sample, n_atom, 3]
            weight (Tensor): weights for each atom
                [n_atom] or [..., n_sample, n_atom]
        """
        if len(pred_coordinate.shape) == 3:
            n_sample = pred_coordinate.shape[-3]
        else:
            n_sample = 1
        weight = (
            1
            + self.weight_dna * is_dna
            + self.weight_rna * is_rna
            + self.weight_ligand * is_ligand
        )  # [n_atom] or [..., n_atom]

        # Apply coordinate_mask
        weight = weight * coordinate_mask  # [n_atom] or [..., n_atom]
        true_coordinate = true_coordinate * \
            ops.expand_dims(coordinate_mask, -1)
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]

        # Reshape to add "n_sample" dimension
        true_coordinate = expand_at_dim(
            true_coordinate, dim=-3, n=n_sample
        )  # [..., n_sample, n_atom, 3]
        if len(weight.shape) > 1:
            weight = expand_at_dim(
                weight, dim=-2, n=n_sample
            )  # [..., n_sample, n_atom]

        # Align GT coords to predicted coords
        # Some ops in weighted_rigid_align do not support BFloat16 training
        true_coordinate_aligned = weighted_rigid_align(
            x=true_coordinate.astype(ms.float32),  # [..., n_sample, n_atom, 3]
            x_target=pred_coordinate.astype(
                ms.float32
            ),  # [..., n_sample, n_atom, 3]
            atom_weight=weight.astype(
                ms.float32
            ),  # [n_atom] or [..., n_sample, n_atom]
        )  # [..., n_sample, n_atom, 3]
        true_coordinate_aligned = true_coordinate_aligned.astype(
            pred_coordinate.dtype)

        return (true_coordinate_aligned, weight)

    def construct(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        coordinate_mask: Tensor,
        is_dna: Tensor,
        is_rna: Tensor,
        is_ligand: Tensor,
        per_sample_scale: Tensor = None,
    ) -> Tensor:
        """MSELoss

        Args:
            pred_coordinate (Tensor): the denoised coordinates from diffusion module.
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor): the ground truth coordinates.
                [..., n_atom, 3]
            coordinate_mask (Tensor): whether true coordinates exist.
                [n_atom] or [..., n_atom]
            is_dna / is_rna / is_ligand (Tensor): mol type mask.
                [n_atom] or [..., n_atom]
            per_sample_scale (Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., n_sample]

        Returns:
            Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        # True_coordinate_aligned: [..., n_sample, n_atom, 3]
        # Weight: [n_atom] or [..., n_sample, n_atom]
        with _no_grad():
            true_coordinate_aligned, weight = self.weighted_rigid_align(
                pred_coordinate=pred_coordinate,
                true_coordinate=true_coordinate,
                coordinate_mask=coordinate_mask,
                is_dna=is_dna,
                is_rna=is_rna,
                is_ligand=is_ligand,
            )

        # Calculate MSE loss
        per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(
            -1
        )  # [..., n_sample, n_atom]
        per_sample_weighted_mse = (weight * per_atom_se).sum(-1) / (
            ops.sum(coordinate_mask, -1) + self.eps
        )  # [..., n_sample]

        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale

        weighted_align_mse_loss = self.weight_mse * (per_sample_weighted_mse).mean(
            -1
        )  # [...]

        loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)
        return loss


class PLDDTLoss(nn.Cell):
    """
    Implements PLDDT Loss in AF3, different from the paper description.
    Main changes:
    1. use difference of distance instead of predicted distance when calculating plddt
    2. normalize each plddt score within 0-1
    """

    def __init__(
        self,
        min_bin: float = 0,
        max_bin: float = 1,
        no_bins: int = 50,
        is_nucleotide_threshold: float = 30.0,
        is_not_nucleotide_threshold: float = 15.0,
        eps: float = 1e-6,
        normalize: bool = True,
        reduction: str = "mean",
    ) -> None:
        """PLDDT loss
        This loss are between atoms l and m (has some filters) in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 1.
            no_bins (int, optional): number of bins. Defaults to 50.
            is_nucleotide_threshold (float, optional): threshold for nucleotide atoms. Defaults 30.0.
            is_not_nucleotide_threshold (float, optional): threshold for non-nucleotide atoms. Defaults 15.0
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super().__init__()
        self.normalize = normalize
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction
        self.is_nucleotide_threshold = is_nucleotide_threshold
        self.is_not_nucleotide_threshold = is_not_nucleotide_threshold

    @_no_grad()
    def bins_from_lddt(
        self,
        per_atom_lddt: Tensor,
        per_atom_weight: Tensor,
    ):
        """
        bins_from_lddt
        Args:
            per_atom_lddt (Tensor): per-atom lddt
                [..., n_sample, n_atom, 1]
            per_atom_weight (Tensor): per-atom weight
                [..., n_sample, n_atom, 1]
        Returns:
            Tensor: true bins
                [..., n_sample, n_atom, N_bins]
        """
        if self.normalize:
            per_atom_lddt = per_atom_lddt / (per_atom_weight + self.eps)
        # Distribute into bins
        boundaries = ops.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins + 1,
        )  # [N_bins]

        true_bins = ops.sum(
            per_atom_lddt > boundaries, -1
        )  # [...,  n_sample, n_atom], range in [1, no_bins]
        true_bins = ops.clamp(
            true_bins, min=1, max=self.no_bins
        )  # just in case bin=0/no_bins+1 occurs
        true_bins = ops.one_hot(
            true_bins - 1, self.no_bins
        )  # [...,  n_sample, n_atom, N_bins]

        return true_bins

    def construct(
        self,
        logits: Tensor,
        per_atom_lddt: Tensor,
        per_atom_weight: Tensor,
    ):
        """
        Args:
        per_atom_lddt
            [..., n_sample, n_atom, 1]
        per_atom_weight
            [..., n_sample, n_atom, 1]
        Returns:
            Tensor: per-atom lddt bins
                [..., n_sample, n_atom, N_bins]
        """
        true_bins = self.bins_from_lddt(per_atom_lddt, per_atom_weight)
        plddt_loss = softmax_cross_entropy(
            logits=logits,
            labels=true_bins,
        )  # [..., n_sample, n_atom_with_coords]
        # Average over atoms
        plddt_loss = plddt_loss.mean(-1)  # [..., n_sample]
        # Average over samples
        plddt_loss = plddt_loss.mean(-1)  # [...]
        return loss_reduction(plddt_loss, method=self.reduction)


class ProtenixLoss(nn.Cell):
    """Aggregation of the various losses"""

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

        self.alpha_confidence = self.configs.loss.weight.alpha_confidence
        self.alpha_pae = self.configs.loss.weight.alpha_pae
        self.alpha_except_pae = self.configs.loss.weight.alpha_except_pae
        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.alpha_distogram = self.configs.loss.weight.alpha_distogram
        self.alpha_bond = self.configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt

        self.lddt_radius = {
            "is_nucleotide_threshold": 30.0,
            "is_not_nucleotide_threshold": 15.0,
        }

        self.alpha_confidence = 1e-4
        self.alpha_except_pae = 1
        self.alpha_pae = 0
        self.alpha_diffusion = 4.0
        self.alpha_bond = 0
        self.weight_smooth_lddt = 1.0
        self.alpha_distogram = 0.03

        self.loss_weight = {
            # confidence
            "plddt_loss": self.alpha_confidence * self.alpha_except_pae,
            "pde_loss": self.alpha_confidence * self.alpha_except_pae,
            "resolved_loss": self.alpha_confidence * self.alpha_except_pae,
            "pae_loss": self.alpha_confidence * self.alpha_pae,
            "mse_loss": self.alpha_diffusion,
            "bond_loss": self.alpha_diffusion * self.alpha_bond,
            "smooth_lddt_loss": self.alpha_diffusion
            # Different from AF3 appendix eq(6), where smooth_lddt has no weight
            * self.weight_smooth_lddt,
            "distogram_loss": self.alpha_distogram,
        }

        # Loss
        self.plddt_loss = PLDDTLoss(**configs.loss.plddt, **self.lddt_radius)
        self.pde_loss = PDELoss(**configs.loss.pde)
        self.pde_loss.recompute()
        self.resolved_loss = ExperimentallyResolvedLoss(
            **configs.loss.resolved)
        self.resolved_loss.recompute()
        self.pae_loss = PAELoss(**configs.loss.pae)
        self.mse_loss = MSELoss(**configs.loss.diffusion.mse)
        self.mse_loss.recompute()
        self.bond_loss = BondLoss(**configs.loss.diffusion.bond)
        self.bond_loss.recompute()
        self.smooth_lddt_loss = SmoothLDDTLoss(
            **configs.loss.diffusion.smooth_lddt)
        self.smooth_lddt_loss.recompute()
        self.distogram_loss = DistogramLoss(**configs.loss.distogram)
        self.distogram_loss.recompute()

        self.cdist = Cdist()
        self.cdist.recompute()

    def calculate_label(
        self,
        feat_dict: dict[str, Any],
        label_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """calculate true distance, and atom pair mask

        Args:
            feat_dict (dict): Feature dictionary containing additional features.
            label_dict (dict): Label dictionary containing ground truth data.

        Returns:
            label_dict (dict): with the following updates:
                distance (Tensor): true atom-atom distance.
                    [..., n_atom, n_atom]
                distance_mask (Tensor): atom-atom mask indicating whether true distance exists.
                    [..., n_atom, n_atom]
        """
        # Distance mask
        distance_mask = (
            label_dict["coordinate_mask"][..., None]
            * label_dict["coordinate_mask"][..., None, :]
        )
        # Distances for all atom pairs
        # Note: we convert to bf16 for saving cuda memory, if performance drops, do not convert it
        distance = self.cdist(
            label_dict["coordinate"], label_dict["coordinate"])

        is_nucleotide = (feat_dict.is_rna.int() + feat_dict.is_dna.int())

        lddt_mask = compute_lddt_mask(
            true_distance=distance,
            distance_mask=distance_mask,
            is_nucleotide=is_nucleotide,
            **self.lddt_radius,
        )

        label_dict["lddt_mask"] = lddt_mask
        label_dict["distance_mask"] = distance_mask
        if not self.configs.loss_metrics_sparse_enable:
            label_dict["distance"] = distance
        del distance, distance_mask, lddt_mask
        return label_dict

    def calculate_prediction(
        self,
        pred_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """get more predictions used for calculating difference losses

        Args:
            pred_dict (dict[str, Tensor]): raw prediction dict given by the model

        Returns:
            dict[str, Tensor]: updated predictions
        """
        if not self.configs.loss_metrics_sparse_enable:
            pred_dict["distance"] = self.cdist(
                pred_dict["coordinate"], pred_dict["coordinate"])
        return pred_dict

    def _calculate_individual_losses(
        self,
        pred_dict,
        label_dict,
        feat_dict,
        diffusion_per_sample_scale,
        coord_mask,
        per_atom_lddt,
        per_atom_weight,
    ):
        """
        Calculate all individual loss components used during model training.

        Args:
            pred_dict (dict): Dictionary containing model predictions.
            label_dict (dict): Dictionary containing ground-truth labels.
            feat_dict (Any): Dictionary or object containing input features and annotations.
            diffusion_per_sample_scale (Tensor or None): Per-sample scaling factors for diffusion loss (if used).
            coord_mask (Tensor): Mask indicating which coordinates are valid for loss computation.
            per_atom_lddt (Tensor): Per-atom lDDT scores (if available, may be used by individual losses).
            per_atom_weight (Tensor): Per-atom weighting factors for losses.

        Returns:
            tuple: Individual loss components (should be unpacked/consumed by the calling function).
        """
        smooth_lddt = self.smooth_lddt_loss(
            pred_coordinate=pred_dict["coordinate"],
            true_coordinate=label_dict["coordinate"],
            lddt_mask=label_dict["lddt_mask"],
            diffusion_chunk_size=None,
        )
        mse = self.mse_loss(
            pred_coordinate=pred_dict["coordinate"],
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            is_rna=feat_dict.is_rna,
            is_dna=feat_dict.is_dna,
            is_ligand=feat_dict.is_ligand,
            per_sample_scale=diffusion_per_sample_scale,
        )

        distogram_logits = pred_dict["distogram"]["logits"]
        distogram = self.distogram_loss(
            logits=distogram_logits,
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            rep_atom_mask=feat_dict.distogram_rep_atom_mask,
        )

        bond = self.bond_loss.sparse_forward(
            pred_coordinate=pred_dict["coordinate"],
            true_coordinate=label_dict["coordinate"],
            distance_mask=label_dict["distance_mask"],
            bond_mask=feat_dict.bond_mask,
            per_sample_scale=diffusion_per_sample_scale,
        )

        pae = self.pae_loss(
            logits=pred_dict["pae"],
            pred_coordinate=pred_dict["coordinate_mini"],
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            frame_atom_index=feat_dict.frame_atom_index,
            rep_atom_mask=feat_dict.pae_rep_atom_mask,
            has_frame=feat_dict.frames_mask,
        )

        resolved = self.resolved_loss(
            logits=pred_dict["resolved"],
            coordinate_mask=label_dict["coordinate_mask"],
        )

        pde = self.pde_loss(
            logits=pred_dict["pde"],
            pred_coordinate=pred_dict["coordinate_mini"],
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            rep_atom_mask=feat_dict.distogram_rep_atom_mask,
        )

        plddt = self.plddt_loss(
            logits=pred_dict["plddt"][..., coord_mask, :],
            per_atom_lddt=per_atom_lddt,
            per_atom_weight=per_atom_weight,
        )

        return {
            "smooth_lddt": smooth_lddt,
            "mse_loss": mse,
            "distogram_loss": distogram,
            "bond_loss": bond,
            "pae_loss": pae,
            "resolved_loss": resolved,
            "pde_loss": pde,
            "plddt_loss": plddt,
        }

    def construct(
        self,
        feat_dict,
        pred_dict: dict[str, Tensor],
        label_dict: dict[str, Any],
        mode: str = "train",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[Tensor, dict[str, Tensor]]:
                - cum_loss (Tensor): Cumulative loss.
                - losses (dict[str, Tensor]): Dictionary containing aggregated metrics.
        """

        if mode not in ["train", "eval", "inference"]:
            raise ValueError(f"Invalid mode {mode}")
        # Pre-computations
        with _no_grad():
            label_dict = self.calculate_label(feat_dict, label_dict)

        pred_dict = self.calculate_prediction(pred_dict)

        if not self.configs.train_confidence_only:
            # Scale diffusion loss with noise-level
            diffusion_per_sample_scale = (
                pred_dict["noise_level"] ** 2 + self.configs.sigma_data**2
            ) / (self.configs.sigma_data * pred_dict["noise_level"]) ** 2
        else:
            diffusion_per_sample_scale = None

        coord_mask = label_dict["coordinate_mask"].squeeze().bool()
        per_atom_lddt, per_atom_weight = self.calculate_atom_bespoke_lddt(
            pred_coordinate=pred_dict["coordinate_mini"][..., coord_mask, :],
            true_coordinate=label_dict["coordinate"][..., coord_mask, :],
            is_nucleotide=(feat_dict.is_rna.int() + feat_dict.is_dna.int())[
                coord_mask
            ].bool(),
            is_polymer=1 - feat_dict.is_ligand[coord_mask],
            rep_atom_mask=feat_dict.plddt_m_rep_atom_mask[coord_mask].bool(),
            **self.lddt_radius,
        )

        losses = self._calculate_individual_losses(
            pred_dict,
            label_dict,
            feat_dict,
            diffusion_per_sample_scale,
            coord_mask,
            per_atom_lddt,
            per_atom_weight,
        )

        cum_loss = (
            self.loss_weight["smooth_lddt_loss"] * losses["smooth_lddt"]
            + self.loss_weight["mse_loss"] * losses["mse_loss"]
            + self.loss_weight["distogram_loss"] * losses["distogram_loss"]
            + self.loss_weight["bond_loss"] * losses["bond_loss"]
            + self.loss_weight["pae_loss"] * losses["pae_loss"]
            + self.loss_weight["resolved_loss"] * losses["resolved_loss"]
            + self.loss_weight["pde_loss"] * losses["pde_loss"]
            + self.loss_weight["plddt_loss"] * losses["plddt_loss"]
        )

        return cum_loss, losses

    def calculate_atom_bespoke_lddt(
        self,
        pred_coordinate: Tensor,
        true_coordinate: Tensor,
        is_nucleotide: Tensor,
        is_polymer: Tensor,
        rep_atom_mask: Tensor,
        is_nucleotide_threshold: float = 30.0,
        is_not_nucleotide_threshold: float = 15.0,
    ) -> tuple[Tensor, Tensor]:
        """calculate the bespoke lddt as described in Sec 4.3.1.
        Args:
            pred_coordinate (Tensor):
                [..., n_sample, n_atom, 3]
            true_coordinate (Tensor):
                [..., n_atom]
            is_nucleotide (Tensor):
                [n_atom] or [..., n_atom]
            is_polymer (Tensor):
                [n_atom]
            rep_atom_mask (Tensor):
                [n_atom]
        Returns:
            Tensor: per-atom lddt
                [..., n_sample, n_atom, 1]
            Tensor: per-atom lddt weight
                [..., n_sample, n_atom, 1]
        """

        n_atom = true_coordinate.shape[-2]
        atom_m_mask = (rep_atom_mask * is_polymer).bool()  # [n_atom]
        # Distance: d_lm
        pred_d_lm = self.cdist(
            pred_coordinate, pred_coordinate[..., atom_m_mask, :])
        true_d_lm = self.cdist(
            true_coordinate, true_coordinate[..., atom_m_mask, :])
        delta_d_lm = ops.abs(
            pred_d_lm - ops.expand_dims(true_d_lm, -3)
        )  # [..., n_sample, n_atom, n_atom(m)]
        # Pair-wise lddt
        thresholds = [0.5, 1, 2, 4]
        lddt_lm = (
            ops.stack([delta_d_lm < t for t in thresholds], -1)
            .astype(delta_d_lm.dtype)
            .mean(-1)
        )  # [..., n_sample, n_atom, n_atom(m)]
        # Select atoms that are within certain threshold to l in ground truth
        # Restrict to bespoke inclusion radius
        is_nucleotide = is_nucleotide[
            ..., atom_m_mask
        ].bool()  # [n_atom(m)] or [..., n_atom(m)]
        locality_mask = ((
            true_d_lm < is_nucleotide_threshold
        ) * ops.expand_dims(
            is_nucleotide, -2
        ).int() + (true_d_lm < is_not_nucleotide_threshold) * (
            ~ops.expand_dims(is_nucleotide, -2).int()
        ).bool()
        )  # [..., n_atom, n_atom(m)]
        # Remove self-distance computation
        diagonal_mask = (1 - ops.eye(n_atom, dtype=ms.bool_))[
            ..., atom_m_mask
        ]  # [n_atom, n_atom(m)]
        # [..., 1, n_atom, n_atom(m)]
        pair_mask = ops.expand_dims(locality_mask * diagonal_mask, -3)
        per_atom_lddt = ops.sum(
            lddt_lm * pair_mask, -1
        )  # [...,  n_sample, n_atom, 1]
        per_atom_lddt = ops.expand_dims(
            per_atom_lddt, -1)  # Add keepdim manually
        per_atom_weight = ops.sum(pair_mask.astype(lddt_lm.dtype), -1)
        per_atom_weight = ops.expand_dims(
            per_atom_weight, -1)  # Add keepdim manually
        return per_atom_lddt, per_atom_weight

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

"""sample confidence"""

from typing import Optional, Union
import mindspore as ms
from mindspore import mint
from ml_collections.config_dict import ConfigDict

from protenix.metrics.clash import Clash


def _compute_full_data_dict(
    configs, plddt_logits, pde_logits, pae_logits, contact_probs
):
    """
    Computes a dictionary containing confidence-related data from network outputs.

    This function processes raw network prediction logits (e.g. for pLDDT, PDE, PAE, and contact probability)
    and constructs a data dictionary with mapped confidence values for downstream analyses and reporting.

    Args:
        configs: Configuration object containing loss and binning parameters.
        plddt_logits (Tensor): Predicted logits for per-atom LDDT scores, shape [N_s, N_atom, ...].
        pde_logits (Tensor): Predicted logits for per-residue (pair) distance error, shape [N_s, N_token, N_token, ...].
        pae_logits (Tensor): Predicted logits for predicted aligned error, shape [N_s, N_token, N_token, ...].
        contact_probs (Tensor): Predicted contact probabilities, shape [N_token, N_token, ...].

    Returns:
        full_data (dict): Dictionary with processed confidence arrays:
            - 'atom_plddt': Per-atom pLDDT score [N_s, N_atom].
            - 'token_pair_pde': Per-token pairwise distance error [N_s, N_token, N_token].
            - 'contact_probs': Contact probabilities [N_token, N_token].
            - 'token_pair_pae': Per-token pairwise aligned error [N_s, N_token, N_token].
        pae_prob (Tensor): Probability tensor for PAE, for further summary statistics.
    """
    full_data = {}
    full_data["atom_plddt"] = logits_to_score(
        plddt_logits, **get_bin_params(configs.loss.plddt)
    )  # [N_s, N_atom]
    # Cpu offload for saving cuda memory
    full_data["token_pair_pde"] = logits_to_score(
        pde_logits, **get_bin_params(configs.loss.pde)
    )  # [N_s, N_token, N_token]
    full_data["contact_probs"] = contact_probs.clone()  # [N_token, N_token]
    full_data["token_pair_pae"], pae_prob = logits_to_score(
        pae_logits, **get_bin_params(configs.loss.pae), return_prob=True
    )  # [N_s, N_token, N_token]
    return full_data, pae_prob


def _compute_base_summary(
    configs, full_data, pae_prob, token_has_frame, token_asym_id
):
    """
    Compute basic summary statistics of model confidence for structure predictions.

    This function calculates summary-level metrics from full_data for per-atom and per-residue predictions, 
    along with global structure confidence scores such as predicted LDDT (plddt),
    gpde (predicted distance errors weighted by contact probability),
    and predicted TM-score (ptm, iptm), based on predicted logits and probability distributions.

    Args:
        configs: Configuration object holding bin parameters for conversion of logits to scores,
        and other loss settings.
        full_data (dict): Dictionary containing computed per-residue and per-pair scores,
        such as atom_plddt, token_pair_pde, contact_probs, token_pair_pae.
        pae_prob (Tensor): Predicted PAE probability tensor, used for TM/iptm calculation.
        token_has_frame (Tensor): Mask indicating presence of a backbone frame for each token.
        token_asym_id (Tensor): Chain identifiers for each token.

    Returns:
        summary_confidence (dict): Dictionary holding scalar summary metrics (per-sample):
            - 'plddt': Mean per-atom LDDT score, scaled to [0, 100].
            - 'gpde': Global predicted distance error, weighted by contact probability.
            - 'ptm': Predicted TM-score.
            - 'iptm': Predicted inter-chain TM-score.
    """
    summary_confidence = {}
    summary_confidence["plddt"] = full_data["atom_plddt"].mean(
        dim=-1) * 100  # [N_s, ]
    summary_confidence["gpde"] = (
        full_data["token_pair_pde"] * full_data["contact_probs"]
    ).sum(dim=[-1, -2]) / full_data["contact_probs"].sum(dim=[-1, -2])

    summary_confidence["ptm"] = calculate_ptm(
        pae_prob, has_frame=token_has_frame, **get_bin_params(configs.loss.pae)
    )  # [N_s, ]
    summary_confidence["iptm"] = calculate_iptm(
        pae_prob,
        has_frame=token_has_frame,
        asym_id=token_asym_id,
        **get_bin_params(configs.loss.pae)
    )  # [N_s, ]
    return summary_confidence


def _update_chain_metrics(
    summary_confidence,
    full_data,
    pae_prob,
    token_has_frame,
    token_asym_id,
    token_is_ligand,
    atom_to_token_idx,
    configs,
):
    """
    Update the summary_confidence dictionary with chain-level quality metrics.

    This function computes and adds per-chain and per-chain-pair metrics derived from model outputs 
    (e.g., global pLDDT, gpde, PTM, IPTM) to the summary_confidence dictionary. It utilizes input data 
    such as model predictions, probability matrices, and chain or atom indices to calculate metrics 
    describing the confidence and accuracy of predicted protein structures at the chain and chain-pair levels.

    Args:
        summary_confidence (dict): Dictionary holding aggregate model prediction metrics, to be updated in place.
        full_data (dict): Dictionary containing model prediction outputs,
        including 'atom_plddt', 'token_pair_pde', and 'contact_probs'.
        pae_prob (Tensor): Predicted probability matrix for inter-residue distances/errors
        (shape: [N_s, N_token, N_token]).
        token_has_frame (Tensor): Boolean mask of tokens with valid backbone frame.
        token_asym_id (Tensor): Array mapping tokens to chain (asym_id) indices.
        token_is_ligand (Tensor): Boolean mask indicating ligand tokens.
        atom_to_token_idx (Tensor): Mapping from atom indices to token indices.
        configs (object): Configuration object, typically containing loss parameterization
        (e.g. binning for pLDDT, pde, pae).

    Returns:
        None. The function updates summary_confidence in place with additional chain-based metrics:
            - 'chain_gpde', 'chain_pair_gpde'
            - 'chain_pair_iptm', 'chain_pair_iptm_global'
            - 'chain_iptm', 'chain_ptm'
            - 'chain_plddt', 'chain_pair_plddt'
    """
    # Add: 'chain_gpde', 'chain_pair_gpde'
    summary_confidence.update(
        calculate_chain_based_gpde(
            token_pair_pde=full_data["token_pair_pde"],
            contact_probs=full_data["contact_probs"],
            asym_id=token_asym_id,
        )
    )
    # Add: 'chain_pair_iptm', 'chain_pair_iptm_global' 'chain_iptm', 'chain_ptm'
    summary_confidence.update(
        calculate_chain_based_ptm(
            pae_prob,
            has_frame=token_has_frame,
            asym_id=token_asym_id,
            token_is_ligand=token_is_ligand,
            **get_bin_params(configs.loss.pae)
        )
    )
    # Add: 'chain_plddt', 'chain_pair_plddt'
    summary_confidence.update(
        calculate_chain_based_plddt(
            full_data["atom_plddt"], token_asym_id, atom_to_token_idx
        )
    )


def _compute_vdw_clash_metrics(
    atom_coordinate,
    token_asym_id,
    atom_to_token_idx,
    atom_is_polymer,
    mol_id,
    elements_one_hot,
    configs,
    summary_confidence,
    interested_asym_id,
):
    """
    Compute and update van der Waals (VDW) clash-related metrics in the summary_confidence dictionary.

    This function calculates VDW clashes between predicted atom coordinates and updates the 
    summary_confidence with flags and penalized ranking scores. For each sample, it determines 
    if any VDW clash exists for a set of "interested" chains, and applies a penalty to the 
    pb_ranking_score accordingly.

    Args:
        atom_coordinate (ms.Tensor): Predicted atomic coordinates, with shape (num_samples, ...).
        token_asym_id (ms.Tensor): Array of chain (asym_id) assignments for tokens.
        atom_to_token_idx (ms.Tensor): Mapping from atom indices to token indices.
        atom_is_polymer (ms.Tensor): Boolean mask indicating polymer status for each atom.
        mol_id (ms.Tensor): Molecule IDs for each atom.
        elements_one_hot (ms.Tensor): One-hot encoded element types for atoms.
        configs: Configuration object with metrics for clash thresholds.
        summary_confidence (dict): Dictionary to update with clash metrics and penalized scores.
        interested_asym_id (ms.Tensor): Indices for chains of interest to evaluate for VDW clashes.

    Returns:
        None. The function updates summary_confidence in place with:
            - 'has_vdw_pl_clash': Boolean flags for clashes per sample.
            - 'pb_ranking_score_vdw_penalized': Updated ranking score applying clash penalty.
    """
    vdw_clash = calculate_vdw_clash(
        pred_coordinate=atom_coordinate,
        asym_id=token_asym_id,
        mol_id=mol_id,
        is_polymer=atom_is_polymer,
        atom_token_idx=atom_to_token_idx,
        elements_one_hot=elements_one_hot,
        threshold=configs.metrics.clash.vdw_clash_threshold,
    )
    num_sample = atom_coordinate.shape[0]
    vdw_clash_per_sample_flag = (
        vdw_clash[:, interested_asym_id, :]
        .reshape(num_sample, -1)
        .max(dim=-1)[0]
    )
    summary_confidence["has_vdw_pl_clash"] = vdw_clash_per_sample_flag
    summary_confidence["pb_ranking_score_vdw_penalized"] = (
        summary_confidence["pb_ranking_score"] -
        100 * vdw_clash_per_sample_flag
    )


def _compute_interested_atom_metrics(
    interested_atom_mask,
    atom_to_token_idx,
    token_asym_id,
    summary_confidence,
    atom_coordinate,
    atom_is_polymer,
    elements_one_hot,
    mol_id,
    configs,
):
    """
    Compute metrics for atoms of interest and update the summary confidence dictionary accordingly.

    This method processes the atoms indicated by `interested_atom_mask` to determine
    chain and token indices of interest, and updates the summary_confidence with additional
    ranking scores and metrics. It ensures that all selected atoms belong to the same chain
    (`asym_id`) and extracts relevant chain-specific ranking scores. If element information
    and molecule IDs are provided, it further computes van der Waals clash metrics and
    integrates penalized ranking scores into summary_confidence.

    Args:
        interested_atom_mask (ms.Tensor): Boolean mask indicating atoms of interest.
        atom_to_token_idx (ms.Tensor): Mapping from atom indices to token indices.
        token_asym_id (ms.Tensor): Array of chain IDs (`asym_id`) for each token.
        summary_confidence (dict): Dictionary to be updated with computed metrics.
        atom_coordinate (ms.Tensor): Coordinates of all atoms.
        atom_is_polymer (ms.Tensor): Boolean array indicating if each atom is a polymer.
        elements_one_hot (Optional[ms.Tensor]): One-hot array of atom elements. Optional.
        mol_id (Optional[ms.Tensor]): Molecule IDs for each atom. Optional.
        configs: Configuration object containing relevant thresholds and metrics.

    Returns:
        None. The function updates the `summary_confidence` dictionary in-place with chain-specific
        ranking scores and penalized ranking scores reflecting possible clashes.
    """
    if interested_atom_mask is not None:
        token_idx = atom_to_token_idx[interested_atom_mask[0].bool()].long()
        asym_ids = token_asym_id[token_idx]
        assert len(mint.unique(asym_ids)) == 1
        interested_asym_id = asym_ids[0].item()
        num_chains = token_asym_id.max().long().item() + 1
        pb_ranking_score = summary_confidence["chain_pair_iptm_global"][
            :, interested_asym_id, mint.arange(num_chains) != interested_asym_id
        ]  # [N_s, num_chain - 1]
        summary_confidence["pb_ranking_score"] = pb_ranking_score[:, 0]
        if elements_one_hot is not None and mol_id is not None:
            _compute_vdw_clash_metrics(
                atom_coordinate,
                token_asym_id,
                atom_to_token_idx,
                atom_is_polymer,
                mol_id,
                elements_one_hot,
                configs,
                summary_confidence,
                interested_asym_id,
            )


def _compute_full_data_and_summary(
    configs: ConfigDict,
    pae_logits: ms.Tensor,
    plddt_logits: ms.Tensor,
    pde_logits: ms.Tensor,
    contact_probs: ms.Tensor,
    token_asym_id: ms.Tensor,
    token_has_frame: ms.Tensor,
    atom_coordinate: ms.Tensor,
    atom_to_token_idx: ms.Tensor,
    atom_is_polymer: ms.Tensor,
    num_recycle: int,
    interested_atom_mask: Optional[ms.Tensor] = None,
    elements_one_hot: Optional[ms.Tensor] = None,
    mol_id: Optional[ms.Tensor] = None,
    return_full_data: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Compute full data and summary confidence scores for the given inputs.
    """
    atom_is_ligand = (1 - atom_is_polymer).long()
    token_is_ligand = (
        mint.zeros(token_asym_id.shape).astype(ms.int32)
        .scatter_add(0, atom_to_token_idx.astype(
                ms.int32), atom_is_ligand.astype(ms.int32)
        )
    )
    token_is_ligand = token_is_ligand > 0

    full_data, pae_prob = _compute_full_data_dict(
        configs, plddt_logits, pde_logits, pae_logits, contact_probs
    )

    summary_confidence = _compute_base_summary(
        configs, full_data, pae_prob, token_has_frame, token_asym_id
    )

    _update_chain_metrics(
        summary_confidence, full_data, pae_prob, token_has_frame,
        token_asym_id, token_is_ligand, atom_to_token_idx, configs,
    )
    summary_confidence["has_clash"] = calculate_clash(
        atom_coordinate, token_asym_id,
        atom_to_token_idx, atom_is_polymer,
        configs.metrics.clash.af3_clash_threshold,
    )
    summary_confidence["num_recycles"] = ms.tensor(num_recycle)
    summary_confidence["disorder"] = mint.zeros(
        summary_confidence["ptm"].shape)
    summary_confidence["ranking_score"] = (
        0.8 * summary_confidence["iptm"]
        + 0.2 * summary_confidence["ptm"]
        + 0.5 * summary_confidence["disorder"]
        - 100 * summary_confidence["has_clash"]
    )
    _compute_interested_atom_metrics(
        interested_atom_mask, atom_to_token_idx,
        token_asym_id, summary_confidence,
        atom_coordinate, atom_is_polymer,
        elements_one_hot, mol_id, configs,
    )

    summary_confidence = break_down_to_per_sample_dict(
        summary_confidence, shared_keys=["num_recycles"]
    )
    if return_full_data:
        # save extra inputs that are used for computing summary_confidence
        full_data["token_has_frame"] = token_has_frame.clone()
        full_data["token_asym_id"] = token_asym_id.clone()
        full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
        full_data["atom_is_polymer"] = atom_is_polymer.clone()
        full_data["atom_coordinate"] = atom_coordinate.clone()

        full_data = break_down_to_per_sample_dict(
            full_data,
            shared_keys=[
                "contact_probs", "token_has_frame",
                "token_asym_id", "atom_to_token_idx",
                "atom_is_polymer",
            ],
        )
        return summary_confidence, full_data
    return summary_confidence, [{}]


def get_bin_params(cfg: ConfigDict) -> dict:
    """
    Extract bin parameters from the configuration object.
    """
    return {"min_bin": cfg.min_bin, "max_bin": cfg.max_bin, "no_bins": cfg.no_bins}


def compute_contact_prob(
    distogram_logits: ms.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres=8.0,
) -> ms.Tensor:
    """
    Compute the contact probability from distogram logits.

    Args:
        distogram_logits (ms.Tensor): Logits for the distogram.
            Shape: [N_token, N_token, N_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        thres (float): Threshold distance for contact probability. Defaults to 8.0.

    Returns:
        ms.Tensor: Contact probability.
            Shape: [N_token, N_token]
    """
    distogram_prob = mint.nn.functional.softmax(
        distogram_logits, dim=-1
    )  # [N_token, N_token, N_bins]
    distogram_bins = get_bin_centers(min_bin, max_bin, no_bins)
    thres_idx = (distogram_bins < thres).sum()
    contact_prob = distogram_prob[..., :thres_idx].sum(-1)
    return contact_prob


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> ms.Tensor:
    """
    Calculate the centers of the bins for a given range and number of bins.

    Args:
        min_bin (float): The minimum value of the bin range.
        max_bin (float): The maximum value of the bin range.
        no_bins (int): The number of bins.

    Returns:
        ms.Tensor: The centers of the bins.
            Shape: [no_bins]
    """
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = mint.linspace(
        start=min_bin,
        end=max_bin - bin_width,
        steps=no_bins,
    )
    bin_centers = boundaries + 0.5 * bin_width
    return bin_centers


def logits_to_prob(logits: ms.Tensor, dim=-1) -> ms.Tensor:
    return mint.nn.functional.softmax(logits, dim=dim)


def logits_to_score(
    logits: ms.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    return_prob=False,
) -> Union[ms.Tensor, tuple[ms.Tensor, ms.Tensor]]:
    """
    Convert logits to a score using bin centers.

    Args:
        logits (ms.Tensor): Logits tensor.
            Shape: [..., no_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        return_prob (bool): Whether to return the probability distribution. Defaults to False.

    Returns:
        score (ms.Tensor): Converted score.
            Shape: [...]
        prob (ms.Tensor, optional): Probability distribution if `return_prob` is True.
            Shape: [..., no_bins]
    """
    prob = logits_to_prob(logits, dim=-1)
    bin_centers = get_bin_centers(min_bin, max_bin, no_bins)
    score = prob @ bin_centers
    if return_prob:
        return score, prob
    return score


def calculate_normalization(n):
    # TM-score normalization constant
    return 1.24 * (max(n, 19) - 15) ** (1 / 3) - 1.8


def calculate_vdw_clash(
    pred_coordinate: ms.Tensor,
    asym_id: ms.Tensor,
    mol_id: ms.Tensor,
    atom_token_idx: ms.Tensor,
    is_polymer: ms.Tensor,
    elements_one_hot: ms.Tensor,
    threshold: float,
) -> ms.Tensor:
    """
    Calculate Van der Waals (VDW) clash for predicted coordinates.

    Args:
        pred_coordinate (ms.Tensor): Predicted coordinates of atoms.
            Shape: [num_sample, N_atom, 3]
        asym_id (ms.Tensor): Asymmetric ID for tokens.
            Shape: [N_token]
        mol_id (ms.Tensor): Molecular ID.
            Shape: [N_atom]
        atom_token_idx (ms.Tensor): Mapping from atoms to tokens.
            Shape: [N_atom]
        is_polymer (ms.Tensor): Indicator for atoms being part of a polymer.
            Shape: [N_atom]
        elements_one_hot (ms.Tensor): One-hot encoding for elements.
            Shape: [N_atom, N_elements]
        threshold (float): Threshold for VDW clash detection.

    Returns:
        ms.Tensor: VDW clash summary.
            Shape: [num_sample]
    """
    clash_calculator = Clash(
        vdw_clash_threshold=threshold, compute_af3_clash=False)
    # Check ligand-polymer VDW clash
    dummy_is_dna = mint.zeros(is_polymer.shape)
    dummy_is_rna = mint.zeros(is_polymer.shape)
    clash_dict = clash_calculator(
        pred_coordinate=pred_coordinate,
        asym_id=asym_id,
        atom_to_token_idx=atom_token_idx,
        mol_id=mol_id,
        is_ligand=1 - is_polymer,
        is_protein=is_polymer,
        is_dna=dummy_is_dna,
        is_rna=dummy_is_rna,
        elements_one_hot=elements_one_hot,
    )
    return clash_dict["summary"]["vdw_clash"]


def calculate_clash(
    pred_coordinate: ms.Tensor,
    asym_id: ms.Tensor,
    atom_to_token_idx: ms.Tensor,
    is_polymer: ms.Tensor,
    threshold: float,
) -> ms.Tensor:
    """Check complex clash

    Args:
        pred_coordinate (ms.Tensor): [num_sample, N_atom, 3]
        asym_id (ms.Tensor): [N_token, ]
        atom_to_token_idx (ms.Tensor): [N_atom, ]
        is_polymer (ms.Tensor): [N_atom, ]
        threshold: (float)

    Returns:
        ms.Tensor: [num_sample] whether there is a clash in the complex
    """
    num_sample = pred_coordinate.shape[0]
    dummy_is_dna = mint.zeros(is_polymer.shape)
    dummy_is_rna = mint.zeros(is_polymer.shape)
    clash_calculator = Clash(
        vdw_clash_threshold=threshold, compute_vdw_clash=False)
    clash_dict = clash_calculator(
        pred_coordinate,
        asym_id,
        atom_to_token_idx,
        1 - is_polymer,
        is_polymer,
        dummy_is_dna,
        dummy_is_rna,
    )
    return clash_dict["summary"]["af3_clash"].reshape(num_sample, -1).max(dim=-1)[0]


def calculate_ptm(
    pae_prob: ms.Tensor,
    has_frame: ms.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    """Compute pTM score

    Args:
        pae_prob (ms.Tensor): Predicted probability from PAE loss head.
            Shape: [..., N_token, N_token, N_bins]
        has_frame (ms.Tensor): Indicator for tokens having a frame.
            Shape: [N_token, ]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        token_mask (Optional[ms.Tensor]): Mask for tokens.
            Shape: [N_token, ] or None

    Returns:
        ms.Tensor: pTM score. Higher values indicate better ranking.
            Shape: [...]
    """
    has_frame = has_frame.bool()

    if token_mask is not None:
        token_mask = token_mask.bool()
        pae_prob = pae_prob[..., token_mask, :, :][
            ..., :, token_mask, :
        ]  # [..., n_d, n_d, N_bins]
        has_frame = has_frame[token_mask]  # [n_d, ]

    if has_frame.sum() == 0:
        return mint.zeros(size=pae_prob.shape[:-3])

    n_d = has_frame.shape[-1]
    ptm_norm = calculate_normalization(n_d)

    bin_center = get_bin_centers(min_bin, max_bin, no_bins)
    per_bin_weight = (1 / (1 + (bin_center / ptm_norm) ** 2))  # [N_bins]

    # [..., n_d, n_d]
    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)

    ptm = token_token_ptm.mean(dim=-1)[..., has_frame].max(dim=-1)[0]
    return ptm


def _compute_chain_pair_iptm(
    pae_prob: ms.Tensor,
    has_frame: ms.Tensor,
    asym_id: ms.Tensor,
    asym_id_to_asym_mask: dict,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    num_chain: int,
    batch_shape: tuple,
) -> ms.Tensor:
    """Compute pairwise chain ipTM scores."""
    chain_pair_iptm = mint.zeros(size=batch_shape + (num_chain, num_chain))
    for aid_1 in range(num_chain):
        for aid_2 in range(aid_1 + 1, num_chain):
            pair_mask = asym_id_to_asym_mask[aid_1].astype(
                ms.int32) + asym_id_to_asym_mask[aid_2].astype(ms.int32)
            iptm_value = calculate_iptm(
                pae_prob,
                has_frame,
                asym_id,
                min_bin,
                max_bin,
                no_bins,
                token_mask=pair_mask,
            )
            chain_pair_iptm[:, aid_1, aid_2] = iptm_value
            chain_pair_iptm[:, aid_2, aid_1] = iptm_value
    return chain_pair_iptm


def _compute_chain_ptm(
    pae_prob: ms.Tensor,
    has_frame: ms.Tensor,
    asym_id_to_asym_mask: dict,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    batch_shape: tuple,
    num_chain: int,
) -> ms.Tensor:
    """Compute per-chain pTM scores."""
    chain_ptm = mint.zeros(size=batch_shape + (num_chain,))
    for aid, asym_mask in asym_id_to_asym_mask.items():
        chain_ptm[:, aid] = calculate_ptm(
            pae_prob,
            has_frame,
            min_bin,
            max_bin,
            no_bins,
            token_mask=asym_mask,
        )
    return chain_ptm


def _compute_chain_iptm(
    chain_pair_iptm: ms.Tensor,
    has_frame: ms.Tensor,
    asym_id_to_asym_mask: dict,
    num_chain: int,
    batch_shape: tuple,
) -> ms.Tensor:
    """Compute per-chain interface pTM scores."""
    chain_has_frame = [
        (asym_id_to_asym_mask[i] * has_frame).any() for i in range(num_chain)
    ]

    chain_iptm = mint.zeros(size=batch_shape + (num_chain,))
    for aid in range(num_chain):
        pairs = [
            (i, j)
            for i in range(num_chain)
            for j in range(num_chain)
            if aid in (i, j) and (i != j) and chain_has_frame[i]
        ]
        vals = [chain_pair_iptm[:, i, j] for (i, j) in pairs]
        if len(vals) > 0:
            chain_iptm[:, aid] = mint.stack(vals, dim=-1).mean(dim=-1)
    return chain_iptm


def _compute_chain_pair_iptm_global(
    chain_iptm: ms.Tensor,
    chain_is_ligand: dict,
    num_chain: int,
    batch_shape: tuple,
) -> ms.Tensor:
    """Compute global pairwise chain interface pTM scores."""
    chain_pair_iptm_global = mint.zeros(
        size=batch_shape + (num_chain, num_chain))
    for aid_1 in range(num_chain):
        for aid_2 in range(num_chain):
            if aid_1 == aid_2:
                continue
            if chain_is_ligand[aid_1]:
                chain_pair_iptm_global[:, aid_1, aid_2] = chain_iptm[:, aid_1]
            elif chain_is_ligand[aid_2]:
                chain_pair_iptm_global[:, aid_1, aid_2] = chain_iptm[:, aid_2]
            else:
                chain_pair_iptm_global[:, aid_1, aid_2] = (
                    chain_iptm[:, aid_1] + chain_iptm[:, aid_2]
                ) * 0.5
    return chain_pair_iptm_global


def calculate_chain_based_ptm(
    pae_prob: ms.Tensor,
    has_frame: ms.Tensor,
    asym_id: ms.Tensor,
    token_is_ligand: ms.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
) -> dict[str, ms.Tensor]:
    """
    Compute chain-based pTM scores.

    Args:
        pae_prob (ms.Tensor): Predicted probability from PAE loss head.
            Shape: [..., N_token, N_token, N_bins]
        has_frame (ms.Tensor): Indicator for tokens having a frame.
            Shape: [N_token, ]
        asym_id (ms.Tensor): Asymmetric ID for tokens.
            Shape: [N_token, ]
        token_is_ligand (ms.Tensor): Indicator for tokens being ligands.
            Shape: [N_token, ]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.

    Returns:
        dict: Dictionary containing chain-based pTM scores.
            - chain_ptm (ms.Tensor): pTM scores for each chain.
            - chain_iptm (ms.Tensor): ipTM scores for chain interface.
            - chain_pair_iptm (ms.Tensor): Pairwise ipTM scores between chains.
            - chain_pair_iptm_global (ms.Tensor): Global pairwise ipTM scores between chains.
    """

    has_frame = has_frame.bool()
    asym_id = asym_id.long()
    asym_id_to_asym_mask = {
        aid.item(): asym_id == aid for aid in mint.unique(asym_id)}
    chain_is_ligand = {
        aid.item(): token_is_ligand[asym_id == aid].sum() >= (asym_id == aid).sum() // 2
        for aid in mint.unique(asym_id)
    }

    batch_shape = pae_prob.shape[:-3]
    num_chain = len(asym_id_to_asym_mask)

    # Compute chain pair ipTM
    chain_pair_iptm = _compute_chain_pair_iptm(
        pae_prob,
        has_frame,
        asym_id,
        asym_id_to_asym_mask,
        min_bin,
        max_bin,
        no_bins,
        num_chain,
        batch_shape,
    )

    # Compute chain pTM
    chain_ptm = _compute_chain_ptm(
        pae_prob,
        has_frame,
        asym_id_to_asym_mask,
        min_bin,
        max_bin,
        no_bins,
        batch_shape,
        num_chain,
    )

    # Compute chain ipTM
    chain_iptm = _compute_chain_iptm(
        chain_pair_iptm,
        has_frame,
        asym_id_to_asym_mask,
        num_chain,
        batch_shape,
    )

    # Compute chain pair ipTM global
    chain_pair_iptm_global = _compute_chain_pair_iptm_global(
        chain_iptm,
        chain_is_ligand,
        num_chain,
        batch_shape,
    )

    return {
        "chain_ptm": chain_ptm,
        "chain_iptm": chain_iptm,
        "chain_pair_iptm": chain_pair_iptm,
        "chain_pair_iptm_global": chain_pair_iptm_global,
    }


def calculate_chain_based_gpde(
    token_pair_pde: ms.Tensor,
    contact_probs: ms.Tensor,
    asym_id: ms.Tensor,
    eps: float = 1e-8,
) -> dict[str, ms.Tensor]:
    """Calculate chain-based gPDE values.

    Args:
        token_pair_pde (ms.Tensor): PDE (Predicted Distance Error) of token-token pairs.
            [..., N_token, N_token]
        contact_probs (ms.Tensor): Contact probabilities.
            [..., N_token, N_token]
        asym_id (ms.Tensor): Asymmetric ID for tokens.

    Returns:
        dict[str, ms.Tensor]: Dictionary containing chain-based gPDE values.
            - chain_gpde (ms.Tensor): Intra-chain gPDE.
            - chain_pair_gpde (ms.Tensor): Interface gPDE.
    """

    asym_id = asym_id.long()
    unique_asym_ids = mint.unique(asym_id)
    num_chain = len(unique_asym_ids)
    assert num_chain == asym_id.max() + 1  # make sure it is from 0 to num_chain-1

    batch_shape = token_pair_pde.shape[:-2]

    def _cal_gpde(token_mask_1, token_mask_2):
        masked_contact_probs = contact_probs[...,
                                             token_mask_1, :][..., token_mask_2]
        masked_pde = token_pair_pde[..., token_mask_1, :][..., token_mask_2]
        return (masked_pde * masked_contact_probs).sum(dim=(-1, -2)) / (
            masked_contact_probs.sum(dim=(-1, -2)) + eps
        )

    # Chain_gpde
    chain_gpde = mint.zeros(size=batch_shape + (num_chain,))
    for aid in range(num_chain):
        chain_gpde[..., aid] = _cal_gpde(
            token_mask_1=asym_id == aid,
            token_mask_2=asym_id == aid,
        )

    # Chain_pair_pde
    chain_pair_gpde = mint.zeros(size=batch_shape + (num_chain, num_chain))
    for aid_1 in range(num_chain):
        for aid_2 in range(num_chain):
            if aid_1 == aid_2:
                continue
            if aid_2 < aid_1:
                chain_pair_gpde[..., aid_1,
                                aid_2] = chain_pair_gpde[..., aid_2, aid_1]
                continue
            chain_pair_gpde[..., aid_1, aid_2] = _cal_gpde(
                token_mask_1=asym_id == aid_1,
                token_mask_2=asym_id == aid_2,
            )

    return {"chain_gpde": chain_gpde, "chain_pair_gpde": chain_pair_gpde}


def calculate_chain_based_plddt(
    atom_plddt: ms.Tensor,
    asym_id: ms.Tensor,
    atom_to_token_idx: ms.Tensor,
) -> dict[str, ms.Tensor]:
    """
    Calculate chain-based pLDDT scores.

    Args:
        atom_plddt (ms.Tensor): Predicted pLDDT scores for atoms.
            Shape: [num_sample, N_atom]
        asym_id (ms.Tensor): Asymmetric ID for tokens.
            Shape: [N_token]
        atom_to_token_idx (ms.Tensor): Mapping from atoms to tokens.
            Shape: [N_atom]

    Returns:
        dict: Dictionary containing chain-based pLDDT scores.
            - chain_plddt (ms.Tensor): pLDDT scores for each chain.
            - chain_pair_plddt (ms.Tensor): Pairwise pLDDT scores between chains.
    """

    asym_id = asym_id.long()
    asym_id_to_asym_mask = {
        aid.item(): asym_id == aid for aid in mint.unique(asym_id)}
    num_chain = len(asym_id_to_asym_mask)
    assert num_chain == asym_id.max() + 1  # make sure it is from 0 to num_chain-1

    def _calculate_lddt_with_token_mask(token_mask):
        atom_mask = token_mask[atom_to_token_idx]
        sub_plddt = atom_plddt[:, atom_mask].mean(-1)
        return sub_plddt

    batch_shape = atom_plddt.shape[:-1]
    # Chain_plddt
    chain_plddt = mint.zeros(size=batch_shape + (num_chain,))
    for aid, asym_mask in asym_id_to_asym_mask.items():
        chain_plddt[:, aid] = _calculate_lddt_with_token_mask(
            token_mask=asym_mask)

    # Chain_pair_plddt
    chain_pair_plddt = mint.zeros(size=batch_shape + (num_chain, num_chain))
    for aid_1 in asym_id_to_asym_mask:
        for aid_2 in asym_id_to_asym_mask:
            if aid_1 == aid_2:
                continue
            pair_mask = asym_id_to_asym_mask[aid_1].astype(
                ms.int32) + asym_id_to_asym_mask[aid_2].astype(ms.int32)
            chain_pair_plddt[:, aid_1, aid_2] = _calculate_lddt_with_token_mask(
                token_mask=pair_mask
            )

    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}


def calculate_iptm(
    pae_prob: ms.Tensor,
    has_frame: ms.Tensor,
    asym_id: ms.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[ms.Tensor] = None,
    eps: float = 1e-8,
):
    """
    Compute ipTM score.

    Args:
        pae_prob (ms.Tensor): Predicted probability from PAE loss head.
            Shape: [..., N_token, N_token, N_bins]
        has_frame (ms.Tensor): Indicator for tokens having a frame.
            Shape: [N_token, ]
        asym_id (ms.Tensor): Asymmetric ID for tokens.
            Shape: [N_token, ]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        token_mask (Optional[ms.Tensor]): Mask for tokens.
            Shape: [N_token, ] or None
        eps (float): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        ms.Tensor: ipTM score. Higher values indicate better ranking.
            Shape: [...]
    """
    has_frame = has_frame.bool()
    if token_mask is not None:
        token_mask = token_mask.bool()
        pae_prob = pae_prob[..., token_mask, :, :][
            ..., :, token_mask, :
        ]  # [..., n_d, n_d, N_bins]
        has_frame = has_frame[token_mask]  # [n_d, ]
        asym_id = asym_id[token_mask]  # [n_d, ]

    if has_frame.sum() == 0:
        return mint.zeros(size=pae_prob.shape[:-3])

    n_d = has_frame.shape[-1]
    ptm_norm = calculate_normalization(n_d)

    bin_center = get_bin_centers(min_bin, max_bin, no_bins)
    per_bin_weight = (1 / (1 + (bin_center / ptm_norm) ** 2))  # [N_bins]

    # [..., n_d, n_d]
    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)

    is_diff_chain = asym_id[None, :] != asym_id[:, None]  # [n_d, n_d]

    iptm = (token_token_ptm * is_diff_chain).sum(dim=-1) / (
        eps + is_diff_chain.sum(dim=-1)
    )  # [..., n_d]
    iptm = iptm[..., has_frame].max(dim=-1)[0]

    return iptm


def break_down_to_per_sample_dict(input_dict: dict, shared_keys=None) -> list[dict]:
    """
    Break down a dictionary containing tensors into a list of dictionaries, each corresponding to a sample.

    Args:
        input_dict (dict): Dictionary containing tensors.
        shared_keys (list): List of keys that are shared across all samples. Defaults to None.

    Returns:
        list[dict]: List of dictionaries, each containing data for a single sample.
    """
    if shared_keys is None:
        shared_keys = []
    per_sample_keys = [key for key in input_dict if key not in shared_keys]
    assert len(per_sample_keys) > 0
    num_sample = input_dict[per_sample_keys[0]].shape[0]
    for key in per_sample_keys:
        assert input_dict[key].shape[0] == num_sample

    per_sample_dict_list = []
    for i in range(num_sample):
        sample_dict = {key: input_dict[key][i] for key in per_sample_keys}
        sample_dict.update({key: input_dict[key] for key in shared_keys})
        per_sample_dict_list.append(sample_dict)

    return per_sample_dict_list


def compute_full_data_and_summary(
    configs,
    pae_logits,
    plddt_logits,
    pde_logits,
    contact_probs,
    token_asym_id,
    token_has_frame,
    atom_coordinate,
    atom_to_token_idx,
    atom_is_polymer,
    num_recycle,
    return_full_data: bool = False,
    interested_atom_mask=None,
    mol_id=None,
    elements_one_hot=None,
):
    """Wrapper of `_compute_full_data_and_summary` by enumerating over N samples"""

    num_sample = pae_logits.shape[0]
    if contact_probs.dim() == 2:
        # Convert to [num_sample, N_token, N_token]
        contact_probs = contact_probs.unsqueeze(
            dim=0).expand((num_sample, -1, -1))
    else:
        assert contact_probs.dim() == 3
    assert (
        contact_probs.shape[0] == plddt_logits.shape[0] == pde_logits.shape[0] == num_sample
    )

    summary_confidence = []
    full_data = []
    for i in range(num_sample):
        summary_confidence_i, full_data_i = _compute_full_data_and_summary(
            configs=configs,
            pae_logits=pae_logits[i: i + 1],
            plddt_logits=plddt_logits[i: i + 1],
            pde_logits=pde_logits[i: i + 1],
            contact_probs=contact_probs[i],
            token_asym_id=token_asym_id,
            token_has_frame=token_has_frame,
            atom_coordinate=atom_coordinate[i: i + 1],
            atom_to_token_idx=atom_to_token_idx,
            atom_is_polymer=atom_is_polymer,
            num_recycle=num_recycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=return_full_data,
            mol_id=mol_id,
            elements_one_hot=elements_one_hot,
        )
        summary_confidence.extend(summary_confidence_i)
        full_data.extend(full_data_i)
    return summary_confidence, full_data

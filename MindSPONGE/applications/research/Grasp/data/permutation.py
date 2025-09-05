import numpy as np
import pickle

from data import OUTPUT_LABEL_KEYS
from mindsponge1.common.residue_constants import atom_order
from mindsponge1.data.data_transform import pseudo_beta_fn

GT_KEYS = ["pseudo_beta", "pseudo_beta_mask", "residx_atom14_to_atom37",
            "backbone_affine_tensor", "backbone_affine_mask", "rigidgroups_gt_frames",
            "rigidgroups_gt_exists", "rigidgroups_alt_gt_frames", "torsion_angles_sin_cos", "chi_mask",
            "atom14_gt_positions", "atom14_alt_gt_positions", "atom14_atom_is_ambiguous", "atom14_gt_exists",
            "atom14_atom_exists", "atom14_alt_gt_exists", "all_atom_positions", "all_atom_mask",
            "true_msa", "bert_mask",
            "restype_atom14_bond_lower_bound","restype_atom14_bond_upper_bound","atomtype_radius",
            "use_clamped_fape", "filter_by_solution", "asym_mask"]


def multi_chain_perm_align_v3(final_atom_positions, input_feats, labels, shuffle_times=3):


    assert isinstance(labels, list)

    pred_cb_pos, pred_cb_mask = pseudo_beta_fn(input_feats["aatype"][0], final_atom_positions, input_feats["atom37_atom_exists"])
    pred_cb_pos, pred_cb_mask = pred_cb_pos.astype(np.float32), pred_cb_mask.astype(np.float32)
    true_cb_poses = []
    true_cb_masks = []
    for label in labels:
        true_cb_pose, true_cb_mask = pseudo_beta_fn(label["aatype_per_chain"], label["all_atom_positions"], label["all_atom_mask"])
        true_cb_poses.append(true_cb_pose.astype(np.float32))
        true_cb_masks.append(true_cb_mask.astype(np.float32))

    unique_asym_ids = np.unique(input_feats["asym_id"])

    per_asym_residue_index = {}
    for cur_asym_id in unique_asym_ids:
        asym_mask = (input_feats["asym_id"] == cur_asym_id).astype(bool)
        per_asym_residue_index[int(cur_asym_id)] = input_feats["residue_index"][asym_mask]



    unique_entity_ids = np.unique(input_feats["entity_id"])
    entity_2_asym_list = {}
    for cur_ent_id in unique_entity_ids:
        ent_mask = input_feats["entity_id"] == cur_ent_id
        cur_asym_id = np.unique(input_feats["asym_id"][ent_mask])
        entity_2_asym_list[int(cur_ent_id)] = cur_asym_id

    asym_2_entity_list = {}
    for ent, asys in entity_2_asym_list.items():
        for asy in asys:
            asym_2_entity_list[asy] = ent

    # find anchor pred chain
    anchor_gt_asym, anchor_pred_asym = get_anchor_candidates(
        input_feats, per_asym_residue_index, true_cb_masks
    )
    anchor_gt_idxs = entity_2_asym_list[asym_2_entity_list[anchor_gt_asym]]

    max_chain_length = 0
    for cur_asym_id in anchor_pred_asym:
        asym_mask = (input_feats["asym_id"] == cur_asym_id).astype(bool)
        if asym_mask.sum() > max_chain_length:
            max_chain_length = asym_mask.sum()
            final_asym_mask = asym_mask
            anchor_residue_idx = per_asym_residue_index[int(cur_asym_id)]
    
    # find optimal transforms
    best_rmsd = 1e9
    best_r, best_x = None, None
    for anchor_gt_idx in anchor_gt_idxs:
        anchor_gt_idx = anchor_gt_idx - 1
        anchor_true_pos = true_cb_poses[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_pos = pred_cb_pos[final_asym_mask]
        anchor_true_mask = true_cb_masks[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_mask = pred_cb_mask[final_asym_mask]
        r, x = get_optimal_transform(
            anchor_true_pos,
            anchor_pred_pos,
            (anchor_true_mask * anchor_pred_mask).astype(bool),
        )
        
        aligned_anchor_true_pos = anchor_true_pos @ r + x
        rmsd = compute_rmsd(aligned_anchor_true_pos, anchor_pred_pos, anchor_true_mask.astype(np.int32))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_r = r
            best_x = x

    best_labels = None
    aligned_true_cb_poses = [cb @ best_r + best_x for cb in true_cb_poses]  # apply transforms

    # greedy align
    best_rmsd = 1e9
    for i in range(shuffle_times):
        np.random.seed(i)
        shuffle_idx = np.random.permutation(unique_asym_ids.shape[0])
        np.random.seed()
        shuffled_asym_ids = unique_asym_ids[shuffle_idx]
        align = greedy_align(
            input_feats,
            per_asym_residue_index,
            shuffled_asym_ids,
            entity_2_asym_list,
            pred_cb_pos,
            pred_cb_mask,
            aligned_true_cb_poses,
            true_cb_masks,
        )

        merged_labels = merge_labels(
            input_feats,
            per_asym_residue_index,
            labels,
            align,
        )
        
        merged_ca_pose, merged_ca_mask = pseudo_beta_fn(merged_labels["aatype_per_chain"], merged_labels["all_atom_positions"], merged_labels["all_atom_mask"])

        rmsd = kabsch_rmsd(
            merged_ca_pose @ best_r + best_x,
            pred_cb_pos,
            (pred_cb_mask * merged_ca_mask).astype(bool),
        )

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_labels = merged_labels

    return best_labels


def multi_chain_perm_align_v2(final_atom_positions, input_feats, labels, shuffle_times=3):
    # print(input_feats["asym_id"])
    # print(input_feats["residue_index"])
    # print(input_feats["entity_id"])
    # print(input_feats["num_sym"])
    

    assert isinstance(labels, list)

    # ca_idx = atom_order["CA"]
    # pred_ca_pos = final_atom_positions[..., ca_idx, :].astype(np.float32)  # [bsz, nres, 3]
    # pred_ca_mask = input_feats["atom37_atom_exists"][..., ca_idx].astype(np.float32)  # [bsz, nres]
    # # import time
    # # time.sleep(10000)
    # true_ca_poses = [l["all_atom_positions"][..., ca_idx, :].astype(np.float32) for l in labels]  # list([nres, 3])
    # true_ca_masks = [l["all_atom_mask"][..., ca_idx].astype(np.float32) for l in labels]  # list([nres,])


    pred_cb_pos, pred_cb_mask = pseudo_beta_fn(input_feats["aatype"][0], final_atom_positions, input_feats["atom37_atom_exists"])
    pred_cb_pos, pred_cb_mask = pred_cb_pos.astype(np.float32), pred_cb_mask.astype(np.float32)
    true_cb_poses = []
    true_cb_masks = []
    for label in labels:
        true_cb_pose, true_cb_mask = pseudo_beta_fn(label["aatype_per_chain"], label["all_atom_positions"], label["all_atom_mask"])
        true_cb_poses.append(true_cb_pose.astype(np.float32))
        true_cb_masks.append(true_cb_mask.astype(np.float32))

    unique_asym_ids = np.unique(input_feats["asym_id"])

    per_asym_residue_index = {}
    for cur_asym_id in unique_asym_ids:
        asym_mask = (input_feats["asym_id"] == cur_asym_id).astype(bool)
        per_asym_residue_index[int(cur_asym_id)] = input_feats["residue_index"][asym_mask]

    anchor_gt_asym, anchor_pred_asym = get_anchor_candidates(
        input_feats, per_asym_residue_index, true_cb_masks
    )
    anchor_gt_idx = int(anchor_gt_asym) - 1


    unique_entity_ids = np.unique(input_feats["entity_id"])
    entity_2_asym_list = {}
    for cur_ent_id in unique_entity_ids:
        ent_mask = input_feats["entity_id"] == cur_ent_id
        cur_asym_id = np.unique(input_feats["asym_id"][ent_mask])
        entity_2_asym_list[int(cur_ent_id)] = cur_asym_id

    # find optimal transforms
    best_rmsd = 1e9
    best_r, best_x = None, None
    for cur_asym_id in anchor_pred_asym:
        asym_mask = (input_feats["asym_id"] == cur_asym_id).astype(bool)
        anchor_residue_idx = per_asym_residue_index[int(cur_asym_id)]
        anchor_true_pos = true_cb_poses[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_pos = pred_cb_pos[asym_mask]
        anchor_true_mask = true_cb_masks[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_mask = pred_cb_mask[asym_mask]
        r, x = get_optimal_transform(
            anchor_true_pos,
            anchor_pred_pos,
            (anchor_true_mask * anchor_pred_mask).astype(bool),
        )

        aligned_anchor_true_pos = anchor_true_pos @ r + x
        rmsd = compute_rmsd(aligned_anchor_true_pos, anchor_pred_pos, anchor_true_mask.astype(np.int32))
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_r = r
            best_x = x

    best_labels = None
    aligned_true_cb_poses = [cb @ best_r + best_x for cb in true_cb_poses]  # apply transforms

    # greedy align
    best_rmsd = 1e9
    for i in range(shuffle_times):
        np.random.seed(i)
        shuffle_idx = np.random.permutation(unique_asym_ids.shape[0])
        np.random.seed()
        shuffled_asym_ids = unique_asym_ids[shuffle_idx]
        align = greedy_align(
            input_feats,
            per_asym_residue_index,
            shuffled_asym_ids,
            entity_2_asym_list,
            pred_cb_pos,
            pred_cb_mask,
            aligned_true_cb_poses,
            true_cb_masks,
        )

        merged_labels = merge_labels(
            input_feats,
            per_asym_residue_index,
            labels,
            align,
        )
        
        merged_ca_pose, merged_ca_mask = pseudo_beta_fn(merged_labels["aatype_per_chain"], merged_labels["all_atom_positions"], merged_labels["all_atom_mask"])

        rmsd = kabsch_rmsd(
            merged_ca_pose @ best_r + best_x,
            pred_cb_pos,
            (pred_cb_mask * merged_ca_mask).astype(bool),
        )

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_labels = merged_labels

    # print("multi_chain_perm_align", best_rmsd)
    return best_labels


def multi_chain_perm_align_v1(final_atom_positions, input_feats, labels, shuffle_times=2):


    assert isinstance(labels, list)

    pred_ca_pos, pred_ca_mask = pseudo_beta_fn(input_feats["aatype"][0], final_atom_positions, input_feats["atom37_atom_exists"])
    pred_ca_pos, pred_ca_mask = pred_ca_pos.astype(np.float32), pred_ca_mask.astype(np.float32)
    true_ca_poses = []
    true_ca_masks = []
    for label in labels:
        true_ca_pose, true_ca_mask = pseudo_beta_fn(label["aatype_per_chain"], label["all_atom_positions"], label["all_atom_mask"])
        true_ca_poses.append(true_ca_pose.astype(np.float32))
        true_ca_masks.append(true_ca_mask.astype(np.float32))

    unique_asym_ids = np.unique(input_feats["asym_id"])

    per_asym_residue_index = {}
    for cur_asym_id in unique_asym_ids:
        asym_mask = (input_feats["asym_id"] == cur_asym_id).astype(bool)
        per_asym_residue_index[int(cur_asym_id)] = input_feats["residue_index"][asym_mask]
    
    anchor_gt_asym, anchor_pred_asym = get_anchor_candidates(
        input_feats, per_asym_residue_index, true_ca_masks
    )
    anchor_gt_idx = int(anchor_gt_asym) - 1

    best_rmsd = 1e9
    best_labels = None

    unique_entity_ids = np.unique(input_feats["entity_id"])
    entity_2_asym_list = {}
    for cur_ent_id in unique_entity_ids:
        ent_mask = input_feats["entity_id"] == cur_ent_id
        cur_asym_id = np.unique(input_feats["asym_id"][ent_mask])
        entity_2_asym_list[int(cur_ent_id)] = cur_asym_id


    for cur_asym_id in anchor_pred_asym:
        asym_mask = (input_feats["asym_id"] == cur_asym_id).astype(bool)
        anchor_residue_idx = per_asym_residue_index[int(cur_asym_id)]


        anchor_true_pos = true_ca_poses[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_pos = pred_ca_pos[asym_mask]
        anchor_true_mask = true_ca_masks[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_mask = pred_ca_mask[asym_mask]
        r, x = get_optimal_transform(
            anchor_true_pos,
            anchor_pred_pos,
            (anchor_true_mask * anchor_pred_mask).astype(bool),
        )
        
        

        aligned_true_ca_poses = [ca @ r + x for ca in true_ca_poses]  # apply transforms

        for i in range(shuffle_times):
            np.random.seed(i)
            shuffle_idx = np.random.permutation(unique_asym_ids.shape[0])
            np.random.seed()
            shuffled_asym_ids = unique_asym_ids[shuffle_idx]
            align = greedy_align(
                input_feats,
                per_asym_residue_index,
                shuffled_asym_ids,
                entity_2_asym_list,
                pred_ca_pos,
                pred_ca_mask,
                aligned_true_ca_poses,
                true_ca_masks,
            )
            merged_labels = merge_labels(
                input_feats,
                per_asym_residue_index,
                labels,
                align,
            )

            merged_ca_pose, merged_ca_mask = pseudo_beta_fn(merged_labels["aatype_per_chain"], merged_labels["all_atom_positions"], merged_labels["all_atom_mask"])

            rmsd = kabsch_rmsd(
                merged_ca_pose @ r + x,
                pred_ca_pos,
                (pred_ca_mask * merged_ca_mask).astype(bool),
            )

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_labels = merged_labels
                
    return best_labels


def get_anchor_candidates(input_feats, per_asym_residue_index, true_masks):
    def find_by_num_sym(min_num_sym):
        best_len = -1
        best_gt_asym = None
        asym_ids = np.unique(input_feats["asym_id"][input_feats["num_sym"] == min_num_sym])
        for cur_asym_id in asym_ids:
            assert cur_asym_id > 0
            cur_residue_index = per_asym_residue_index[int(cur_asym_id)]
            j = int(cur_asym_id - 1)
            cur_true_mask = true_masks[j][cur_residue_index]
            cur_len = cur_true_mask.sum()
            if cur_len > best_len:
                best_len = cur_len
                best_gt_asym = cur_asym_id
        return best_gt_asym, best_len

    sorted_num_sym = np.sort(input_feats["num_sym"][input_feats["num_sym"] > 0])
    best_gt_asym = None
    best_len = -1
    for cur_num_sym in sorted_num_sym:
        if cur_num_sym <= 0:
            continue
        cur_gt_sym, cur_len = find_by_num_sym(cur_num_sym)
        if cur_len > best_len:
            best_len = cur_len
            best_gt_asym = cur_gt_sym
        if best_len >= 3:
            break
    best_entity = input_feats["entity_id"][input_feats["asym_id"] == best_gt_asym][0]
    best_pred_asym = np.unique(input_feats["asym_id"][input_feats["entity_id"] == best_entity])
    return best_gt_asym, best_pred_asym


def get_optimal_transform(src_atoms, tgt_atoms, mask = None):
    assert src_atoms.shape == tgt_atoms.shape, (src_atoms.shape, tgt_atoms.shape)
    assert src_atoms.shape[-1] == 3
    if mask is not None:
        assert mask.dtype == bool
        assert mask.shape[-1] == src_atoms.shape[-2]
        if mask.sum() == 0:
            src_atoms = np.zeros((1, 3)).astype(np.float32)
            tgt_atoms = src_atoms
        else:
            src_atoms = src_atoms[mask, :]
            tgt_atoms = tgt_atoms[mask, :]
    src_center = src_atoms.mean(-2, keepdims=True)
    tgt_center = tgt_atoms.mean(-2, keepdims=True)

    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = P.transpose(-1, -2) @ Q
    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = V @ W
    return U


def greedy_align(
    input_feats,
    per_asym_residue_index,
    unique_asym_ids,
    entity_2_asym_list,
    pred_ca_pos,
    pred_ca_mask,
    true_ca_poses,
    true_ca_masks,
    ):
    used = [False for _ in range(len(true_ca_poses))]
    align = []
    for cur_asym_id in unique_asym_ids:
        # skip padding
        if cur_asym_id == 0:
            continue
        i = int(cur_asym_id - 1)
        asym_mask = input_feats["asym_id"] == cur_asym_id
        num_sym = input_feats["num_sym"][asym_mask][0]
        # don't need to align
        if (num_sym) == 1:
            align.append((i, i))
            assert used[i] == False
            used[i] = True
            continue
        cur_entity_ids = input_feats["entity_id"][asym_mask][0]
        best_rmsd = 1e10
        best_idx = None
        cur_asym_list = entity_2_asym_list[int(cur_entity_ids)]
        cur_residue_index = per_asym_residue_index[int(cur_asym_id)]
        cur_pred_pos = pred_ca_pos[asym_mask]
        cur_pred_mask = pred_ca_mask[asym_mask]
        for next_asym_id in cur_asym_list:
            if next_asym_id == 0:
                continue
            j = int(next_asym_id - 1)
            if not used[j]:  # posesible candidate
                cropped_pos = true_ca_poses[j][cur_residue_index]
                mask = true_ca_masks[j][cur_residue_index]
                rmsd = compute_rmsd(
                    cropped_pos, cur_pred_pos, (cur_pred_mask * mask).astype(bool)
                )
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_idx = j

        assert best_idx is not None
        used[best_idx] = True
        align.append((i, best_idx))

    return align


def compute_rmsd(true_atom_pos, pred_atom_pos, atom_mask = None, eps = 1e-6,):
    # shape check
    sq_diff = np.square(true_atom_pos - pred_atom_pos).sum(axis=1, keepdims=False)
    if len(sq_diff) == 1:
        return 1e8
    if atom_mask is not None:
        sq_diff = sq_diff[atom_mask]
    msd = np.mean(sq_diff)
    msd = np.nan_to_num(msd, nan=1e8)
    return np.sqrt(msd + eps)


def merge_labels(input_feats, per_asym_residue_index, labels, align):
    """
    input_feats:
    labels: list of label dicts, each with shape [nk, *]
    align: list of int, such as [2, None, 0, 1], each entry specify the corresponding label of the asym.
    """
    num_res = input_feats["msa_mask"].shape[-1]
    outs = {}
    for k, v in labels[0].items():
        if k in [
            "resolution",
        ]:
            continue
        cur_out = {}
        for i, j in align:
            label = labels[j][k]
            # to 1-based
            cur_residue_index = per_asym_residue_index[i + 1]
            cur_out[i] = label[cur_residue_index]
        cur_out = [x[1] for x in sorted(cur_out.items())]
        new_v = np.concatenate(cur_out, axis=0)
        merged_nres = new_v.shape[0]
        assert (
            merged_nres <= num_res
        ), f"bad merged num res: {merged_nres} > {num_res}. something is wrong."
        if merged_nres < num_res:  # must pad
            pad_dim = new_v.shape[1:]
            pad_v = np.zeros((num_res - merged_nres, *pad_dim)).astype(new_v.dtype)
            new_v = np.concatenate((new_v, pad_v), axis=0)
        outs[k] = new_v
    return outs


def kabsch_rmsd(true_atom_pos, pred_atom_pos, atom_mask,):
    r, x = get_optimal_transform(
        true_atom_pos,
        pred_atom_pos,
        atom_mask,
    )
    aligned_true_atom_pos = true_atom_pos @ r + x
    return compute_rmsd(aligned_true_atom_pos, pred_atom_pos, atom_mask)


def placeholder_data_genenrator(num_res, num_msa):
    

    data = {}
    data["atomtype_radius"] = np.zeros((3, )).astype(np.float16)
    data["restype_atom14_bond_lower_bound"] = np.zeros((21, 14, 14)).astype(np.float16)
    data["restype_atom14_bond_upper_bound"] = np.zeros((21, 14, 14)).astype(np.float16)
    data["use_clamped_fape"] = np.zeros((1,)).astype(np.float16)
    data["filter_by_solution"] = np.array(0).astype(np.float16)

    data["prot_name_index"] = np.zeros((1, )).astype(np.float16)

    data["seq_mask"] = np.zeros((num_res,)).astype(np.float16)
    data["aatype"] = np.zeros((num_res,)).astype(np.int32)
    data["residue_index"] = np.zeros((num_res,)).astype(np.int32)
    data["true_msa"] = np.zeros((num_msa, num_res)).astype(np.int32)
    data["bert_mask"] = np.zeros((num_msa, num_res)).astype(np.int32)
    


    data["pseudo_beta"] = np.zeros((num_res, 3)).astype(np.float16)
    data["pseudo_beta_mask"] = np.zeros((num_res,)).astype(np.float16)
    data["all_atom_mask"] = np.zeros((num_res, 37)).astype(np.float16)
    data["atom37_atom_exists"] = np.zeros((num_res, 37)).astype(np.float16)
    data["residx_atom14_to_atom37"] = np.zeros((num_res, 14)).astype(np.int32)
    data["atom14_atom_exists"] = np.zeros((num_res, 14)).astype(np.float16)
    data["backbone_affine_tensor"] = np.zeros((num_res, 7)).astype(np.float16)
    data["backbone_affine_mask"] = np.zeros((num_res,)).astype(np.float16)

    data["atom14_gt_positions"] = np.zeros((num_res, 14, 3)).astype(np.float16)
    data["atom14_alt_gt_positions"] = np.zeros((num_res, 14, 3)).astype(np.float16)
    data["atom14_atom_is_ambiguous"] = np.zeros((num_res, 14)).astype(np.float16)
    data["atom14_gt_exists"] = np.zeros((num_res, 14)).astype(np.float16)
    data["atom14_alt_gt_exists"] = np.zeros((num_res, 14)).astype(np.float16)

    data["all_atom_positions"] = np.zeros((num_res, 37, 3)).astype(np.float16)
    data["rigidgroups_gt_frames"] = np.zeros((num_res, 8, 12)).astype(np.float16)
    data["rigidgroups_gt_exists"] = np.zeros((num_res, 8)).astype(np.float16)
    data["rigidgroups_alt_gt_frames"] = np.zeros((num_res, 8, 12)).astype(np.float16)
    data["torsion_angles_sin_cos"] = np.zeros((num_res, 4, 2)).astype(np.float16)
    data["chi_mask"] = np.zeros((num_res, 4)).astype(np.float16)
    
    data["asym_mask"] = np.zeros((256, num_res)).astype(np.float16)

    gt_fake =  [data[key] for key in GT_KEYS]

    return gt_fake


def ground_truth_generator(input_data, atom37_position_pred, max_recycle):
    def extract_labels(d):
        all_labels = []
        for cur_chain_index in range(np.max(d["chain_index"]) + 1):
            all_label = {}
            for key in OUTPUT_LABEL_KEYS:
                all_label[key] = d[key][d["chain_index"] == cur_chain_index]
            all_labels.append(all_label)
        return all_labels
    all_labels = extract_labels(input_data)
    # for i, all_label in enumerate(all_labels):
    #     print("\n\n\n===============", i)
    #     for key, value in all_label.items():
    #         print(key, value.shape, value.dtype)

    input_data_single = {}
    for key, value in input_data.items():
        if len(value.shape) > 0 and value.shape[0] == input_data["msa_feat"].shape[0]:
            value = value[max_recycle-1]

        input_data_single[key] = value
    
    asym_id = input_data_single["asym_id"]
    asym_type = np.arange(1, np.max(asym_id) + 1) 
    asym_mask = (asym_id[None, :] == asym_type[:, None]).astype(np.float16) # [NC, NR]
    # print(asym_mask)
    asym_mask = np.pad(asym_mask, ((0, 256 - asym_mask.shape[0]), (0, 0))).astype(np.float16)
    # print(asym_mask[:4])
    # print(asym_mask.shape)
    input_data_single["asym_mask"] = asym_mask

    final_labels = multi_chain_perm_align_v1(atom37_position_pred, 
                                            input_data_single,
                                            all_labels,
                                            shuffle_times=4)
    # for key, value in final_labels.items():
    #     print(key, value.shape, value.dtype)
    
    final_labels_keys = list(final_labels.keys())

    # print(set(GT_KEYS) - set(final_labels_keys))
    # {'bert_mask', 'true_msa', 'restype_atom14_bond_lower_bound', 'restype_atom14_bond_upper_bound', 'filter_by_solution', 'use_clamped_fape', 'atomtype_radius', }

    # print(set(final_labels_keys) - set(GT_KEYS))
    #　{'chain_index', 'atom37_atom_exists_per_chain', 'aatype_per_chain'}

    # print(set(GT_KEYS).intersection(set(final_labels_keys)))
    # {'atom14_alt_gt_exists', 'pseudo_beta_mask', 'all_atom_mask', 'atom14_gt_exists', 'chi_mask', 'atom14_atom_is_ambiguous', 'backbone_affine_tensor', 'pseudo_beta', 'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'all_atom_positions', 'atom14_alt_gt_positions', 'atom14_gt_positions', 'backbone_affine_mask', 'residx_atom14_to_atom37', 'rigidgroups_alt_gt_frames', 'torsion_angles_sin_cos', 'atom14_atom_exists'}

    input_keys = ['restype_atom14_bond_lower_bound', 'restype_atom14_bond_upper_bound',
                  'filter_by_solution', 'use_clamped_fape', 'atomtype_radius'] + \
                 ['bert_mask', 'true_msa',"asym_mask"]

    gt_keys_useful = set(GT_KEYS).intersection(set(final_labels_keys))
    
    # print("\n\n\n\n final gt data====================")
    final_gt_data = []
    for key in GT_KEYS:
        if key in input_keys:
            value = input_data_single[key]
        else:
            value = final_labels[key]
        
        final_gt_data.append(value)
    #     print(key, value.shape, value.dtype)

    return final_gt_data


def ground_truth_generator_v2(input_data, atom37_position_pred):
    def extract_labels(d):
        all_labels = []
        for cur_chain_index in range(np.max(d["chain_index"]) + 1):
            all_label = {}
            for key in OUTPUT_LABEL_KEYS:
                all_label[key] = d[key][d["chain_index"] == cur_chain_index]
            all_labels.append(all_label)
        return all_labels
    all_labels = extract_labels(input_data)
    # for i, all_label in enumerate(all_labels):
    #     print("\n\n\n===============", i)
    #     for key, value in all_label.items():
    #         print(key, value.shape, value.dtype)

    input_data_single = input_data
    
    asym_id = input_data_single["asym_id"]
    asym_type = np.arange(1, np.max(asym_id) + 1) 
    asym_mask = (asym_id[None, :] == asym_type[:, None]).astype(np.float16) # [NC, NR]
    # print(asym_mask)
    asym_mask = np.pad(asym_mask, ((0, 256 - asym_mask.shape[0]), (0, 0))).astype(np.float16)
    # print(asym_mask[:4])
    # print(asym_mask.shape)
    input_data_single["asym_mask"] = asym_mask

    final_labels = multi_chain_perm_align_v1(atom37_position_pred, 
                                            input_data_single,
                                            all_labels,
                                            shuffle_times=4)
    # for key, value in final_labels.items():
    #     print(key, value.shape, value.dtype)
    
    final_labels_keys = list(final_labels.keys())

    # print(set(GT_KEYS) - set(final_labels_keys))
    # {'bert_mask', 'true_msa', 'restype_atom14_bond_lower_bound', 'restype_atom14_bond_upper_bound', 'filter_by_solution', 'use_clamped_fape', 'atomtype_radius', }

    # print(set(final_labels_keys) - set(GT_KEYS))
    #　{'chain_index', 'atom37_atom_exists_per_chain', 'aatype_per_chain'}

    # print(set(GT_KEYS).intersection(set(final_labels_keys)))
    # {'atom14_alt_gt_exists', 'pseudo_beta_mask', 'all_atom_mask', 'atom14_gt_exists', 'chi_mask', 'atom14_atom_is_ambiguous', 'backbone_affine_tensor', 'pseudo_beta', 'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'all_atom_positions', 'atom14_alt_gt_positions', 'atom14_gt_positions', 'backbone_affine_mask', 'residx_atom14_to_atom37', 'rigidgroups_alt_gt_frames', 'torsion_angles_sin_cos', 'atom14_atom_exists'}

    input_keys = ['restype_atom14_bond_lower_bound', 'restype_atom14_bond_upper_bound',
                  'filter_by_solution', 'use_clamped_fape', 'atomtype_radius'] + \
                 ['bert_mask', 'true_msa',"asym_mask"]

    gt_keys_useful = set(GT_KEYS).intersection(set(final_labels_keys))
    
    # print("\n\n\n\n final gt data====================")
    final_gt_data = []
    for key in GT_KEYS:
        if key in input_keys:
            value = input_data_single[key]
        else:
            value = final_labels[key]
        
        final_gt_data.append(value)
    #     print(key, value.shape, value.dtype)

    return final_gt_data

'''


==========================feature
aatype (384,) int64
residue_index (384,) int64
seq_length () int64
msa_chains (124, 1) float64
template_aatype (4, 384) int64
template_all_atom_mask (4, 384, 37) float32
template_all_atom_positions (4, 384, 37, 3) float32
all_atom_positions (384, 37, 3) float32
all_atom_mask (384, 37) float32
resolution () float32
asym_id (384,) float64
sym_id (384,) float64
entity_id (384,) float64
num_sym (384,) float64
assembly_num_chains (1,) int64
cluster_bias_mask (124,) float32
bert_mask (124, 384) float32
msa_mask (124, 384) float32
asym_len (5,) int64
num_recycling_iters () int64
use_clamped_fape () int64
is_distillation () int64
seq_mask (384,) float32
msa_row_mask (124,) float32
template_mask (4,) float32
template_pseudo_beta (4, 384, 3) float32
template_pseudo_beta_mask (4, 384) float32
template_torsion_angles_sin_cos (4, 384, 7, 2) float32
template_alt_torsion_angles_sin_cos (4, 384, 7, 2) float32
template_torsion_angles_mask (4, 384, 7) float32
residx_atom14_to_atom37 (384, 14) int64
residx_atom37_to_atom14 (384, 37) int64
atom14_atom_exists (384, 14) float32
atom37_atom_exists (384, 37) float32
target_feat (384, 22) float32
extra_msa (1152, 384) int64
extra_msa_mask (1152, 384) float32
extra_msa_row_mask (1152,) float32
true_msa (124, 384) int64
msa_feat (124, 384, 49) float32
extra_msa_has_deletion (1152, 384) float32
extra_msa_deletion_value (1152, 384) float32




==========================labels
aatype (216,) int64
all_atom_positions (216, 37, 3) float32
all_atom_mask (216, 37) float32
resolution (1,) float32
residx_atom14_to_atom37 (216, 14) int64
residx_atom37_to_atom14 (216, 37) int64
atom14_atom_exists (216, 14) float32
atom37_atom_exists (216, 37) float32
atom14_gt_exists (216, 14) float32
atom14_gt_positions (216, 14, 3) float32
atom14_alt_gt_positions (216, 14, 3) float32
atom14_alt_gt_exists (216, 14) float32
atom14_atom_is_ambiguous (216, 14) float32
rigidgroups_gt_frames (216, 8, 4, 4) float32
rigidgroups_gt_exists (216, 8) float32
rigidgroups_group_exists (216, 8) float32
rigidgroups_group_is_ambiguous (216, 8) float32
rigidgroups_alt_gt_frames (216, 8, 4, 4) float32
torsion_angles_sin_cos (216, 7, 2) float32
alt_torsion_angles_sin_cos (216, 7, 2) float32
torsion_angles_mask (216, 7) float32
pseudo_beta (216, 3) float32
pseudo_beta_mask (216,) float32
true_frame_tensor (216, 4, 4) float32
frame_mask (216,) float32
chi_angles_sin_cos (216, 4, 2) float32
chi_mask (216, 4) float32




==========================output
aatype (384,) int64
all_atom_positions (384, 37, 3) float32
all_atom_mask (384, 37) float32
residx_atom14_to_atom37 (384, 14) int64
residx_atom37_to_atom14 (384, 37) int64
atom14_atom_exists (384, 14) float32
atom37_atom_exists (384, 37) float32
atom14_gt_exists (384, 14) float32
atom14_gt_positions (384, 14, 3) float32
atom14_alt_gt_positions (384, 14, 3) float32
atom14_alt_gt_exists (384, 14) float32
atom14_atom_is_ambiguous (384, 14) float32
rigidgroups_gt_frames (384, 8, 4, 4) float32
rigidgroups_gt_exists (384, 8) float32
rigidgroups_group_exists (384, 8) float32
rigidgroups_group_is_ambiguous (384, 8) float32
rigidgroups_alt_gt_frames (384, 8, 4, 4) float32
torsion_angles_sin_cos (384, 7, 2) float32
alt_torsion_angles_sin_cos (384, 7, 2) float32
torsion_angles_mask (384, 7) float32
pseudo_beta (384, 3) float32
pseudo_beta_mask (384,) float32
true_frame_tensor (384, 4, 4) float32
frame_mask (384,) float32
chi_angles_sin_cos (384, 4, 2) float32
chi_mask (384, 4) float32


'''
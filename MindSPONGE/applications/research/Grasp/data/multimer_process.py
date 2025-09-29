import numpy as np

from mindsponge1.data.data_transform import make_atom14_masks, \
    atom37_to_frames, atom37_to_torsion_angles, pseudo_beta_fn, to_tensor_4x4
from mindsponge1.common.utils import make_atom14_positions
from mindsponge1.common.residue_constants import atom_order

from data.utils import numpy_seed


NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'


def make_pseudo_beta(protein, prefix=""):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ["", "template_"]
    (
        protein[prefix + "pseudo_beta"],
        protein[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        protein["template_aatype" if prefix else "aatype"],
        protein[prefix + "all_atom_positions"],
        protein["template_all_atom_mask" if prefix else "all_atom_mask"],
    )
    return protein


def get_pairwise_distances(coords):
    coord_diff = np.expand_dims(coords, axis=-2) - np.expand_dims(coords, axis=-3)
    return np.sqrt(np.sum(coord_diff**2, axis=-1))


def get_interface_candidates_v2(ca_distances, asym_id, pair_mask, ca_ca_threshold):

    in_same_asym = asym_id[..., None] == asym_id[..., None, :]
    # set distance in the same entity to zero
    ca_distances = ca_distances * (1.0 - in_same_asym.astype(np.float32)) * pair_mask
    interface_candidates = np.array(np.nonzero((ca_distances > 0) & (ca_distances < ca_ca_threshold))).transpose()
    # print("interface_candidates", interface_candidates)
    return interface_candidates


def get_interface_candidates(ca_distances, asym_id, pair_mask, ca_ca_threshold):

    in_same_asym = asym_id[..., None] == asym_id[..., None, :]
    # set distance in the same entity to zero
    ca_distances = ca_distances * (1.0 - in_same_asym.astype(np.float32)) * pair_mask
    cnt_interfaces = np.sum((ca_distances > 0) & (ca_distances < ca_ca_threshold), axis=-1)
    interface_candidates = np.nonzero(cnt_interfaces)[0]
    return interface_candidates


def get_crop_sizes_each_chain(asym_len, crop_size, random_seed=None, use_multinomial=False,):
    """get crop sizes for contiguous crop"""
    if not use_multinomial:
        with numpy_seed(random_seed, key="multimer_contiguous_perm"):
            shuffle_idx = np.random.permutation(len(asym_len))
        num_left = np.array(asym_len.sum())
        num_budget = np.array(crop_size)
        crop_sizes = [0 for _ in asym_len]
        for j, idx in enumerate(shuffle_idx):
            this_len = asym_len[idx]
            num_left -= this_len
            # num res at most we can keep in this ent
            max_size = min(num_budget, this_len)
            # num res at least we shall keep in this ent
            min_size = min(this_len, max(0, num_budget - num_left))
            with numpy_seed(random_seed, j, key="multimer_contiguous_crop_size"):
                this_crop_size = int(np.random.randint(low=int(min_size), high=int(max_size) + 1))
            num_budget -= this_crop_size
            crop_sizes[idx] = this_crop_size
        crop_sizes = np.array(crop_sizes)
    else:  # use multinomial
        # TODO: better multimer
        entity_probs = asym_len / np.sum(asym_len)
        crop_sizes = np.random.multinomial(crop_size, pvals=entity_probs)
        crop_sizes = np.min(crop_sizes, asym_len)
    return crop_sizes


def get_contiguous_crop_idx(protein, crop_size, random_seed=None, use_multinomial=False):

    num_res = protein["aatype"].shape[0]
    if num_res <= crop_size:
        return np.arange(num_res)

    assert "asym_len" in protein
    asym_len = protein["asym_len"]

    crop_sizes = get_crop_sizes_each_chain(asym_len, crop_size, random_seed, use_multinomial)
    crop_idxs = []
    asym_offset = np.array(0, dtype=np.int64)
    with numpy_seed(random_seed, key="multimer_contiguous_crop_start_idx"):
        for l, csz in zip(asym_len, crop_sizes):
            this_start = np.random.randint(0, int(l - csz) + 1)
            crop_idxs.append(np.arange(asym_offset + this_start, asym_offset + this_start + csz))
            asym_offset += l

    return np.concatenate(crop_idxs)



def random_num_with_fix_total(maxValue, num):
    # generate 'num - 1' uniformlly distributed integers to split 
    # the whole interval [0, maxValue] into 'num' small intervals
    a = list(np.random.uniform(0, maxValue, size=(num-1)).astype(np.int32))
    a.append(0)
    a.append(maxValue)
    a = sorted(a)
    b = [ a[i]-a[i-1] for i in range(1, len(a)) ]
    # print(b)
    return b


def get_chain_index(nk_all, res_index_all):
    for i, seq_length in enumerate(nk_all):
        if res_index_all < seq_length:
            return i, res_index_all
        else:
            res_index_all -= seq_length


def contact_biased_continous_cropping(chain_lengths, N_res, selected_contacts):

    minimum_crop_size = 16
    nk_all = []
    random_crop_masks = []
    all_seq_length = 0
    for seq_len in chain_lengths:
        nk_all.append(seq_len)
        random_crop_masks.append(np.zeros(seq_len,))
        all_seq_length += seq_len

    if all_seq_length <= N_res:
        random_crop_masks = [np.ones(mask.shape) for mask in random_crop_masks]
        return np.concatenate(random_crop_masks)

    num_contacts = selected_contacts.shape[0]
    used_contact = []
    for i in range(num_contacts * 2):

        # get res info in chain
        res_index_all = selected_contacts[i // 2, i % 2]
        chain_index, res_index_in_chain = get_chain_index(nk_all, res_index_all)

        # get crop size
        n_added = int(sum([mask.sum() for mask in random_crop_masks]))
        n_left = N_res - n_added
        if n_left < minimum_crop_size: 
            break
        randoms = random_num_with_fix_total(n_left - minimum_crop_size, num_contacts * 2 - i)
        cur_crop_size = min(randoms[0] + minimum_crop_size, nk_all[chain_index])

        # get crop start & stop from contact infos
        random_start = min(max(res_index_in_chain - cur_crop_size + minimum_crop_size // 2, 0), nk_all[chain_index] - cur_crop_size)
        random_stop = min(max(res_index_in_chain - minimum_crop_size // 2 + 1, 0), nk_all[chain_index] - cur_crop_size)
        # print(random_start, random_stop)
        crop_start = int(np.random.uniform(random_start, random_stop))
        # print(nk_all[chain_index], res_index_in_chain, crop_start, cur_crop_size)
        keep = [i for i in range(crop_start, crop_start + cur_crop_size)]
        # print(res_index_all, chain_index, res_index_in_chain, crop_start, len(keep))
        random_crop_masks[chain_index][keep] = 1

        if i % 2 == 1:
            used_contact.append(i//2)
    # print("used_contact")
    # print("len(used_contact)", len(used_contact))
    return np.concatenate(random_crop_masks)


def get_chain_lengths(protein):

    last_asym_id = -1
    chain_length = 0
    chain_lengths  = []
    for asym_id in protein["asym_id"]:
        if asym_id != last_asym_id:
            last_asym_id = asym_id
            chain_length = 1
            chain_lengths.append(1)
        else:
            chain_length += 1
            chain_lengths[-1] = chain_length
    asym_id = protein["asym_id"]
    chain_lengths2 = (asym_id[None, :] == np.array(sorted(list(set(list(asym_id)))))[:, None]).sum(-1)

    if np.sum(np.abs(chain_lengths - chain_lengths2)) > 0:
        print("error !!!")
        print(list(chain_lengths))
        print(list(chain_lengths2))
    return chain_lengths


def get_spatial_crop_idx_v2(protein, crop_size, random_seed, ca_ca_threshold, inf=3e4):

    ca_idx = atom_order["CA"]
    ca_coords = protein["all_atom_positions"][..., ca_idx, :]
    ca_mask = protein["all_atom_mask"][..., ca_idx].astype(np.bool)
    # if there are not enough atoms to construct interface, use contiguous crop
    if (ca_mask.sum(axis=-1) <= 1).all():
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    ca_distances = get_pairwise_distances(ca_coords)

    interface_candidates = get_interface_candidates_v2(ca_distances,
                                                    protein["asym_id"],
                                                    pair_mask,
                                                    ca_ca_threshold)

    if interface_candidates.any():
        with numpy_seed(random_seed, key="multimer_spatial_crop"):
            np.random.shuffle(interface_candidates)
    else:
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    chain_lengths  = get_chain_lengths(protein)

    random_masks_all = contact_biased_continous_cropping(chain_lengths, crop_size, interface_candidates)
    ret = list(np.where(np.array(random_masks_all) > 0)[0])
    return ret


def get_spatial_crop_idx(protein, crop_size, random_seed, ca_ca_threshold, inf=3e4):

    ca_idx = atom_order["CA"]
    ca_coords = protein["all_atom_positions"][..., ca_idx, :]
    ca_mask = protein["all_atom_mask"][..., ca_idx].astype(np.bool)
    # if there are not enough atoms to construct interface, use contiguous crop
    if (ca_mask.sum(axis=-1) <= 1).all():
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    pair_mask = ca_mask[..., None] * ca_mask[..., None, :]
    ca_distances = get_pairwise_distances(ca_coords)

    interface_candidates = get_interface_candidates(ca_distances,
                                                    protein["asym_id"],
                                                    pair_mask,
                                                    ca_ca_threshold)

    if interface_candidates.any():
        with numpy_seed(random_seed, key="multimer_spatial_crop"):
            target_res = int(np.random.choice(interface_candidates))
    else:
        return get_contiguous_crop_idx(protein, crop_size, random_seed)

    to_target_distances = ca_distances[target_res]
    # set inf to non-position residues
    to_target_distances[~ca_mask] = inf
    break_tie = (np.arange(0, to_target_distances.shape[-1], dtype=np.float32) * 1e-3)
    to_target_distances += break_tie
    ret = np.argsort(to_target_distances)[:crop_size]
    ret.sort()
    return ret


def apply_crop_idx(protein, shape_schema, crop_idx):
    cropped_protein = {}
    for k, v in protein.items():
        if k not in shape_schema:  # skip items with unknown shape schema
            continue
        for i, dim_size in enumerate(shape_schema[k]):
            if dim_size == NUM_RES:
                v = np.take(v, crop_idx, axis=i)
        cropped_protein[k] = v
    return cropped_protein


def select_feat(protein, feature_list):
    feature_list.pop("msa")
    feature_list.pop("msa_chains")
    feature_list.pop("deletion_matrix")
    feature_list.pop("num_alignments")
    feature_list.pop("hhblits_profile")
    return {k: v for k, v in protein.items() if k in feature_list}


def make_fixed_size(protein, shape_schema, msa_cluster_size, extra_msa_size, num_res=0, num_templates=0,):
    """Guess at the MSA and sequence dimension to make fixed size."""
    def get_pad_size(cur_size, multiplier=4):
        return  max(multiplier, 
                ((cur_size + multiplier - 1) // multiplier) * multiplier
            )
    if num_res is not None:
        input_num_res = (
            protein["aatype"].shape[0]
            if "aatype" in protein
            else protein["msa_mask"].shape[1]
        )
        if input_num_res != num_res:
            num_res = get_pad_size(input_num_res, 4)
    # if "extra_msa_mask" in protein:
    #     input_extra_msa_size = protein["extra_msa_mask"].shape[0]
    #     if input_extra_msa_size != extra_msa_size:
    #         print(input_extra_msa_size, extra_msa_size)
    #         # import time
    #         # time.sleep(100)
    #         extra_msa_size = get_pad_size(input_extra_msa_size, 8)
    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]

        padding = []
        for i, p in enumerate(pad_size):
            if (p - v.shape[i]) >= 0:
                padding.append((0, p - v.shape[i]))
            else:
                padding.append((0, 0))
                v = v.take(np.arange(v.shape[i]+(p - v.shape[i])), axis=i)
        if padding:
            protein[k] = np.pad(v, padding)
            protein[k] = protein[k].reshape(pad_size)

    return protein


def pad_then_stack(values):
    if len(values[0].shape) >= 1:
        size = max(v.shape[0] for v in values)
        new_values = []
        for v in values:
            if v.shape[0] < size:
                res = np.zeros((size, *v.shape[1:]), dtype=values[0].dtype)
                res[:v.shape[0], ...] = v
            else:
                res = v
            new_values.append(res)
    else:
        new_values = values
    return np.stack(new_values, axis=0)


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = pad_then_stack([dict_i[feat] for dict_i in ensembles])
    return ensembled_dict

def get_train_labels_old(aatype, atom37_positions, atom37_mask, chain_index):
    """get train labels"""

    seq_len = len(aatype)
    # get ground truth of atom14
    label_features = {'aatype': aatype,
                      'all_atom_positions': atom37_positions,
                      'all_atom_mask': atom37_mask}

    atom14_features = make_atom14_positions(aatype, atom37_mask, atom37_positions)
    atom14_keys = ["atom14_atom_exists", "atom14_gt_exists", "atom14_gt_positions", "residx_atom14_to_atom37",
                    "residx_atom37_to_atom14", "atom37_atom_exists", "atom14_alt_gt_positions",
                    "atom14_alt_gt_exists", "atom14_atom_is_ambiguous"]
    for index, array in enumerate(atom14_features):
        label_features[atom14_keys[index]] = array

    # get ground truth of rigid groups
    rigidgroups_label_feature = atom37_to_frames(aatype, atom37_positions, atom37_mask, is_affine=True)
    label_features.update(rigidgroups_label_feature)

    # get ground truth of angle
    angle_label_feature = atom37_to_torsion_angles(aatype.reshape((1, -1)),
                                                    atom37_positions.reshape((1, seq_len, 37, 3)),
                                                    atom37_mask.reshape((1, seq_len, 37)), True)
    label_features.update(angle_label_feature)

    # get pseudo_beta, pseudo_beta_mask
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(aatype, atom37_positions, atom37_mask)
    label_features["pseudo_beta"] = pseudo_beta
    label_features["pseudo_beta_mask"] = pseudo_beta_mask
    label_features["chi_mask"] = label_features.get("torsion_angles_mask")[:, 3:]
    label_features['torsion_angles_sin_cos'] = label_features.get('torsion_angles_sin_cos')[:, 3:, :]
    label_features['backbone_affine_mask'] = pseudo_beta_mask
    label_features["chain_index"] = (np.ones(pseudo_beta_mask.shape) * chain_index).astype(np.int32)
    label_features["aatype_per_chain"] = label_features["aatype"]
    label_features["atom37_atom_exists_per_chain"] = label_features["atom37_atom_exists"]
    # print(np.allclose(label_features["atom37_atom_exists"], label_features["all_atom_mask"]))
    # print(label_features["chain_index"])

    return label_features

def process_single_label(label, chain_index):
    assert "aatype_pdb" in label
    assert "all_atom_positions" in label
    assert "all_atom_mask" in label

    label_features = get_train_labels_old(label["aatype_pdb"], label["all_atom_positions"], label["all_atom_mask"], chain_index)

    return label_features

def process_labels(labels_list):
    return [process_single_label(l, chain_index) for chain_index, l in enumerate(labels_list)]

def label_transform_fn(label):

    aatype = label["aatype"]
    atom14_atom_exists, residx_atom14_to_atom37, residx_atom37_to_atom14, \
        atom37_atom_exists = make_atom14_masks(aatype)
    label["residx_atom14_to_atom37"] = residx_atom14_to_atom37
    label["residx_atom37_to_atom14"] = residx_atom37_to_atom14
    label["atom14_atom_exists"] = atom14_atom_exists
    label["atom37_atom_exists"] = atom37_atom_exists

    all_atom_mask = label["all_atom_mask"]
    all_atom_positions = label["all_atom_positions"]
    atom14_atom_exists, atom14_gt_exists, atom14_gt_positions, _, _, _, \
        atom14_alt_gt_positions, atom14_alt_gt_exists, atom14_atom_is_ambiguous = \
        make_atom14_positions(aatype, all_atom_mask, all_atom_positions)
    label["atom14_atom_exists"] = atom14_atom_exists
    label["atom14_gt_exists"] = atom14_gt_exists
    label["atom14_gt_positions"] = atom14_gt_positions
    label["atom14_alt_gt_positions"] = atom14_alt_gt_positions
    label["atom14_alt_gt_exists"] = atom14_alt_gt_exists
    label["atom14_atom_is_ambiguous"] = atom14_atom_is_ambiguous

    label_f = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)
    label["mrigidgroups_gt_frames"] = label_f["rigidgroups_gt_frames"]
    label["rigidgroups_gt_exists"] = label_f["rigidgroups_gt_exists"]
    label["rigidgroups_group_exists"] = label_f["rigidgroups_group_exists"]
    label["rigidgroups_group_is_ambiguous"] = label_f["rigidgroups_group_is_ambiguous"]
    label["mrigidgroups_alt_gt_frames"] = label_f["rigidgroups_alt_gt_frames"]

    label["rigidgroups_gt_frames"] = to_tensor_4x4(label["mrigidgroups_gt_frames"])
    label["rigidgroups_alt_gt_frames"] = to_tensor_4x4(label["mrigidgroups_alt_gt_frames"])

    angle_label_feature = atom37_to_torsion_angles(aatype.reshape((1, -1)), all_atom_positions.reshape((1, -1, 37, 3)), all_atom_mask.reshape((1, -1, 37)), alt_torsions=True)
    label["torsion_angles_sin_cos"] = angle_label_feature["torsion_angles_sin_cos"]
    label["alt_torsion_angles_sin_cos"] = angle_label_feature["alt_torsion_angles_sin_cos"]
    label["torsion_angles_mask"] = angle_label_feature["torsion_angles_mask"]

    label = make_pseudo_beta(label, "")

    label["true_frame_tensor"] = label["rigidgroups_gt_frames"][..., 0, :, :]
    label["frame_mask"] = label["rigidgroups_gt_exists"][..., 0]

    dtype = label["all_atom_mask"].dtype
    label["chi_angles_sin_cos"] = (label["torsion_angles_sin_cos"][..., 3:, :]).astype(dtype)
    label["chi_mask"] = label["torsion_angles_mask"][..., 3:].astype(dtype)

    return label
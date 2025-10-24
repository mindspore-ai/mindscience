# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
"sumproduct pair"

from .constants import T_MATMUL, T_MUL, T_OUT, MUST_KM, MUST_MK, MUST_ALL


def _apply_permute(perm_ids, ary):
    """
    Apply a permutation to an array
    """
    if perm_ids:
        return [ary[i] for i in perm_ids]
    return ary


def prod_lst(lst):
    """
    Calculate the product of all elements in the list.
    """
    p = 1
    for k in lst:
        p *= k

    return p


def _prod_shape(a_shape, raw=True):
    """
    Calculate the product of the elements in a_shape.
    If raw is True, a_shape is treated as a list; otherwise, it is treated as a dictionary.
    """
    if raw:
        p = prod_lst(a_shape)
    else:
        p = prod_lst(a_shape.values())

    return p


def _pop_s1_dims(a_shape, a_dims_to_sum):
    """
    Set the dimensions in a_dims_to_sum to 1 in a_shape, then remove all dimensions with value 1.
    Calculate the volume of the modified shape
    """
    for idx in a_dims_to_sum:
        a_shape[idx] = 1

    vol = _prod_shape(a_shape, raw=False)

    for k in list(a_shape.keys()):
        if a_shape[k] == 1:
            a_shape.pop(k)

    return vol


def _is_ab_pat(a, b):
    """
    Check if the pattern (a, b) matches one of the predefined conditions.
    """
    flag = False
    if a > 64000 and a > b > 128:
        flag = True

    if a < 16 and b >= 1024:
        flag = True

    return flag


def _judge_mk_helper(m, k):
    """
    judge if transpose_a can be used; False for some special shape
    """
    if _is_ab_pat(k, m):
        return MUST_KM

    if _is_ab_pat(m, k):
        return MUST_MK

    return MUST_ALL


def _judge_cacl_by_shape(k, m, n, a_shape, b_shape):
    """judge calculate type and mk type

    Args:
        B (set): labels of batch of bmm
        K (set): labels of K-axis of bmm
        M (set): labels of M-axis of bmm's first tensor
        N (set): labels of N-axis of bmm's second tensor
        a_shape (dict[str, int]): label, size pairs of first tensor
        b_shape (dict[str, int]): label, size pairs of second tensor
    """
    k_size = _prod_shape([a_shape[d] for d in k])
    m_size = _prod_shape([a_shape[d] for d in m])
    n_size = _prod_shape([b_shape[d] for d in n])

    r1 = _judge_mk_helper(m_size, k_size)
    r2 = _judge_mk_helper(n_size, k_size)

    use_mul = False
    if k_size == 1:
        use_mul = True

    if m_size == 1 and n_size == 1:
        use_mul = True

    return r1, r2, use_mul


def sumproduct_pair_info(a_shape, b_shape, a_labels_to_sum, b_labels_to_sum, sum_labels):
    """This function calculates the necessary information for performing a sum-product operation on two tensors.
    It determines the batch dimensions, the dimensions to be summed, and the dimensions for matrix multiplication.
    It also updates the information for each tensor and returns the new shape and operation details.

    Args:
        a_shape (dict[str, int]): Shape of the first tensor.
        b_shape (dict[str, int]): Shape of the second tensor.
        a_labels_to_sum (list[str, ...]): Labels to sum for the first tensor.
        b_labels_to_sum (list[str, ...]): Labels to sum for the second tensor.
        sum_labels (list[str, ...]): Labels to be summed in the operation.

    Returns:
        new_shape (dict): The new shape after the operation.
        a_info (dict): Information for the first tensor.
        b_info (dict): Information for the second tensor.
        out_info (dict): Output information.
    """
    a_weight = _pop_s1_dims(a_shape, a_labels_to_sum)
    b_weight = _pop_s1_dims(b_shape, b_labels_to_sum)

    # Determine batch dimensions
    a_keys, b_keys = set(a_shape.keys()), set(b_shape.keys())
    sum_keys = set(sum_labels)
    batch_dims = (a_keys & b_keys) - sum_keys

    # Determine M and N dimensions
    a_dims = a_keys - batch_dims - sum_keys
    b_dims = b_keys - batch_dims - sum_keys

    a_bms, b_bms, use_mul = _judge_cacl_by_shape(sum_keys, a_dims, b_dims, a_shape, b_shape)

    t_type = T_MUL if use_mul else T_MATMUL
    a_info = {"CACL": t_type, "SUMS": a_labels_to_sum, "WEIGHT": a_weight}
    b_info = {"CACL": t_type, "SUMS": b_labels_to_sum, "WEIGHT": b_weight}

    all_shape = a_shape | b_shape
    new_shape = {k: all_shape[k] for k in all_shape if k not in sum_labels}

    if use_mul:
        return new_shape, a_info, b_info, {"FROM": T_MUL, "SUMS": sum_labels}

    b, k = "".join(batch_dims), "".join(sum_labels)
    m, n = "".join(a_dims), "".join(b_dims)

    a_info.update({"B": b, "M": m, "K": k, "LEFT": True, "BMM_MUST_SEQ": a_bms})
    b_info.update({"B": b, "M": n, "K": k, "LEFT": False, "BMM_MUST_SEQ": b_bms})
    out_info = {"B": b, "M": m, "N": n, "FROM": T_MATMUL}

    return new_shape, a_info, b_info, out_info


def out_cacl_info(a_shape, rhs):
    """This function calculates the output information for a tensor after applying a reduction operation.
    It updates the shape of the tensor and calculates the volume of the reduced dimensions.

    Args:
        a_shape (dict[str, int]): Shape of the tensor.
        rhs (str): Labels to keep in the output.

    Returns:
        info (dict): Output information including the operation type, output labels, weight, and labels to sum.
    """
    new_a_shape = {k: v for k, v in a_shape.items() if k in rhs}
    vol = _prod_shape(new_a_shape, raw=False)
    sum_labels = [k for k in a_shape if k not in rhs]
    info = {"CACL": T_OUT, "OUT": rhs, "WEIGHT": vol, "SUMS": sum_labels}
    return info


def _identity_perm(perm):
    """This function checks if a permutation is the identity permutation (i.e., no permutation).
    If it is the identity permutation, it returns None; otherwise, it returns the permutation as a tuple.

    Args:
        perm (list): Permutation to check.

    Returns:
        tuple or None: The permutation as a tuple or None if it is the identity permutation.
    """
    flag = True
    for i in range(len(perm) - 1):
        if perm[i] >= perm[i+1]:
            flag = False

    if flag:
        return None
    return tuple(perm)


def _rearrange_tensor_to_mul_helper(order_labels, a_shape):
    """This function rearranges the labels of a tensor to match a specified order.
    It also returns the permutation and the new shape dictionary.

    Args:
        order_labels (list): Desired order of labels.
        a_shape (list of tuples): Shape of the tensor as a list of (label, size) pairs.

    Returns:
        a_permute (tuple or None): Permutation to apply or None if no permutation is needed.
        permed_labels (list): Labels after permutation.
        a_shape_dict (dict): Shape dictionary of the tensor.
    """
    a_labels = [label for label, _ in a_shape]
    a_shape_dict = {label: sp for label, sp in a_shape}
    a_permute = []

    for label in order_labels:
        if label in a_labels:
            a_permute.append(a_labels.index(label))

    permed_labels = _apply_permute(a_permute, a_labels)
    a_permute = _identity_perm(a_permute)
    return a_permute, permed_labels, a_shape_dict


def _process_labels(out_labels, a_labels):
    """This function processes the labels to find missing labels and generates permutation indices.
    It returns the number of missing labels and the permutation indices.

    Args:
        out_labels (list): Desired output labels.
        a_labels (list): Current labels of the tensor.

    Returns:
        tuple: Number of missing labels and permutation indices.
    """
    missing = []
    for label in out_labels:
        if label not in a_labels:
            missing.append(label)

    perm_ids = _get_s2t_perm_indices(a_labels + missing, out_labels)
    return (len(missing), tuple(perm_ids))


def rearrange_tensor_to_mul(order_labels, a_shape, b_shape):
    """
    Generate permute information for tensors a and b based on the order of labels.

    Args:
        order_labels (str): Ordered labels (e.g., "ijk").
        a_shape, b_shape: list[tuple[str, int], ...]
        a_shape: List of tuples (label, size) for tensor a.
        b_shape: List of tuples (label, size) for tensor b.
        e.g., [('i', 128), ('j', 64), ('k', 256)]

    Returns:
        tuple: (a_permute, b_permute) where each is a list of indices for permute.
    """
    a_perm, a_labels, a_shape_dict = _rearrange_tensor_to_mul_helper(order_labels, a_shape)
    b_perm, b_labels, b_shape_dict = _rearrange_tensor_to_mul_helper(order_labels, b_shape)

    out_labels = [label for label in order_labels if label in set(a_labels + b_labels)]

    a_sp1_info = _process_labels(out_labels, a_labels)
    b_sp1_info = _process_labels(out_labels, b_labels)

    s_dict = a_shape_dict | b_shape_dict
    new_shape = [(label, s_dict[label]) for label in out_labels]

    return a_perm, b_perm, (a_sp1_info, b_sp1_info), new_shape


def _sort_bmm_labels(order_labels, a_info, left, labels_g1):
    """Sorts labels into B, M, K groups based on a_info and order_labels.

    Args:
        order_labels (str): labels to be sorted.
        a_info (dict[str, int]): Dictionary containing B, M, K labels.
        left (bool): Determines the order of the output.
        labels_g1 (str): String of labels to be sorted.

    Returns:
        tuple: Two strings representing the sorted labels in BKM and BKM or BKM and BMK order,
        and a tuple of individual B, M, K labels.
    """
    rb, rm, rk = "", "", ""

    for label in order_labels:
        if label in labels_g1:
            if label in a_info["B"]:
                rb += label
            elif label in a_info["M"]:
                rm += label
            elif label in a_info["K"]:
                rk += label

    bmk = rb + rm + rk
    bkm = rb + rk + rm
    if left:
        return bmk, bkm, (rb, rm, rk)
    return bkm, bmk, (rb, rm, rk)


def _group_bmm_indices(cur_labels: str, group_labels: tuple[str, ...]):
    """
    Groups indices of labels into B, M, K categories based on group_labels.

    Args:
        cur_labels (str): Current labels to be grouped.
        group_labels (tuple): Tuple of B, M, K labels.

    Returns:
        tuple: Tuples of indices for B, M, K labels.
    """
    b_idx, m_idx, k_idx = [], [], []

    rb, rm, rk = group_labels
    for i, label in enumerate(cur_labels):
        if label in rb:
            b_idx.append(i)
        elif label in rm:
            m_idx.append(i)
        elif label in rk:
            k_idx.append(i)
        else:
            raise ValueError("Error")

    return tuple(b_idx), tuple(m_idx), tuple(k_idx)


def _get_s2t_perm_indices(s1, s2):
    """
    Gets the permutation indices to transform s1 to s2.

    Args:
        s1 (str): Source string.
        s2 (str): Target string.

    Returns:
        list: List of indices to permute s1 to match s2.
    """
    indices = []
    for char in s2:
        index = s1.index(char)
        indices.append(index)

    return indices


def _judge_transpose_condition(bmm_must_seq, is_left):
    """
    Determines if a transpose is needed based on bmm_must_seq and if it is left operand.

    Args:
        bmm_must_seq (str): Sequence that must be followed.
        is_left (bool): Indicates if the operation is on the left side.

    Returns:
        bool: Whether a transpose is needed.
    """
    if bmm_must_seq == MUST_MK:
        transpose_a = not is_left
    elif bmm_must_seq == MUST_KM:
        transpose_a = is_left
    else:
        transpose_a = None

    return transpose_a


def _rearrange_tensor_to_bmm_helper(order_labels, a_shape, a_info, left):
    """
    Helper function to rearrange tensor labels and determine if a transpose is needed.

    Args:
        order_labels (list): List of labels to be sorted.
        a_shape (list): List of tuples (label, size) for tensor; example: [('i', 128), ('j', 64), ('k', 256)].
        a_info (dict): Dictionary containing B, M, K labels; B:["z"], M:["ij"] K:["k"]
        left (bool): Determines the order of the output.

    Returns:
        tuple: Permutation indices, transpose information, and new shapes for B and M.
    """
    a_labels = "".join([label for label, _ in a_shape])
    labels1, labels2, group_labels = _sort_bmm_labels(order_labels, a_info, left, a_labels)

    transpose_a = _judge_transpose_condition(a_info["BMM_MUST_SEQ"], left)
    if transpose_a is None:
        if a_labels == labels1:
            transpose_a = False
        elif a_labels == labels2:
            transpose_a = True
        else:
            transpose_a = False

    target_labels = labels2 if transpose_a else labels1
    perm_ids = _get_s2t_perm_indices(a_labels, target_labels)

    a_shape_dict = {label: sp for label, sp in a_shape}
    new_shape_b = tuple((label, a_shape_dict[label]) for label in group_labels[0])
    new_shape_m = tuple((label, a_shape_dict[label]) for label in group_labels[1])

    group_shape_idxs = _group_bmm_indices(target_labels, group_labels)
    new_perm_ids = _identity_perm(perm_ids)

    return new_perm_ids, (transpose_a, group_shape_idxs), (new_shape_b, new_shape_m)


def rearrange_tensor_to_bmm(order_labels, a_shape, a_info, b_shape, b_info):
    """
    Rearranges tensors to prepare for batch matrix multiplication (BMM).

    Args:
        order_labels (str): Labels to be sorted.
        a_shape (list[tuple[str, int], ...]): List of tuples (label, size) for tensor a.
        a_info (dict[str, int]): Dictionary containing B, M, K labels for tensor a.
        b_shape (list[tuple[str, int], ...]): List of tuples (label, size) for tensor b.
        b_info (dict[str, int]): Dictionary containing B, M, K labels for tensor b.

    Returns:
        tuple: Permutation indices for a and b, BMM information, and the new shape.
    """
    a_perm, a_bmm_info, a_group_shapes = _rearrange_tensor_to_bmm_helper(order_labels, a_shape, a_info, True)
    b_perm, b_bmm_info, b_group_shapes = _rearrange_tensor_to_bmm_helper(order_labels, b_shape, b_info, False)
    new_shape = a_group_shapes[0] + a_group_shapes[1] + b_group_shapes[1]
    transpose_a, a_shape_idxs = a_bmm_info
    transpose_b, b_shape_idxs = b_bmm_info

    a_b, a_m, a_k = a_shape_idxs
    b_m = b_shape_idxs[1]

    is_batch = len(a_b) > 0
    bmm_info = (is_batch, transpose_a, transpose_b, a_b, a_m, b_m, a_k)

    return a_perm, b_perm, bmm_info, new_shape


def rearrange_tensor_to_out(a_shape, out_labels):
    """
    Rearranges tensor labels to match the output labels.

    Args:
        a_shape (list[tuple[str, int], ...]): List of tuples (label, size) for tensor a.
        out_labels (str): Desired output labels.

    Returns:
        list: Permutation indices to rearrange the tensor to match the output labels.
    """
    in_labels = "".join([k for k, _ in a_shape])
    perm = _get_s2t_perm_indices(in_labels, out_labels)
    return _identity_perm(perm)

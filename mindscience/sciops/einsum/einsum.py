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
"einsum main file"

import math
from collections import defaultdict

from mindspore import mint, nn, mutable
from mindspore import ops as P
from mindspore.common.tensor import Tensor
from mindspore.ops._primitive_cache import _get_cache_prim

from . import constants as C
from .label_order import LabelOrder
from .opt_einusm_path import parse_opt_trace
from .sumproduct_pair import (sumproduct_pair_info, out_cacl_info, rearrange_tensor_to_mul,
                              rearrange_tensor_to_bmm, rearrange_tensor_to_out, prod_lst)


def _parse_equation(equation: str):
    """
    Parse the einsum equation into left-hand side (LHS), right-hand side (RHS), and number of operands.
    """
    arrow_pos = equation.find("->")
    if arrow_pos == -1:
        raise ValueError(f"invalid equation {equation}: require '->'")

    equation = equation.replace('...', '.')
    arrow_pos = equation.find("->")
    lhs = equation[:arrow_pos]
    rhs = equation[arrow_pos + 2:]
    num_ops = lhs.count(",") + 1

    return lhs, rhs, num_ops


def _parse_ellipsis(lhs: str, rhs: str):
    """
    Parse the ellipsis dims of equation
    """
    op_labels = lhs.split(",") + [rhs]
    ellipsis_idxes = []
    has_ellipsis = False
    for s in op_labels:
        ecnt = s.count(".")
        if ecnt > 1:
            raise ValueError(f"invalid equation {lhs} with multiple '...'")
        if ecnt == 1:
            pre, post = s.split(".")
            ellipsis_idxes.append((len(pre), len(post)))
            has_ellipsis = True
        else:
            ellipsis_idxes.append(None)

    if not has_ellipsis:
        return None

    return ellipsis_idxes


def _sum_dims_helper(a_shape: list, a_sums: tuple[str, ...]):
    """
    Helper function to filter out dimensions to be summed and return
      the remaining dimensions and their indices.
    a_shape: list[tuple[str, int], ...]; like this:  [('i', 0), ('j', 1)]
    a_sums: tuple[str, ...]):
    """
    res = []
    sum_dims = []
    for i, (k, v) in enumerate(a_shape):
        if k not in a_sums:
            res.append((k, v))
        else:
            sum_dims.append(i)

    return res, tuple(sum_dims)


def _cacl_mul_reshape(tensor: Tensor, add_dim_info: tuple[int, tuple[int, ...]]):
    """
    Calculate the new shape and permutation indices for multiplication operations.
    """
    if add_dim_info[0] == 0:
        return tensor

    add_dims, perm_ids = add_dim_info
    added_shape = tensor.shape + (tuple([1]) * add_dims)
    new_shape = tuple(added_shape[i] for i in perm_ids)
    return tensor.reshape(new_shape)


def _reshape_of_bmm(ta: Tensor, gb: tuple, m: int, k: int, is_trans: bool):
    """
    reshape tensor for bmm with BMK or BKM format
    """
    new_shape = gb + (k, m) if is_trans else gb + (m, k)
    if new_shape != ta.shape:
        return ta.reshape(new_shape)
    return ta


def _cacl_matmul_reshape(ta, tb, bmm_info):
    """Reshape the tensor for matrix multiplication operations.
    Types:
        ta: Tensor
        tb: Tensor
        bmm_info: tuple[bool, bool, bool, tuple[int, ...], tuple[int, ...],
                  tuple[int, ...], tuple[int, ...]]
    """
    a_shape, b_shape = ta.shape, tb.shape
    is_batch, transpose_a, transpose_b, a_b, a_m, b_n, a_k = bmm_info

    m_dims = tuple(a_shape[d] for d in a_m)
    m = prod_lst(m_dims)
    n_dims = tuple(b_shape[d] for d in b_n)
    n = prod_lst(n_dims)
    k = prod_lst(tuple(a_shape[d] for d in a_k))

    gb, b_dims = (), ()
    if is_batch:
        b_dims = tuple(a_shape[d] for d in a_b)
        b = prod_lst(b_dims)
        gb = (b,)

    out_shape = b_dims + m_dims + n_dims
    if out_shape == gb + (m, n):
        out_shape = None

    # transpose_a and left or right in bmm indicate BMK or BKM
    ta = _reshape_of_bmm(ta, gb, m, k, transpose_a)
    tb = _reshape_of_bmm(tb, gb, n, k, not transpose_b)
    return ta, tb, out_shape


def _remove_a_diagonal(labels: str, shape: tuple[int, ...]):
    """
    Removes a diagonal element from the labels and shape, ensuring no duplicate labels.
    """
    if len(labels) != len(shape):
        raise ValueError(f"labels: {labels} and tensor shape: {shape} are different size")

    for i in range(len(labels) - 1, 0, -1):
        c = labels[i]
        idx = labels.find(c, 0, i)
        if idx >= 0:
            if shape[i] != shape[idx]:
                raise ValueError(f"tensor diagonal requires same size, \
                                 while with {shape[i]} and {shape[idx]}")

            pairs = [(labels[j], shape[j]) for j in range(len(labels)) if j not in (i, idx)]
            new_labels = [a for a, _ in pairs] + [c]
            new_shape = tuple(b for _, b in pairs) + (shape[i],)

            return (idx, i), "".join(new_labels), new_shape

    return None, labels, shape


def _flat_empty_struct(st: list):
    """
    Flattens an empty structure to None if it contains no non-empty elements.
    """
    for e in st:
        if e:
            return tuple(st)

    return None


def _convert_1_to_2(s: int):
    if s == 1:
        return 2
    return s


def _replace_e1_shape(shapes):
    """Shape equal to 1 will affect preprocessing, use 2 instead.
    Replaces all shape elements equal to 1 with 2.

    Args:
        shapes: list[tuple[int, ...], ...]
    """
    res = []
    for shape in shapes:
        new_shape = tuple(_convert_1_to_2(s) for s in shape)
        res.append(new_shape)

    return tuple(res)


def _get_ellipsis_shape(shape, label_part: tuple[int, int], elli_shapes: tuple[int, ...]):
    """
    replace shape of ellipsis dims
    """
    pre_ellipsis, post_ellipsis = label_part
    num_dims = len(shape)

    total_labels = pre_ellipsis + post_ellipsis
    if num_dims < total_labels:
        raise ValueError(f"({shape}) is invalid for given equtation, require not less than {total_labels}.")

    # The shape of the dimension before '...'
    pre_ellipsis_shape = shape[: pre_ellipsis]
    # The shape of the dimension after '...'
    post_ellipsis_shape = shape[num_dims - post_ellipsis :]

    if elli_shapes is not None:
        # note: elli_shapes may be tuple([])
        new_shape = pre_ellipsis_shape + elli_shapes + post_ellipsis_shape
    else:
        elli_shapes = tuple(shape[pre_ellipsis: num_dims - post_ellipsis])
        new_shape = pre_ellipsis_shape + tuple([prod_lst(elli_shapes)]) + post_ellipsis_shape

    return new_shape, elli_shapes


def _update_weight(cacl_info):
    """
    update weight by tensor's data volume
    """
    for info in cacl_info:
        w = max(info["WEIGHT"], C.MIN_WEIGHT_PROD)
        info["WEIGHT"] = math.log(w / C.MIN_WEIGHT_PROD, 64) + 1.0


class Einsum(nn.Cell):
    """
    Einsum operation
    """

    def __init__(self, equation, use_opt=True):
        """
        This operator performs tensor computations using Einstein summation convention (Einsum).
        Supports diagonalization, reduction, transposition, matrix multiplication, product operations,
        inner products, etc.

        Args:
            - equation (str)
                Specifies the computation to be performed. Only accepts:
                Letters ([a-z][A-Z]): Represent dimensions of input tensors
                ...: anonymous dimensions
                Commas (','): Separate tensor dimensions
                Arrow ('->'): Left side specifies input tensors, right side specifies desired output dimensions

            - use_opt (bool), optional
                Defaults to `True`. When set to `False`, performs contraction path optimization.

        Inputs:
            - *tensors,  list of tensor inputs of variable length

        Outputs:
            - output (Tensor)

        Supported Platforms:
            ``Ascend`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn, Tensor, ops
            >>> import numpy as np
            >>> import Einsum

            >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ms.float32)
            >>> y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), ms.float32)
            >>> equation = "ij,jk->ik"
            >>> einsum = Einsum(equation, use_opt=False)
            >>> output = einsum(x, y)
            >>> print(output.shape)
                (2, 2)

            >>> shapes = [(156, 16, 16), (660, 128, 16), (660, 128, 16)]
            >>> x, y, z = [ops.randn(tp) for tp in shapes]
            >>> equation = "ijk,zui,zuj->zuk"
            >>> einsum = Einsum(equation, use_opt=True)
            >>> output = einsum(x, y, z)

            # example: Linear layer implemented using einsum
            class EinsumLinear(nn.Cell):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features

                    self.weight = ms.Parameter(
                        Tensor(np.random.randn(out_features, in_features).astype(np.float32)),
                        name='weight'
                    )
                    self.bias = ms.Parameter(
                        Tensor(np.random.randn(out_features).astype(np.float32)),
                        name='bias'
                    )

                    # Define einsum operation
                    self.einsum = Einsum("ij,bj->bi")  # Define matrix multiplication pattern

                def construct(self, x):
                    # Perform matrix multiplication using einsum: output = x @ weight.T + bias
                    output = self.einsum(self.weight, x) + self.bias
                    return output
        """
        super().__init__()
        if not isinstance(equation, str):
            raise TypeError(f"For einsum, 'equation' must be a str, but got {type(equation)}.")
        self.equation = equation.replace(" ", "")
        self.lhs, self.rhs, self.num_ops = _parse_equation(self.equation)
        self.num_tensors = 2 * self.num_ops - 1
        self.contract_dims = self._get_contract_dims()
        self.ellipsis_idxes = _parse_ellipsis(self.lhs, self.rhs)
        self.use_opt = use_opt

        # uninited
        self.has_inited = False
        self.trace = None
        self.order_labels = None
        self.diag_ops = None
        self.sums_ops, self.perm_ops, self.step_ops = None, None, None

        if not use_opt or self.num_ops < 2:
            shapes = self._generate_a_random_shape()
            self._post_init(shapes)


    @staticmethod
    def _count_labels(op_labels):
        """
        Counts the occurrences of each unique label in the operation labels.

        Args:
            op_labels: list[str, ...]
        Returns:
            dict: A dictionary mapping each label to its count.
        """
        letter_count = defaultdict(int)

        for s in op_labels:
            unique_letters = set(s)
            for letter in unique_letters:
                letter_count[letter] += 1

        return dict(letter_count)


    @staticmethod
    def _bind_shape_with_label(in_shapes, op_labels, rt_list=True):
        """bind shape with label
        Args:
            in_shapes: tuple[tuple[int, ...], ...]
            op_labels: list[str, ...]
            rt_list: bool
        return example [{'i':2, 'j':3}, {'j':3, 'k':4}, {'k':4, 'i':2}]
        """
        bound_shapes = []
        for indices, shape in zip(op_labels, in_shapes):
            if rt_list:
                bound_shape = [(idx, dim) for idx, dim in zip(indices, shape)]
            else:
                bound_shape = {idx: dim for idx, dim in zip(indices, shape)}

            bound_shapes.append(bound_shape)

        return bound_shapes


    def _post_init(self, shapes):
        """
            Determine whether it has been initialized. If not, it will be called the first time it runs.
            1. Apply path contraction by opt_einsum
            2. Apply label order optimization
            3. Build calculation steps
        """
        base_trace = parse_opt_trace(self.equation, shapes, self.use_opt)

        op_labels, self.diag_ops, rm_diag_shapes = self._process_diagonal(shapes)
        rm_diag_shapes = _replace_e1_shape(rm_diag_shapes)

        tensor_infos = self._build_cacl_steps(rm_diag_shapes, op_labels, base_trace)
        base_order = self._get_base_order()
        order = LabelOrder(tensor_infos, base_trace, base_order)
        self.order_labels, self.trace = order.get_order()

        self.sums_ops, self.perm_ops, self.step_ops = self._build(rm_diag_shapes, op_labels, tensor_infos)
        self.has_inited = True


    def _get_base_order(self):
        """
        Generates a base order string by appending characters from lhs to rhs,
        excluding commas and duplicates.

        Returns:
            str: The base order string.
        """
        res = self.rhs
        for c in self.lhs:
            if c != ',' and c not in res:
                res += c

        return res


    def _process_diagonal(self, shapes):
        """
        Processes the diagonal elements of the tensors specified by the operation labels.

        Args:
            shapes: The shapes of the tensors.
              list[tuple[int, ...], ...]

        Returns:
            tuple: A tuple containing the new operation labels, diagonal operations, and new shapes.
        """
        op_labels = self.lhs.split(",")
        new_op_labels, new_shapes = [], []
        diag_ops = []
        for op, shape in zip(op_labels, shapes):
            diag_pairs = []
            while True:
                diag_pair, op, shape = _remove_a_diagonal(op, shape)
                if not diag_pair:
                    break
                diag_pairs.append(tuple(diag_pair))

            diag_ops.append(tuple(diag_pairs))
            new_op_labels.append(op)
            new_shapes.append(shape)

        for _ in range(self.num_ops - 1):
            diag_ops.append(None)
        new_diag_ops = _flat_empty_struct(diag_ops)
        return new_op_labels, new_diag_ops, new_shapes


    def _generate_a_random_shape(self):
        """
        Generates a random shape for the tensors based on the operation labels.

        Returns:
            list of tuples: The generated shapes for the tensors.
        """
        all_indices = set(self.lhs)

        # a random size
        dims = {idx: 8 for idx in all_indices}

        op_labels = self.lhs.split(",")
        input_shapes = []
        for labels in op_labels:
            shape = tuple(dims[label] for label in labels)
            input_shapes.append(shape)

        return input_shapes


    def _get_contract_dims(self):
        """
        Determines the dimensions to be contracted by comparing the sets of lhs and rhs.

        Returns:
            tuple: The dimensions to be contracted.
        """
        set1 = set(self.lhs)
        set2 = set(self.rhs + ",")
        diff_set = set1 - set2
        return tuple(diff_set)


    def _build_cacl_steps(self, in_shapes, op_labels, base_trace):
        """
        Builds the calculation steps for tensor contractions based on the input shapes and operation labels.

        Args:
            in_shapes (list of tuples): The shapes of the input tensors.
            op_labels: The List of labels; example: ["ijk, "zui", "zuj"]
            base_trace: list of Int pair; like [(1, 0), (2, 3)]

        Types:
            in_shapes: tuple[tuple[int, ...], ...]
            op_labels: list[str, ...]):

        Returns:
            tuple: A tuple of tuples, each containing input and calculation information for each step.
        """
        label_counts = Einsum._count_labels(op_labels)
        ops = Einsum._bind_shape_with_label(in_shapes, op_labels, rt_list=False)

        cacl_info = [None] * self.num_tensors

        input_info = []
        for labels in op_labels:
            input_info.append({"IN": labels, "FROM": C.T_INPUT})

        for i, j in base_trace:
            a_shape = ops[i]
            b_shape = ops[j]

            sum_labels = []
            a_labels_to_sum, b_labels_to_sum = [], []
            for d in self.contract_dims:
                if d in a_shape and d in b_shape:
                    label_counts[d] -= 1
                    if label_counts[d] == 1:
                        sum_labels.append(d)
                        label_counts[d] = 0
                elif label_counts[d] == 1:
                    if d in a_shape:
                        a_labels_to_sum.append(d)
                        label_counts[d] = 0
                    elif d in b_shape:
                        b_labels_to_sum.append(d)
                        label_counts[d] = 0

            new_shape, a_info, b_info, out_info = sumproduct_pair_info(a_shape, b_shape, a_labels_to_sum,
                                                                       b_labels_to_sum, sum_labels)
            ops.append(new_shape)
            input_info.append(out_info)

            # dict of calculate info about: matmul or mul
            cacl_info[i] = a_info
            cacl_info[j] = b_info

        cacl_info[-1] = out_cacl_info(ops[self.num_tensors - 1], self.rhs)
        _update_weight(cacl_info)

        res = tuple(zip(input_info, cacl_info))
        return res


    def _build(self, in_shapes, op_labels, ops):
        """
        Builds the tensor operations and permutations for the given input shapes.

        Args:
            in_shapes (list of tuples): The shapes of the input tensors.
            op_labels (list): The List of labels; example: ["ijk, "zui", "zuj"].
            ops (list): result of function _build_cacl_steps.

        Types:
            in_shapes: tuple[tuple[int, ...], ...]
            op_labels: list[str, ...]
            ops: tuple[tuple[dict[str, str], ...], ...])

        Returns:
            tuple: A tuple containing the sum dimensions, permutations, and step operations.
        """
        shape_infos = Einsum._bind_shape_with_label(in_shapes, op_labels, rt_list=True)

        perm_ops = [None] * self.num_tensors
        sums_ops = [None] * self.num_tensors
        step_ops = []

        for i, j in self.trace:
            a_mul_sums, b_mul_sums = ops[i][0].get("SUMS", []), ops[j][0].get("SUMS", [])
            a_info, b_info = ops[i][1], ops[j][1]
            t_type = a_info["CACL"]
            a_shape, a_sum_dims = _sum_dims_helper(shape_infos[i], a_info["SUMS"] + a_mul_sums)
            b_shape, b_sum_dims = _sum_dims_helper(shape_infos[j], b_info["SUMS"] + b_mul_sums)

            if t_type == C.T_MUL:
                a_perm, b_perm, cacl_info, new_shape = rearrange_tensor_to_mul(self.order_labels, a_shape, b_shape)
            else:
                a_perm, b_perm, cacl_info, new_shape = rearrange_tensor_to_bmm(self.order_labels, a_shape,
                                                                               a_info, b_shape, b_info)

            shape_infos.append(new_shape)
            perm_ops[i], perm_ops[j] = a_perm, b_perm
            sums_ops[i], sums_ops[j] = a_sum_dims, b_sum_dims
            step_ops.append((t_type, cacl_info))

        # out
        out_shape, out_sum_dims = _sum_dims_helper(shape_infos[self.num_tensors-1], self.contract_dims)
        sums_ops[-1] = out_sum_dims
        perm_ops[-1] = rearrange_tensor_to_out(out_shape, self.rhs)

        sums_ops = _flat_empty_struct(sums_ops)
        return sums_ops, tuple(perm_ops), tuple(step_ops)


    def _reshape_ellipsis(self, operands):
        """
        reshape the dims indicated by ellipses.
        """
        if not self.ellipsis_idxes:
            return operands, None

        new_operands = mutable(list())
        elli_shapes = None
        for i in range(len(operands)):
            if self.ellipsis_idxes[i]:
                new_shape, elli_shapes = _get_ellipsis_shape(operands[i].shape,
                                                             self.ellipsis_idxes[i], None)
                new_operands.append(operands[i].reshape(new_shape))
            else:
                new_operands.append(operands[i])

        return new_operands, elli_shapes


    def _reshape_ellipsis_out(self, out: Tensor, elli_shapes: tuple[int, ...]):
        # note: elli_shapes may be tuple([])
        if elli_shapes is not None and self.ellipsis_idxes[-1]:
            new_shape, _ = _get_ellipsis_shape(out.shape, self.ellipsis_idxes[-1], elli_shapes)
            return out.reshape(new_shape)
        return out


    def _apply_preprocess(self, t, i):
        """
        Applies a series of preprocessing operations on the tensor `t` based on the operations
        defined in `diag_ops`, `sums_ops`, and `perm_ops`.

        Args:
        - t (Tensor): The input tensor to be preprocessed.
        - i (int): The index used to access the specific operations for this tensor.

        Returns:
        - Tensor: The preprocessed tensor.
        """
        # diagonal
        if self.diag_ops and self.diag_ops[i]:
            for prev_dim, dim in self.diag_ops[i]:
                t = t.diagonal(0, prev_dim, dim)

        # sums
        if self.sums_ops and self.sums_ops[i]:
            t = mint.sum(t, dim=self.sums_ops[i], keepdim=False)

        # permute
        if self.perm_ops[i]:
            t = mint.permute(t, self.perm_ops[i])

        return t


    def _check_inputargs(self, operands):
        """Check operands."""
        if len(operands) != self.num_ops:
            raise ValueError("The number of input tensors is inconsistent with the expression.")
        for operand in operands:
            if not isinstance(operand, Tensor):
                raise TypeError(f"For einsum, members of 'operands' must be Tensor, but got {type(operand)}.")


    def construct(self, *operands):
        """
        Constructs the final output tensor by applying a series of operations defined in `trace` and `step_ops`.

        Args:
        - *operands: Variable number of input tensors.

        Returns:
        - Tensor: The final output tensor after applying all the operations.
        """
        self._check_inputargs(operands)
        operands, elli_shapes = self._reshape_ellipsis(operands)

        if not self.has_inited:
            shapes = [t.shape for t in operands]
            self._post_init(shapes)

        data = mutable(list(operands))

        for k in range(len(self.trace)):
            i, j = self.trace[k]
            t_type, bmm_info = self.step_ops[k]

            # Apply preprocessing to the selected tensors
            ta = self._apply_preprocess(data[i], i)
            tb = self._apply_preprocess(data[j], j)

            # Perform the specified operation (mul or matmul)
            if t_type == C.T_MUL:
                ta = _cacl_mul_reshape(ta, bmm_info[0])
                tb = _cacl_mul_reshape(tb, bmm_info[1])
                t_out = ta * tb
            else:
                mm_class = P.BatchMatMul if bmm_info[0] else P.MatMul
                matmul = _get_cache_prim(mm_class)(transpose_a=bmm_info[1], transpose_b=bmm_info[2])
                ta, tb, out_shape = _cacl_matmul_reshape(ta, tb, bmm_info)
                t_out = matmul(ta, tb)
                if out_shape:
                    t_out = t_out.reshape(out_shape)

            # append new tensor
            data.append(t_out)

        # Apply final preprocessing to the last tensor
        n = self.num_tensors - 1
        out_tensor = self._apply_preprocess(data[n], n)
        out_tensor = self._reshape_ellipsis_out(out_tensor, elli_shapes)

        return out_tensor

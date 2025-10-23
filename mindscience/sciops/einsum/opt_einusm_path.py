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
"opt einsum"

try:
    import opt_einsum
except ImportError:
    opt_einsum = None


def get_l2r_path(num_ops):
    """
    Generates a left-to-right contraction path for a given number of operations.

    Parameters:
    - num_ops (int): The number of operations (tensors) to contract.

    Returns:
    - trace (list of tuples): A list of tuples representing the contraction path.
    """
    trace = []
    init_idx = num_ops
    for i in range(1, num_ops):
        if i == 1:
            trace.append((0, i))
        else:
            trace.append((init_idx, i))
            init_idx += 1

    return trace


def parse_opt_trace(equation, shapes, use_opt):
    """
    Parses the optimal contraction path for a given equation and shapes of tensors.
    If opt_einsum is not available, or not requested, or there are fewer than 2 operations,
    use the default left-to-right path.

    Parameters:
    - equation (str): The Einstein summation equation.
    - shapes (list of tuples): The shapes of the tensors to be contracted.
    - use_opt (bool): A flag indicating whether to use the optimal contraction path.

    Returns:
    - trace (list of tuples): A list of tuples representing the contraction path.
    """
    num_ops = len(shapes)
    # use left to right case
    if not opt_einsum or not use_opt or num_ops <= 2:
        trace = get_l2r_path(num_ops)
        return trace

    que = list(range(num_ops))
    tupled_path = []

    # Get the optimal contraction path using opt_einsum.
    path_info = opt_einsum.contract_path(equation, *shapes, shapes=True)
    for contraction in path_info[1].contraction_list:
        inds, idx_rm, _, _, _ = contraction
        tupled_path.append((inds[0], inds[1], idx_rm))

    trace = []
    idx = num_ops
    for i, j, _ in tupled_path:
        ind1, ind2 = que[i], que[j]

        if i > j:
            i, j = j, i
        que.pop(j)
        que.pop(i)

        que.append(idx)
        idx += 1

        trace.append((ind1, ind2))

    return trace

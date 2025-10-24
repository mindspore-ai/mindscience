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
"label order"

from . import constants as C
from .graph_utils import (build_adja_mul, build_adja_bmm, union_adjacency_lists, difference_adjacency_lists,
                          symmetric_difference_adjacency_lists, topological_sort, is_not_conflict, reorder_subseq)


def _solve_max_weight_independent_set(graph, weights):
    """
    Find the maximum weight independent set in a given graph.
    An independent set is a set of nodes in which no two nodes are adjacent.
    The function uses a depth-first search (DFS) approach to explore all possible combinations of nodes,
    keeping track of the total weight and the selected nodes.
    It returns the maximum weight and the corresponding independent set.

    Args:
        graph (list[list[int, ...], ...]): adjacency table
        weights (list[int, ...]): list of weights

    Returns:
        max weight and selected indices
    """
    n = len(graph)
    # Preprocess the neighbor mask for each node
    neighbor_masks = [0] * n
    for i in range(n):
        for j in graph[i]:
            neighbor_masks[i] |= 1 << j

    max_weight, max_ind_set = 0, tuple([])

    def dfs(i, mask, total_weight, selected):
        nonlocal max_weight, max_ind_set
        if i >= n:
            if total_weight > max_weight:
                max_weight = total_weight
                max_ind_set = tuple(selected)
            return

        # Case where the current node is not selected
        if neighbor_masks[i] != 0:
            dfs(i + 1, mask, total_weight, selected)

        # Case where the current node is selected, update the mask to exclude neighbors
        if not mask & (1 << i):
            new_mask = mask | neighbor_masks[i]
            selected.append(i)
            dfs(i + 1, new_mask, total_weight + weights[i], selected)
            selected.pop(-1)

    selected_nodes = []
    dfs(0, 0, 0, selected_nodes)
    return max_ind_set, max_weight


class Constraint:
    """Represents a constraint with a node ID, adjacency list, weight, and swap flag.

    Args:
        node_id (int): The ID of the node.
        adja (list): The adjacency list of the node.
        weight (int): The weight of the node.
        swap_mn (bool): A flag indicating whether to swap the M, N of matrix multiplication.
        is_transpose (bool): A flag indicating whether is transposed in matrix multiplication.
        is_swapped (bool): A flag indicating whether swap "a@b" in matrix multiplication.
    """
    def __init__(self, id_, adja_, w_, sw_, t_):
        self.node_id: int = id_
        self.adja: list = adja_
        self.weight: float = w_
        self.swap_mn: bool = sw_
        self.is_transpose: bool = t_
        self.is_swapped: bool = False

    def __str__(self):
        return (f"{self.node_id}, "
                f"{self.adja}, "
                f"{self.weight}, "
                f"{self.swap_mn}, "
                f"{self.is_transpose}, "
                f"{self.is_swapped}")

    def add_extra_weight(self):
        # not transposed will has a little extra weight leading to mininize its number
        if not self.is_transpose:
            self.weight += 0.2

    def do_swap(self):
        # create a new object with swapped status
        res = Constraint(self.node_id, self.adja, self.weight, self.swap_mn,
                         not self.is_transpose)
        res.is_swapped = True
        res.add_extra_weight()
        return res


class LabelOrder:
    """Manages the order of labels for tensor operations and builds constraints
       for the maximum weighted independent set (MWIS).
    """

    def __init__(self, tensors_info, trace, base_order):
        """Initializes the LabelOrder with tensor information, trace, and base order.

        Args:
            tensors_info (dict): Information about the tensors involved in the operations.
            trace (list): A trace of the operations.
            base_order (list): The base order of labels.
        """
        self.trace = trace
        self.ops = tensors_info
        self.base_order = base_order
        self.constraints = []
        self.graph = self._build_mwis()


    @staticmethod
    def _get_adja_tp(in_info, out_info):
        """Determines the type of adjacency list to build based on input and output tensor information.

        Args:
            in_info (dict): Information about the input tensor.
            out_info (dict): Information about the output tensor.

        Returns:
            tuple: A tuple containing the input and output types.
        """
        t1, t2 = "", ""
        if in_info["FROM"] == C.T_INPUT:
            t1 = C.INPUT
        elif in_info["FROM"] == C.T_MUL:
            t1 = C.MID_MUL
        elif in_info["FROM"] == C.T_MATMUL:
            t1 = C.MID_MATMUL

        if out_info["CACL"] == C.T_MUL:
            t2 = C.MUL
        elif out_info["CACL"] == C.T_MATMUL:
            t2 = C.MATMUL
        elif out_info["CACL"] == C.T_OUT:
            t2 = C.OUT

        return (t1, t2)


    @staticmethod
    def _build_adja_bmm_helper(out_info, adja_in):
        """Helper function to build adjacency lists for batch matrix multiplication (BMM).

        Args:
            out_info (dict[str, int]): Information about the output tensor.
            adja_in (adja_in: dict[str, list[str, ...]]): The input adjacency list.

        Returns:
            tuple: A tuple containing conflict flags and adjacency lists for BMM.
        """
        b, m, k = out_info["B"], out_info["M"], out_info["K"]
        bmm_must_seq = out_info["BMM_MUST_SEQ"]

        no_conf_bmk, no_conf_bkm = False, False
        adja_bmk, adja_bkm = {}, {}
        if bmm_must_seq % 2 == 1:
            adja_bmk = build_adja_bmm(b, m, k, [])
            no_conf_bmk = is_not_conflict(adja_bmk, adja_in)

        if bmm_must_seq >= 2:
            adja_bkm = build_adja_bmm(b, k, m, [])
            no_conf_bkm = is_not_conflict(adja_bkm, adja_in)

        return no_conf_bmk, no_conf_bkm, adja_bmk, adja_bkm

    @staticmethod
    def is_swap_conf(ci: Constraint, cj: Constraint, swap_dict: dict[int, int]):
        is_swap_pair = swap_dict.get(ci.node_id, -1) == cj.node_id
        if is_swap_pair:
            return ci.is_swapped ^ cj.swap_mn

        return False

    def _build_mwis(self):
        """Builds the maximum weighted independent set (MWIS) from the tensor operations.

        Returns:
            tuple: A tuple containing the list of constraints and the graph representing the constraints.
        """
        swap_mn_pos = {}
        constraints_ = []
        num_ops = len(self.trace) + 1
        for id_, (in_info, out_info) in enumerate(self.ops):
            weight = out_info["WEIGHT"]
            adjas = self._build_adja(in_info, out_info)
            for adja in adjas:
                if isinstance(adja, tuple):
                    adja_, sw, is_transpose = adja
                    cs = Constraint(id_, adja_, weight, sw, is_transpose)
                    if sw:
                        idx = id_ - num_ops
                        pi_, pj_ = self.trace[idx]
                        swap_mn_pos[pi_] = id_
                        swap_mn_pos[pj_] = id_
                    constraints_.append(cs)
                elif adja:
                    cs = Constraint(id_, adja, weight, False, False)
                    constraints_.append(cs)

        for cs in constraints_:
            if cs.node_id in swap_mn_pos:
                new_cs = cs.do_swap()
                self.constraints.append(new_cs)

            cs.add_extra_weight()
            self.constraints.append(cs)

        n = len(self.constraints)
        graph = [list() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                ci, cj = self.constraints[i], self.constraints[j]
                if ci.node_id == cj.node_id or not is_not_conflict(ci.adja, cj.adja) or \
                        LabelOrder.is_swap_conf(ci, cj, swap_mn_pos):
                    graph[i].append(j)
                    graph[j].append(i)

        return graph


    def _build_adja_mid_matmul_helper(self, tp, in_info, out_info, swap_mn):
        """Helper function to build adjacency lists for intermediate matrix multiplication.

        Args:
            tp (str): The type of operation (MUL, MATMUL, OUT).
            in_info (dict[str, str]): Information about the input tensor.
            out_info (dict[str, int]): Information about the output tensor.
            swap_mn (bool): A flag indicating whether to swap the M, N.

        Returns:
            list: A list of adjacency lists and swap flags.
        """
        res = []
        if swap_mn:
            b, m, n = in_info["B"], in_info["N"], in_info["M"]
        else:
            b, m, n = in_info["B"], in_info["M"], in_info["N"]
        adja_bmn = build_adja_bmm(b, m, n, out_info["SUMS"])

        if tp == C.MUL:
            res.append((adja_bmn, swap_mn, False))

        elif tp == C.MATMUL:
            no_conf_bmk, no_conf_bkm, adja_bmk, adja_bkm = LabelOrder._build_adja_bmm_helper(out_info, adja_bmn)
            # swap operands will change position in matmul a@b
            is_left = out_info["LEFT"] ^ swap_mn
            if no_conf_bmk:
                adja = symmetric_difference_adjacency_lists(adja_bmk, adja_bmn)
                res.append((adja, swap_mn, not is_left))
            if no_conf_bkm:
                adja = symmetric_difference_adjacency_lists(adja_bkm, adja_bmn)
                # is_transpose if is_left
                res.append((adja, swap_mn, is_left))

        elif tp == C.OUT:
            adja_out = build_adja_mul(out_info["OUT"], [], expand=True)
            if is_not_conflict(adja_bmn, adja_out):
                adja = difference_adjacency_lists(adja_out, adja_bmn)
                res.append((adja, swap_mn, False))
        else:
            raise ValueError("Error calculate type:", tp)

        return res


    def _build_adja(self, in_info, out_info):
        """Builds the adjacency list for the given input and output tensor information.

        Args:
            in_info (dict): Information about the input tensor.
            out_info (dict): Information about the output tensor.

        Returns:
            list: A list of adjacency lists.
        """
        res = []
        tp = LabelOrder._get_adja_tp(in_info, out_info)
        if tp == (C.INPUT, C.MUL):
            adja = build_adja_mul(in_info["IN"], out_info["SUMS"])
            res.append(adja)
        elif tp == (C.MID_MUL, C.OUT):
            adja = build_adja_mul(out_info["OUT"], [])
            res.append(adja)
        elif tp == (C.INPUT, C.MATMUL):
            adja_in = build_adja_mul(in_info["IN"], out_info["SUMS"])
            no_conf_bmk, no_conf_bkm, _, _ = LabelOrder._build_adja_bmm_helper(out_info, adja_in)

            if no_conf_bmk or no_conf_bkm:
                all_adja = {}
                for k in "BMK":
                    reorder_seq = reorder_subseq(out_info[k], in_info["IN"])
                    adja = build_adja_mul(reorder_seq, [])
                    all_adja = union_adjacency_lists(all_adja, adja)

                # is_transpose is determinted
                is_transpose = bool(no_conf_bmk) ^ out_info["LEFT"]
                res.append((all_adja, False, is_transpose))

        elif tp == (C.MID_MUL, C.MATMUL):
            _, _, adja_bmk, adja_bkm = LabelOrder._build_adja_bmm_helper(out_info, {})
            is_left = out_info["LEFT"]
            if adja_bmk:
                res.append((adja_bmk, False, not is_left))
            if adja_bkm:
                res.append((adja_bkm, False, is_left))
        elif tp[0] == C.MID_MATMUL:
            # intermediate matrix multiplication with and without swapping M and N dimensions.
            not_swap = self._build_adja_mid_matmul_helper(tp[1], in_info, out_info, False)
            swapped = self._build_adja_mid_matmul_helper(tp[1], in_info, out_info, True)
            res = not_swap + swapped
            # if empty indicating no constraints; but need the potential possibility of swap
            if not res:
                res = [({}, False, False), ({}, True, False)]
        else:
            # (C.INPUT, C.OUT), (C.MID_MUL, C.MUL) these cases don't need process
            pass

        return res

    def _greedy_add(self):
        """Finds the maximum weighted independent set (MWIS) using a greedy algorithm.

        Returns:
            tuple: A tuple containing the independent set and the total weight.
        """
        all_adja = {}
        ary = list(enumerate(self.constraints))
        ary.sort(key=lambda x: x[1].weight, reverse=True)
        has_seen = set()
        independent_set = []
        total_weight = 0
        for idx, c in ary:
            if idx not in has_seen and is_not_conflict(c.adja, all_adja):
                has_seen.update(self.graph[idx])
                all_adja = union_adjacency_lists(all_adja, c.adja)
                independent_set.append(idx)
                total_weight += c.weight

        return independent_set, total_weight

    def _comp_order(self, sel_order):
        """Computes the order of labels based on the selected order.

        Args:
            sel_order (list[str, ...]): The selected order of labels.

        Returns:
            list: The computed order of labels.
        """
        add_labels = [c for c in self.base_order if c not in sel_order]
        return "".join(sel_order + add_labels)

    def _swap_mn_to_trace(self, swapped_indices: tuple[int, ...]):
        """
        Purpose: Swaps the indices in the trace for the specified operations to adjust the contraction order.
        Steps:
            Adjusts the indices in the trace based on the swap_mn list.
            Converts the trace to a tuple to ensure immutability.
        """
        swapped_trace = []
        for pi, pj in self.trace:
            if pi in swapped_indices or pj in swapped_indices:
                swapped_trace.append((pj, pi))
            else:
                swapped_trace.append((pi, pj))

        return tuple(swapped_trace)

    def get_order(self):
        """Return best order and swapped operations of BMM."""
        if len(self.constraints) > C.SEARCH_K_THRE:
            independent_set, _ = self._greedy_add()
        else:
            weights = tuple(c.weight for c in self.constraints)
            independent_set, _ = _solve_max_weight_independent_set(self.graph, weights)

        all_adja = {}

        swapped_indices = []
        for idx in independent_set:
            cs = self.constraints[idx]
            all_adja = union_adjacency_lists(all_adja, cs.adja)
            if cs.is_swapped:
                swapped_indices.append(cs.node_id)

        swapped_trace = self._swap_mn_to_trace(swapped_indices)

        sel_order = []
        if all_adja:
            sel_order = topological_sort(all_adja)

        return self._comp_order(sel_order), swapped_trace

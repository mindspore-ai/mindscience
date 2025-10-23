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
"graph utils"

from collections import defaultdict


def build_adja_mul(in_labels, sum_labels, expand=False):
    """
    Build an adjacency list from the input labels, excluding the sum labels.
    If expand is True, each character is connected to all subsequent characters.
    Otherwise, each character is connected to the next character only.
    """
    adja = {}

    s = [c for c in in_labels if c not in sum_labels]

    for i in range(len(s) - 1):
        current_char = s[i]
        if expand:
            next_chars = list(s[i+1:])
        else:
            next_chars = [s[i + 1]]

        adja[current_char] = next_chars

    return adja


def build_adja_bmm(bb: str, mm: str, kk: str, sum_labels: list[str]):
    """
    Build an adjacency list from the given lists bb, mm, and kk, excluding the sum labels.
    Each element in bb is connected to all elements in mm and kk.
    Each element in mm is connected to all elements in kk.
    """
    bb = [c for c in bb if c not in sum_labels]
    mm = [c for c in mm if c not in sum_labels]
    kk = [c for c in kk if c not in sum_labels]
    adja = {}
    for b in bb:
        adja[b] = list(mm + kk)

    for m in mm:
        adja[m] = list(kk)

    return adja


def union_adjacency_lists(adj_list1, adj_list2):
    """
    Compute the union of two adjacency lists.
    The resulting adjacency list contains all nodes and their neighbors from both input lists.
    Types:
        adj_list1: dict[str, list[str, ...]]
        adj_list2: dict[str, list[str, ...]]):
    """
    union = {}
    # Traverse the first adjacency list
    for node, neighbors in adj_list1.items():
        union[node] = set(neighbors)
    # Traverse the second adjacency list
    for node, neighbors in adj_list2.items():
        if node in union:
            union[node].update(neighbors)
        else:
            union[node] = set(neighbors)
    # Convert sets back to lists
    for node in union:
        union[node] = list(union[node])
    return union


def difference_adjacency_lists(adj_list1, adj_list2):
    """
    Compute the difference between two adjacency lists.
    The resulting adjacency list contains nodes and their neighbors from the first list that are not in the second list.
    Types:
        adj_list1: dict[str, list[str, ...]]
        adj_list2: dict[str, list[str, ...]]):
    """
    difference = {}
    for node, neighbors in adj_list1.items():
        if node in adj_list2:
            # Compute the difference set
            diff_neighbors = set(neighbors) - set(adj_list2[node])
            if diff_neighbors:
                difference[node] = list(diff_neighbors)
        else:
            difference[node] = neighbors
    return difference


def intersection_adjacency_lists(adj_list1, adj_list2):
    """
    Compute the intersection of two adjacency lists.
    The resulting adjacency list contains nodes and their neighbors that are common to both input lists.
    Types:
        adj_list1: dict[str, list[str, ...]]
        adj_list2: dict[str, list[str, ...]]):
    """
    intersection = {}
    for node, neighbors in adj_list1.items():
        if node in adj_list2:
            # Compute the intersection set
            inter_neighbors = set(neighbors) & set(adj_list2[node])
            if inter_neighbors:
                intersection[node] = list(inter_neighbors)
    return intersection


def symmetric_difference_adjacency_lists(adj_list1, adj_list2):
    """
    Compute the symmetric difference of two adjacency lists.
    The resulting adjacency list contains nodes and their neighbors that are in one list but not the other.
    Types:
        adj_list1: dict[str, list[str, ...]]
        adj_list2: dict[str, list[str, ...]]):
    """
    # Compute the difference A - B
    diff1 = difference_adjacency_lists(adj_list1, adj_list2)
    # Compute the difference B - A
    diff2 = difference_adjacency_lists(adj_list2, adj_list1)
    # Combine the two differences
    symmetric_diff = union_adjacency_lists(diff1, diff2)
    return symmetric_diff


def topological_sort(graph):
    """
    Calculate the in-degree of each node
    graph type: dict[str, list[str, ...]]
    """
    in_degree = defaultdict(int)

    for u in graph:
        in_degree[u] = 0

    # Initialize in-degrees
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    # Add all nodes with in-degree 0 to the queue
    queue = [u for u in in_degree if in_degree[u] == 0]

    num_nodes = len(in_degree)

    # Store the result of the topological sort
    topo_order = []

    while queue:
        u = queue.pop(0)
        topo_order.append(u)

        # Decrease the in-degree of adjacent nodes
        for v in graph.get(u, []):
            in_degree[v] -= 1
            # If in-degree becomes 0, add to the queue
            if in_degree[v] == 0:
                queue.append(v)

    # If the topological sort result contains all nodes, return the result
    if len(topo_order) == num_nodes:
        return topo_order
    return []


def is_not_conflict(adja_l, adja_r):
    """
    Check if there is no conflict between two adjacency lists
    """
    if not adja_l or not adja_r:
        return True
    adja = union_adjacency_lists(adja_l, adja_r)
    order = topological_sort(adja)
    return len(order) > 0


def reorder_subseq(subseq: str, seq: str):
    """
    Reorder a subsequence to match the order in the given sequence
    """
    return tuple(c for c in seq if c in subseq)

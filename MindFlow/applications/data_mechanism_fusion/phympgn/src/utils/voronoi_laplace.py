# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""voronoi laplace"""
from shapely.geometry import MultiPoint
from tqdm import tqdm
import numpy as np


def compute_discrete_laplace(pos, edge_index, face):
    """
    Compute discrete Laplace-Beltrami operator.

    Args:
        pos: shape (n, 2), position of nodes
        edge_index: shape (2, e)
        face: shape (3, n_tri)

    Returns:
        l_matrix: shape (n, n), laplace matrix
        d_inv: d vector
    """
    w_matrix = compute_weight_matrix(pos, edge_index, face)  # (n, n)
    v = np.sum(w_matrix, axis=1)
    v_matrix = np.diag(v)  # (n, n)
    a_matrix = v_matrix - w_matrix
    assert not np.isinf(a_matrix).any()
    assert not np.isnan(a_matrix).any()

    d = compute_d_vector(pos, face)
    d_inv = 1 / d
    d_inv[np.isinf(d_inv)] = 0
    d_matrix_inv = np.diag(d_inv)  # (n, n)
    assert not np.isinf(d_matrix_inv).any()
    assert not np.isnan(d_matrix_inv).any()

    l_matrix_ = d_matrix_inv @ a_matrix
    return -l_matrix_, d_inv


# todo: optimize complexity
def compute_weight_matrix(pos, edge_index, face):
    """
    Compute weight matrix of discrete Laplace-Beltrami operator proposed by
    Pinkall and Polthier.

    Args:
        pos: shape (n, 2), position of nodes
        edge_index: shape (2, e)
        face: shape (3, n_tri)

    Returns:
        weights: shape (n, n), weight matrix
    """
    n = pos.shape[0]
    e = edge_index.shape[1]
    weights = np.zeros((n, n), dtype=np.float32)
    eps = np.finfo(np.float32).eps
    for e_i in tqdm(range(e)):
        edge = edge_index[:, e_i]
        i, j = edge
        nodes = find_opposite_nodes(edge, face)
        if nodes:
            p, q = nodes
            alpha = compute_opposite_angle([
                pos[i], pos[j], pos[p]
            ])
            beta = compute_opposite_angle([
                pos[i], pos[j], pos[q]
            ])
            if np.isnan(alpha) or np.isnan(beta):
                w = 0.
            elif alpha < eps or beta < eps:
                w = 0.
            else:
                w = (cot(alpha) + cot(beta)) / 2
            if np.isnan(w):  # for debug
                print('weights nan, e_{}, n_{}-n_{}'.format(e_i, i, j))
            weights[i, j] = w
    return weights


# todo: optimize complexity
def compute_d_vector(pos, face):
    """
    Compute d matrix of discrete Laplace-Beltrami operator proposed by Meyer.
    """
    d_vector = []
    n = pos.shape[0]
    for i in tqdm(range(n)):
        tris = find_node_triangles(i, face)
        area = compute_all_voronoi_area(pos, tris)
        d_vector.append(area)
        if np.isnan(area): # for debug
            print('d nan, n_{}'.format(i))
    return np.array(d_vector, dtype=np.float32)


def find_opposite_nodes(edge, triangles):
    """
    Find the two opposite nodes of the edge in the triangles mesh.
    Args:
        edge: shape (2,), (v_i, v_j)
        triangles: shape (3, n_tri), each column is (v_p, v_q, v_r)

    Returns:
        nodes: null List if the two opposite nodes aren't found or (v_a, v_b)
    """
    nodes = []
    n_tri = triangles.shape[1]
    for i in range(n_tri):
        tri = triangles[:, i]
        is_subset = np.all(np.isin(edge, tri))
        if is_subset:
            mask = ~np.isin(tri, edge)
            diff = tri[mask]
            nodes.append(diff.item())
    if len(nodes) == 1:
        nodes = []
    assert len(nodes) in {0, 2}
    return nodes


def compute_opposite_angle(triangle):
    """
    Compute the opposite angle of edge ij in triangle.
    Args:
        triangle: len (3,), position three nodes (i, j, k)

    Returns:
        angle: shape (1,), the angle of vector ki and kj.
    """
    v_i, v_j, v_k = triangle[0], triangle[1], triangle[2]
    e_ki = v_k - v_i
    e_kj = v_k - v_j
    cos = np.dot(e_ki, e_kj) / \
          (np.linalg.norm(e_ki) * np.linalg.norm(e_kj))
    # if cos > 1. or cos < -1., angle will be nan.
    angle = np.arccos(cos)
    return angle


def cot(theta):
    """
    cot = 1 / tan
    """
    ret = 1 / np.tan(theta)
    if ret < np.finfo(np.float32).eps:
        ret = 0.
    return ret


def compute_tri_circumcenter(triangle):
    """compute triangle circumcenter"""
    a, b, c = triangle[0], triangle[1], triangle[2]
    if a.shape[0] == 2:
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (
            a[1] - b[1]))
        ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) + (
            b[0] ** 2 + b[1] ** 2) * (c[1] - a[1]) + (
                c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
        uy = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) + (
            b[0] ** 2 + b[1] ** 2) * (a[0] - c[0]) + (
                c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d
        center = np.stack([ux, uy], dtype=np.float32)
    else:
        ab = b - a
        ac = c - a
        ab_magnitude = np.linalg.norm(ab)
        ac_magnitude = np.linalg.norm(ac)

        # calculate triangle normal
        n = np.cross(ab, ac)
        n_magnitude = np.linalg.norm(n)

        # Calculate circumcenter
        center = a + (ab_magnitude * np.cross(
            ab_magnitude * ac - ac_magnitude * ab, n)) / (2 * n_magnitude ** 2)
    return center


def compute_voronoi_area(triangle):
    """
    Compute voronoi area.
    """
    a, b, c = triangle[0], triangle[1], triangle[2]

    ab = b - a
    ac = c - a
    cos_a = np.dot(ab, ac) / \
               (np.linalg.norm(ab) * np.linalg.norm(ac))

    eps = np.finfo(np.float32).eps
    if np.abs(cos_a - 1.0) < eps:
        area = 0.
    elif cos_a < 0:  # A is obtuse
        area = 0.5 * MultiPoint(triangle).convex_hull.area
    else:
        circumcenter = compute_tri_circumcenter(triangle)
        mab = (a + b) / 2
        mac = (a + c) / 2
        area = MultiPoint([a, mab, circumcenter, mac]).convex_hull.area
    return area


def compute_all_voronoi_area(pos, tris):
    """
    Compute area.
    """
    areas_sum = 0
    n_tri = tris.shape[0]
    for tri_i in range(n_tri):
        i, j, k = tris[tri_i]
        area = compute_voronoi_area([pos[i], pos[j], pos[k]])
        areas_sum += area
    return areas_sum


def find_node_triangles(node, triangles):
    """
    Find the triangles consists of the node
    """
    tris = []
    n_tri = triangles.shape[1]
    for i in range(n_tri):
        triangle = triangles[:, i]
        if node in triangles[:, i]:
            # move node to the first loconcatenateion in triangle
            tri = np.concatenate([[node], triangle[triangle != node]])
            tris.append(tri)
    return np.stack(tris)

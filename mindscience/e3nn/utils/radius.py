# Copyright 2022 Huawei Technologies Co., Ltd
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
"""radius"""
from scipy.spatial import cKDTree
import numpy as np


def _reshape_and_batch(x, batch_x):
    """_reshape_and_batch"""
    if x.ndim > 2:
        if batch_x is None:
            batch_x = np.broadcast_to(np.arange(0, x.shape[0]).reshape(-1, 1), (x.shape[0], x.shape[1])).flatten()
        x = x.reshape(-1, x.shape[-1])
    else:
        if batch_x is None:
            batch_x = np.zeros(x.shape[0], dtype=x.dtype)
        x = x.reshape((-1, 1)) if x.ndim == 1 else x

    return x, batch_x.astype(np.int64)


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    r"""
    Find all points in `x` for each element in `y` within distance `r`.

    Args:
        x (ndarray): node feature matrix of x.
        y (ndarray): node feature matrix of y.
        r (ndarray, float): the radius.
        batch_x (ndarray): batch vector of x. If it is none, then calculate based on x and return. Default: ``None``.
        batch_y (ndarray): batch vector of y. If it is none, then calculate based on y and return. Default: ``None``.
        max_num_neighbors (int): The maximum number of neighbors to return for each element in `y`. Dufault: ``32``.

    Returns:
        edge_index (numpy.ndarray) - including edges of source and destination.

        batch_x (numpy.ndarray) - batch vector of x.

        batch_y (numpy.ndarray) - batch vector of y.

    Raises:
        ValueError: If the last dimension of `x` and `y` do not match.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.utils import radius
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = np.random.random((5, 12, 3))
        >>> r = 0.5
        >>> edge_index, batch_x, batch_y = radius(x, x, r)
        >>> print(edge_index.shape)
        (2, 222)
        >>> print(batch_x.shape)
        (60,)
        >>> print(batch_y.shape)
        (60,)

    """
    if not x.shape[-1] == y.shape[-1]:
        raise ValueError(f"Feature size do not match.")
    if max_num_neighbors < 1:
        raise Warning(f'max_num_neighbors: {max_num_neighbors}')

    x, batch_x = _reshape_and_batch(x, batch_x)
    y, batch_y = _reshape_and_batch(y, batch_y)

    x = np.concatenate((x, 2 * r * batch_x.reshape(-1, 1).astype(x.dtype)), axis=-1)
    y = np.concatenate((y, 2 * r * batch_y.reshape(-1, 1).astype(y.dtype)), axis=-1)

    tree = cKDTree(x)
    _, col = tree.query(y, k=max_num_neighbors, distance_upper_bound=r + 1e-8)
    row = [np.full_like(c, i) for i, c in enumerate(col)]
    col = col.flatten()
    row = np.concatenate(row, axis=0)
    mask = col < int(tree.n)

    return np.stack([row[mask], col[mask]], axis=0), batch_x, batch_y


# pylint: disable=C0103
# pylint: disable=W0612
def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target'):
    r"""
    Computes graph edges to all points within a given distance.

    Args:
        x (ndarray): node feature matrix.
        r (ndarray, float): the radius.
        batch (Tensor): batch vector. If it is none, then calculate and return. Default: ``None``.
        loop (bool): whether contain self-loops in the graph. Dufault: ``False``.
        max_num_neighbors (int): The maximum number of neighbors to return for each element in `y`. Dufault: ``32``.
        flow (str): {'source_to_target', 'target_to_source'}, the flow direction when using in combination with
            message passing. Dufault: ``'source_to_target'``.

    Returns:
        edge_index (ndarray) - including edges of source and destination.

        batch (ndarray) - batch vector.

    Raises:
        ValueError: If `flow` is not in {'source_to_target', 'target_to_source'}.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.utils import radius_graph
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = np.random.random((5, 12, 3))
        >>> r = 0.5
        >>> edge_index, batch = radius_graph(x, r)
        >>> print(edge_index.shape)
        (2, 162)
        >>> print(batch.shape)
        (60,)
    """

    if flow not in ['source_to_target', 'target_to_source']:
        raise ValueError(f'`flow` should be in ["source_to_target", "target_to_source"].')
    (row, col), batch, _ = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return np.stack([row, col], axis=0), batch


def radius_full(x, y, batch_x=None, batch_y=None):
    r"""
    Find all points in `x` for each element in `y`.

    Args:
        x (Tensor): node feature matrix.
        y (Tensor): node feature matrix.
        batch_x (ndarray): batch vector of x. If it is none, then calculate based on x and return. Default: ``None``.
        batch_y (ndarray): batch vector of y. If it is none, then calculate based on y and return. Default: ``None``.

    Returns:
        edge_index (numpy.ndarray) - including edges of source and destination.

        batch_x (numpy.ndarray) - batch vector of x.

        batch_y (numpy.ndarray) - batch vector of y.

    Raises:
        ValueError: If the last dimension of `x` and `y` do not match.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.utils import radius_full
        >>> from mindspore import ops, Tensor
        >>> x = Tensor(ops.ones((5, 12, 3)))
        >>> edge_index, batch_x, batch_y = radius_full(x, x)
        >>> print(edge_index.shape)
        (2, 720)
        >>> print(batch_x.shape)
        (60,)
        >>> print(batch_y.shape)
        (60,)

    """
    if not x.shape[-1] == y.shape[-1]:
        raise ValueError(f"Feature size do not match.")

    if x.ndim > 2 and y.ndim > 2:
        b_x, b_y = x.shape[0], y.shape[0]
        len_x, len_y = x.shape[1], y.shape[1]
    else:
        b_x, b_y = 1, 1
        len_x, len_y = x.shape[0], y.shape[0]

    x, batch_x = _reshape_and_batch(x, batch_x)
    y, batch_y = _reshape_and_batch(y, batch_y)

    batch_unique = np.unique(batch_x)
    _row = []
    edge_dst = []
    for i in batch_unique:
        _row.extend(np.arange(len_y) + i * len_y)
        _col = np.arange(len_x) + i * len_x
        edge_dst.extend(np.broadcast_to(_col, (len_y, len_x)).flatten())
    edge_src = np.broadcast_to(np.array(_row).reshape(-1, 1), (len(_row), len_x)).flatten()
    edge_dst = np.array(edge_dst)

    return np.stack([edge_src, edge_dst]), batch_x, batch_y


def radius_graph_full(x, batch=None, loop=False, flow='source_to_target'):
    r"""
    Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): node feature matrix.
        batch (Tensor): batch vector. If it is none, then calculate and return. Default: ``None``.
        loop (bool): whether contain self-loops in the graph. Dufault: ``False``.
        flow (str): {'source_to_target', 'target_to_source'}, the flow direction when using in combination with
            message passing. Dufault: ``'source_to_target'``.

    Returns:
        edge_index (ndarray) - including edges of source and destination.

        batch (ndarray) - batch vector.

    Raises:
        ValueError: If `flow` is not in {'source_to_target', 'target_to_source'}.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.utils import radius_graph_full
        >>> from mindspore import ops, Tensor
        >>> x = Tensor(ops.ones((5, 12, 3)))
        >>> edge_index, batch = radius_graph_full(x)
        >>> print(edge_index.shape)
        (2, 660)
        >>> print(batch.shape)
        (60,)

    """
    if flow not in ['source_to_target', 'target_to_source']:
        raise ValueError(f'`flow` should be in ["source_to_target", "target_to_source"].')

    (row, col), batch, _ = radius_full(x, x, batch, batch)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return np.stack([row, col], axis=0), batch

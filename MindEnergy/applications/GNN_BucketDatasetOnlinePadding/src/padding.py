# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Batch-level padding for variable-length graph samples.

A sample keeps its own node count until the moment a batch is assembled; this
module is the only place that knows how to align them. Attributes are declared
in a table, so a column's axis is read from the table rather than from its
position in a list.

The two axes are handled separately:

* node axis: node-level attributes are padded to the batch maximum, so sample
  ``i`` owns rows ``[i * max_node, (i + 1) * max_node)`` of every such column;
* edge axis: edge-level attributes are only concatenated, so their values are
  left untouched. Which sample an edge belongs to is carried by
  ``edge_batch_ids``.
"""


from typing import Dict, List, Sequence, Tuple

import numpy as np


# Node-level columns of this application, with the value written into the pad
# region. Every fill is 0.0 today; the table exists because it is also the
# declaration of *which* columns are node-level, so a new column is registered
# here once and nothing else has to be kept in sync.
# Keep the fill semantic: a derived mask column must be filled with the value
# that means "virtual", not with 0.0.
NODE_FILLS = {
    "x0": 0.0,
    "node_fea1": 0.0,
    "node_fea2": 0.0,
    "node_fea3": 0.0,
    "labels": 0.0,
}

# Edge-level columns. They share one edge count per sample and are concatenated
# as they are; ``EDGE_ATTRS[0]`` is the reference column for the count.
EDGE_ATTRS = ("edge_fea1", "edge_fea2", "mean_edge")


def batch_max_nodes(x: Sequence[np.ndarray]) -> int:
    """Return the largest node count inside one batch.

    Args:
        x: Per-sample node arrays, each shaped ``(n_i, ...)``.

    Returns:
        ``max(n_i)``.

    Raises:
        ValueError: If the batch is empty.
    """
    if len(x) == 0:
        raise ValueError("cannot pad an empty batch")
    return max(int(sample.shape[0]) for sample in x)


def pad_to_max(sample: np.ndarray, max_node: int, fill: float = 0) -> np.ndarray:
    """Pad ``sample`` along axis 0 up to ``max_node`` rows.

    Only the leading axis is padded, so one call serves ``(n,)``, ``(n, d)``
    and ``(n, d1, d2)`` alike.

    Args:
        sample: Array whose first axis is the node axis.
        max_node: Target node count.
        fill: Value written into the padded region.

    Returns:
        The padded array, or ``sample`` itself when no padding is needed.
    """
    short = max_node - int(sample.shape[0])
    if short <= 0:
        return sample
    width = ((0, short),) + ((0, 0),) * (sample.ndim - 1)
    return np.pad(sample, width, mode="constant", constant_values=fill)


def pad_nodes(arrays: Dict[str, List[np.ndarray]],
              fills: Dict[str, float] = None,
              max_node: int = None) -> Tuple[Dict[str, np.ndarray], int]:
    """Pad and concatenate every node-level attribute of one batch.

    Args:
        arrays: ``{name: [per-sample array, ...]}``; all lists share one length.
        fills: ``{name: fill value}``; a name missing here is filled with 0.
        max_node: Override the padding target (defaults to the batch maximum).

    Returns:
        ``(padded, max_node)``; ``padded[name]`` is the concatenated array.

    Raises:
        ValueError: If ``arrays`` is empty, or if ``max_node`` is smaller than
            some sample (that would silently truncate data).
    """
    if not arrays:
        raise ValueError("no node-level attribute to pad")
    fills = NODE_FILLS if fills is None else fills
    if max_node is None:
        max_node = batch_max_nodes(next(iter(arrays.values())))
    for name, samples in arrays.items():
        if any(int(sample.shape[0]) > max_node for sample in samples):
            raise ValueError(f"{name} has a sample longer than max_node={max_node}")
        fill = fills.get(name, 0)
        arrays[name] = np.concatenate(
            [pad_to_max(sample, max_node, fill) for sample in samples], axis=0)
    return arrays, max_node


def edge_batch_ids(edge_counts: Sequence[int]) -> np.ndarray:
    """Map every edge of the batch to the sample it came from.

    Args:
        edge_counts: Edge count of each sample, in batch order.

    Returns:
        ``(sum(edge_counts),)`` integer array.
    """
    ids = []
    for i, count in enumerate(edge_counts):
        ids.extend([i] * int(count))
    return np.array(ids)

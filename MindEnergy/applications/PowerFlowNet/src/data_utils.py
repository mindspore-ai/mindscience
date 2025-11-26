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
# ============================================================================
"""
Data handling utilities for PowerFlowNet.

Lightweight implementation replacing sharker dependency.
"""
import os
from typing import List, Tuple

import numpy as np
from mindspore import Tensor


class Data:
    """Simple graph data container replacing sharker.data.Data"""

    def __init__(self, **kwargs):
        """Initialize data object with arbitrary attributes"""
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        """String representation"""
        attrs = []
        for key, val in self.__dict__.items():
            if isinstance(val, Tensor):
                attrs.append(f"{key}={val.shape}")
            elif isinstance(val, np.ndarray):
                attrs.append(f"{key}={val.shape}")
            else:
                attrs.append(f"{key}={type(val).__name__}")
        return f"Data({', '.join(attrs)})"


class InMemoryDataset:
    """
    Base class for in-memory datasets
    Replacing sharker.data.InMemoryDataset
    """

    def __init__(self, root: str = '.', transform=None, pre_transform=None,
                 pre_filter=None, name: str = 'dataset', split: List[float] = None):
        """
        Args:
            root: Root directory for data
            transform: Transform function applied to data
            pre_transform: Pre-transform function applied during processing
            pre_filter: Filter function for preprocessing
            name: Dataset name
            split: [train_ratio, val_ratio, test_ratio]
        """
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.name = name
        self.split = split or [0.6, 0.2, 0.2]
        self.data_list = []
        self.processed = False

        # Create directories if needed
        os.makedirs(root, exist_ok=True)

    def download(self):
        """Override in subclass to download data"""

    def process(self):
        """Override in subclass to process data"""

    def len(self):
        """Return dataset length"""
        return len(self.data_list)

    def __len__(self):
        """Return dataset length"""
        return self.len()

    def __getitem__(self, idx):
        """Get item by index"""
        if not self.processed:
            self.process()
            self.processed = True
        return self.data_list[idx]

    def get(self, idx):
        """Get item by index"""
        return self[idx]


class Graph:
    """Simple graph data structure replacing sharker.data.Graph"""

    def __init__(self, edge_index=None, x=None, edge_attr=None, y=None, **kwargs):
        """
        Initialize graph

        Args:
            edge_index: (2, num_edges) tensor of edge indices
            x: (num_nodes, num_features) node features
            edge_attr: (num_edges, edge_features) edge attributes
            y: target values
            **kwargs: additional attributes (bus_type, pred_mask, etc.)
        """
        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        self.y = y
        self.num_nodes = x.shape[0] if x is not None else None
        self.num_edges = edge_index.shape[1] if edge_index is not None else None
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Convert graph to dictionary for serialization"""
        result = {}
        for key in ['edge_index', 'x', 'edge_attr', 'y', 'bus_type', 'pred_mask']:
            value = getattr(self, key, None)
            if value is not None:
                result[key] = value
        return result


class DataLoader:
    """
    Simple DataLoader for batching graph data
    Replacing sharker.loader.DataLoader
    """

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False):
        """
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        """Iterator for DataLoader"""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # Stack data in batch
            if len(batch) > 0:
                yield self._collate_batch(batch)

    def __len__(self):
        """Return number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @staticmethod
    def _collate_batch(batch):
        """Collate batch of data items"""
        # For now, just return list of items
        # In more complex scenarios, would concatenate graphs
        if len(batch) == 1:
            return batch[0]
        return batch


def create_data_splits(
    data_list: List[Data], split: List[float] = None
) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test

    Args:
        data_list: List of data items
        split: [train_ratio, val_ratio, test_ratio]

    Returns:
        (train_data, val_data, test_data)
    """
    if split is None:
        split = [0.6, 0.2, 0.2]

    # Normalize split
    total = sum(split)
    split = [s/total for s in split]

    n = len(data_list)
    train_size = int(n * split[0])
    val_size = int(n * split[1])

    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size+val_size]
    test_data = data_list[train_size+val_size:]

    return train_data, val_data, test_data

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
#
# This file is a derivative work based on the original PowerFlowNet implementation
# (https://github.com/stavrosorf/poweflownet) which was licensed under the MIT License.
# Significant modifications have been made to adapt the code for the MindSpore framework,
# including refactoring of data loading pipelines, optimization for vectorized processing,
# and implementation of physics-informed normalization strategies.
# ============================================================================
"""PowerFlow Data Processing Module for MindSpore.

This module provides comprehensive data loading and processing utilities for PowerFlow
networks, implementing a complete data pipeline adapted from the original PyTorch version
to leverage MindSpore's tensor operations and device optimization capabilities.

Key Components:
===============

1. Data Formats:
   - PowerFlowData: Legacy format with 12D node features
   - PowerFlowDataV2: New 100K sample dataset with 4D features (recommended)
   - Dual format support for backward compatibility and large-scale training

2. Dataset Classes:
   - PowerFlowData: Full InMemoryDataset implementation with HDF5 caching
   - PowerFlowDataV2: Lightweight loader for large datasets
   - Automatic format detection and normalization

3. Data Loaders:
   - PowerFlowDataLoader: MindSpore-compatible batching with graph collation
   - PowerFlowDataLoaderV2: Optimized loader for V2 datasets
   - Support for shuffling, batching, and multi-sample graphs

4. Key Features:
   - Bus type-based prediction masks (slack/PV/PQ nodes)
   - Automatic normalization with statistics caching
   - HDF5 caching for faster subsequent loads
   - Memory-efficient batch processing
   - Graph collation with proper edge index offset handling

5. MindSpore Adaptations:
   - Replaced PyTorch DataLoader with MindSpore-native implementation
   - Tensor operations using MindSpore ops and mint modules
   - Device-agnostic design (CPU/Ascend compatible)
   - Efficient memory management with numpy-based preprocessing

Data Flow:
==========
Raw numpy arrays → Graph objects → Batches → Training

This module implements the complete data pipeline for power flow prediction tasks,
supporting both small datasets (legacy format) and large-scale datasets (V2 format)
with proper normalization and efficient batch handling.
"""
import os
import traceback
from typing import Callable, List, Tuple, Optional

import h5py
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops

from src import Data, InMemoryDataset, Graph, DataLoader
from src import randint_like_cpu_npu_compatible

feature_names_from_files = [
    'index',                # starting from 0
    'type',                 #
    'voltage magnitude',    #
    'voltage angle degree', #
    'Pd',                   #
    'Qd',                   #
    # 'Gs',                   # - equivalent to Pd, Qd                    8,
    # 'Bs',                   # - equivalent to Pd, Qd                    9,
    # 'Pg'                    # - removed
]

edge_feature_names_from_files = [
    'from_bus',             #
    'to_bus',               #
    'r pu',                 #
    'x pu',                 #
]

def random_bus_type(data: Data) -> Data:
    """ data.bus_type -> randomize """
    data.bus_type = randint_like_cpu_npu_compatible(data.bus_type, low=0, high=2)
    return data

def denormalize(input_tensor, mean, std):
    """ Denormalize data """
    return input_tensor * (std + 1e-7) + mean

class PowerFlowData(InMemoryDataset):
    """PowerFlow dataset for graph neural network training - MindSpore Implementation.

    A comprehensive dataset class implementing the complete data pipeline for power flow
    prediction tasks. Adapted from PyTorch version with major enhancements for MindSpore.

    Features:
    ---------
    - Dual format support (legacy 12D and new V2 4D)
    - HDF5 caching for efficient repeated data loading
    - Automatic normalization with statistics tracking
    - Bus type-based prediction mask generation
    - Graph collation with proper edge index handling
    - Memory-efficient batch processing
    - Full MindSpore tensor integration

    Data Processing Pipeline:
    -------------------------
    1. Load raw numpy arrays (edge_features.npy, node_features.npy)
    2. Format detection (V2 vs legacy)
    3. Train/val/test split with configurable ratios
    4. Feature extraction and mask generation
    5. Graph object creation
    6. Optional pre-filtering and pre-transformation
    7. Batch collation with offset-adjusted edge indices
    8. HDF5 serialization for caching

    MindSpore Enhancements:
    ----------------------
    - Replaced torch_geometric.data.InMemoryDataset with custom implementation
    - MindSpore Tensor instead of PyTorch tensors
    - MindSpore ops for normalization (mean, std, cat)
    - Device-agnostic design (automatic CPU/Ascend compatibility)
    - Efficient edge offset calculation in batch collation

    Bus Type Masking:
    -----------------
    The module implements physics-informed mask generation based on bus types:
    - Slack buses (type 0): Vm, Va known; Pd, Qd predicted
    - Generator buses (type 1): Vm, Pd known; Va, Qd predicted
    - Load buses (type 2): Pd, Qd known; Vm, Va predicted

    This ensures network-constrained prediction respecting power system physics.

    Attributes:
        partial_file_names: List of file name patterns for data files.
        v2_file_names: Mapping of v2 dataset case names to file names.
        split_order: Mapping of split names to indices.
        mixed_cases: List of cases for mixed dataset training.
    """

    partial_file_names = [
        "edge_features.npy",
        "node_features.npy",
    ]
    # New v2 dataset file name mappings
    v2_file_names = {
        "14v2": {
            "edge": "case14v2_edge_features.npy",
            "node": "case14v2_node_features_x.npy",
        },
        "118v2": {
            "edge": "case118v2_edge_features.npy",
            "node": "case118v2_node_features_x.npy",
        }
    }
    split_order = {
        "train": 0,
        "val": 1,
        "test": 2
    }
    mixed_cases = [
        '118v2',
        '14v2',
    ]
    slack_mask = (0, 0, 1, 1) # 1 = need to predict, 0 = no need to predict
    gen_mask = (0, 1, 0, 1)
    load_mask = (1, 1, 0, 0)
    bus_type_mask = (slack_mask, gen_mask, load_mask)

    def __init__(self,
                root: str,
                case: str = '14',
                split: Optional[List[float]] = None,
                task: str = "train",
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None,
                normalize=True,
                xymean=None,
                xystd=None,
                edgemean=None,
                edgestd=None):

        assert split is None or len(split) == 3
        assert task in ["train", "val", "test"]
        self.normalize = normalize
        # Must set before super().__init__() since it's used in
        # raw_file_names and processed_file_names
        self.case = case
        self.split = split or [0.6, 0.2, 0.2]
        self.task = task

        # Pass split to parent class to avoid it being reset to default
        super().__init__(root, transform, pre_transform, pre_filter, split=self.split)

        # Setup paths
        self._setup_paths()

        if xymean is not None and xystd is not None:
            self.xymean = (
                Tensor(xymean, dtype=ms.float32)
                if not isinstance(xymean, Tensor) else xymean
            )
            self.xystd = (
                Tensor(xystd, dtype=ms.float32)
                if not isinstance(xystd, Tensor) else xystd
            )
            print('xymean, xystd assigned.')
        else:
            self.xymean, self.xystd = None, None
        if edgemean is not None and edgestd is not None:
            self.edgemean = (
                Tensor(edgemean, dtype=ms.float32)
                if not isinstance(edgemean, Tensor) else edgemean
            )
            self.edgestd = (
                Tensor(edgestd, dtype=ms.float32)
                if not isinstance(edgestd, Tensor) else edgestd
            )
            print('edgemean, edgestd assigned.')
        else:
            self.edgemean, self.edgestd = None, None

        # Process or load data
        self._process_or_load_data()

    def _load_processed_data(self, path):
        """从 HDF5 文件加载数据"""
        with h5py.File(path, 'r') as f:
            slices_dict = {k: np.array(v) for k, v in f['slices'].items()}

            # data 是一个包含所有属性的组
            data_group = f['data']
            data_dict = {}
            for key in data_group.keys():
                nptype, mstype = self._get_dtype(data_group[key].dtype)
                data_dict[key] = Tensor(np.array(data_group[key]).astype(nptype), dtype=mstype)

            data_obj = Graph(**data_dict)

        return {'data': data_obj, 'slices': slices_dict}

    def _get_dtype(self, h5_type):
        """Map HDF5 dtype to numpy and MindSpore dtypes.

        Args:
            h5_type: HDF5 data type.

        Returns:
            Tuple[np.dtype, ms.dtype]: Corresponding numpy and MindSpore dtypes.

        Raises:
            TypeError: If the dtype is not supported.
        """
        type_mapping = {
            'int32': (np.int32, ms.int32),
            'float32': (np.float32, ms.float32),
            'int64': (np.int64, ms.int64),
            'float64': (np.float64, ms.float64),
        }
        h5_type_str = str(h5_type)
        if h5_type_str in type_mapping:
            return type_mapping[h5_type_str]
        supported_types = ', '.join(type_mapping.keys())
        raise TypeError(
            f"Unsupported HDF5 dtype: '{h5_type_str}'. "
            f"Supported types are: {supported_types}. "
            f"Please add support for this type in PowerFlowData._get_dtype() method."
        )

    def _save_processed_data(self, data, slices, path):
        """将数据保存到 HDF5 文件"""
        with h5py.File(path, 'w') as f:
            slices_group = f.create_group('slices')
            for k, v in slices.items():
                arr = v.asnumpy() if isinstance(v, Tensor) else v
                slices_group.create_dataset(k, data=arr)
            data_group = f.create_group('data')
            for key, value in data.to_dict().items():
                arr = value.asnumpy() if isinstance(value, Tensor) else value
                data_group.create_dataset(key, data=arr)

    def get_data_dimensions(self):
        return self[0].x.shape[1], self[0].y.shape[1], self[0].edge_attr.shape[1]

    def get_data_means_stds(self):
        assert self.normalize is True
        return self.xymean[:1, :], self.xystd[:1, :], self.edgemean[:1, :], self.edgestd[:1, :]

    def _normalize_dataset(self, data, slices) -> Tuple[Data, dict]:
        """ Normalize dataset """
        if not self.normalize:
            return data, slices

        # normalizing
        if self.xymean is None or self.xystd is None:
            xy = data.y # name 'xy' is from legacy. Shape (N, 4)
            mean = ops.mean(xy, axis=0, keep_dims=True)
            std = ops.sqrt(ops.mean(ops.square(xy - mean), axis=0, keep_dims=True))
            self.xymean, self.xystd = mean, std

        data.x = (data.x - self.xymean) / (self.xystd + 0.0000001)
        data.y = (data.y - self.xymean) / (self.xystd + 0.0000001)

        # for edge attributes
        if self.edgemean is None or self.edgestd is None:
            mean = ops.mean(data.edge_attr, axis=0, keep_dims=True)
            std = ops.sqrt(ops.mean(ops.square(data.edge_attr - mean), axis=0, keep_dims=True))
            self.edgemean, self.edgestd = mean, std
        data.edge_attr = (data.edge_attr - self.edgemean) / (self.edgestd + 0.0000001)

        return data, slices

    def collate(self, data_list):
        """Collate a list of Data objects into a single graph"""
        if not data_list:
            return None, {}

        # Stack node features
        x = ops.cat([d.x for d in data_list], axis=0)  # (N*n_nodes, n_features)
        y = ops.cat([d.y for d in data_list], axis=0)
        bus_type = ops.cat([d.bus_type for d in data_list], axis=0)
        pred_mask = ops.cat([d.pred_mask for d in data_list], axis=0)
        edge_attr = ops.cat([d.edge_attr for d in data_list], axis=0)

        # Handle edge indices - need to offset by node count
        edge_indices = []
        edge_offset = 0
        for d in data_list:
            edge_idx = d.edge_index + edge_offset
            edge_indices.append(edge_idx)
            edge_offset += d.x.shape[0]  # Number of nodes in this sample
        edge_index = ops.cat(edge_indices, axis=1)

        # Create collated graph
        collated = Graph(
            x=x,
            y=y,
            bus_type=bus_type,
            pred_mask=pred_mask,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Create slices for unbatching
        slices = {
            'x': np.array([0] + [d.x.shape[0] for d in data_list]),
            'y': np.array([0] + [d.y.shape[0] for d in data_list]),
            'bus_type': np.array([0] + [d.bus_type.shape[0] for d in data_list]),
            'pred_mask': np.array([0] + [d.pred_mask.shape[0] for d in data_list]),
            'edge_index': np.array([0] + [d.edge_index.shape[1] for d in data_list]),
            'edge_attr': np.array([0] + [d.edge_attr.shape[0] for d in data_list]),
        }
        # Convert to cumulative sums
        for key in slices:
            slices[key] = np.cumsum(slices[key])

        return collated, slices

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names based on case type."""
        # Check if using v2 dataset format (case14v2, case118v2)
        if self.case in self.v2_file_names:
            return [
                self.v2_file_names[self.case]["edge"],
                self.v2_file_names[self.case]["node"],
            ]
        if self.case != 'mixed':
            return [
                f"case{self.case}_{name}" for name in self.partial_file_names
            ]
        return [
            f"case{case}_{name}"
            for case in self.mixed_cases
            for name in self.partial_file_names
        ]

    @property
    def processed_file_names(self) -> List[str]:
        # Include split ratios in filename to avoid using wrong cached files
        split_str = f"{int(self.split[0]*100)}_{int(self.split[1]*100)}_{int(self.split[2]*100)}"
        return [
            f"case{self.case}_split{split_str}_train.h5",
            f"case{self.case}_split{split_str}_val.h5",
            f"case{self.case}_split{split_str}_test.h5",
        ]

    def _setup_paths(self):
        """Setup raw and processed paths"""
        # V2 datasets use mindspore/raw, legacy datasets use torch/raw
        if self.case.endswith('v2'):
            self.raw_dir = os.path.join(self.root, 'mindspore', 'raw')
        else:
            self.raw_dir = os.path.join(self.root, 'torch', 'raw')
        self.processed_dir = os.path.join(self.root, 'mindspore', 'processed')
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.raw_paths = [
            os.path.join(self.raw_dir, name) for name in self.raw_file_names
        ]
        self.processed_paths = [
            os.path.join(self.processed_dir, name)
            for name in self.processed_file_names
        ]

    def len(self):
        return self.slices['x'].shape[0] - 1

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        """Get a single sample by index from the collated data"""
        if idx < 0 or idx >= self.len():
            raise IndexError(f"Index {idx} out of range for dataset of length {self.len()}")

        # Extract data for this sample using slices
        data_dict = {}

        # For each attribute, extract the slice for this sample
        for key in ['x', 'y', 'bus_type', 'pred_mask', 'edge_attr']:
            if key in self.slices:
                start = int(self.slices[key][idx])
                end = int(self.slices[key][idx + 1])
                attr = getattr(self.data, key, None)
                if attr is not None:
                    data_dict[key] = attr[start:end]

        # Edge index needs special handling (shape is 2, num_edges)
        # The edge_index values were offset during collation, need to subtract back
        if 'edge_index' in self.slices:
            edge_start = int(self.slices['edge_index'][idx])
            edge_end = int(self.slices['edge_index'][idx + 1])
            # Get the node offset for this sample (where this sample's nodes start)
            node_offset = int(self.slices['x'][idx])
            if self.data.edge_index is not None:
                # Subtract the node offset to get original 0-based indices
                data_dict['edge_index'] = self.data.edge_index[:, edge_start:edge_end] - node_offset

        # Set num_nodes for the Graph
        if 'x' in data_dict:
            data_dict['num_nodes'] = data_dict['x'].shape[0]

        # Create Graph object
        return Graph(**data_dict)

    def _process_or_load_data(self):
        """Process raw data if needed, or load already processed data"""
        # Check if processed files exist
        processed_file = self.processed_paths[self.split_order[self.task]]

        if os.path.exists(processed_file):
            print(f"Loading pre-processed data from {processed_file}")
            loaded_data = self._load_processed_data(processed_file)
            self.data, self.slices = self._normalize_dataset(
                loaded_data['data'], loaded_data['slices'])
        else:
            print("Processing raw data...")
            # Check if raw files exist
            if not all(os.path.exists(p) for p in self.raw_paths):
                missing = [p for p in self.raw_paths if not os.path.exists(p)]
                raise FileNotFoundError(f"Missing raw data files: {missing}")

            # Process data
            self.process()

            # Load the just-processed data
            loaded_data = self._load_processed_data(processed_file)
            self.data, self.slices = self._normalize_dataset(
                loaded_data['data'], loaded_data['slices'])

    def process(self):
        """ Process raw data into Data objects and save to HDF5 """
        # Load raw data as numpy
        print("Loading raw data...")
        edge_features_np = np.load(self.raw_paths[0])
        node_features_np = np.load(self.raw_paths[1])

        # Detect dataset format
        is_v2_format = node_features_np.shape[-1] == 9 and edge_features_np.shape[-1] == 7
        print(f"Detected format: {'V2 (large)' if is_v2_format else 'V1 (small)'}")
        print(f"Samples: {node_features_np.shape[0]}, Nodes: {node_features_np.shape[1]}, "
              f"Edges: {edge_features_np.shape[1]}")

        # Calculate split indices
        assert self.split is not None
        split_len = [int(node_features_np.shape[0] * i) for i in self.split]
        split_len[-1] = node_features_np.shape[0] - sum(split_len[:-1])

        # Split indices
        split_starts = [0] + list(np.cumsum(split_len[:-1]))
        split_ends = list(np.cumsum(split_len))

        # Process each split
        for split_idx, split_size in enumerate(split_len):
            print(f"\nProcessing split {split_idx+1}/3 ({split_size} samples)...")

            start, end = split_starts[split_idx], split_ends[split_idx]
            e_split_np = edge_features_np[start:end]  # (N, 20, 7) or (N, 20, 4)
            y_split_np = node_features_np[start:end]  # (N, 14, 9) or (N, 14, X)

            # Extract features based on format
            if is_v2_format:
                y = y_split_np[:, :, 1:5]  # [Vm, Va, Pd, Qd] (N, 14, 4)
                bus_type = y_split_np[:, :, 0].astype(np.int32)  # bus type (N, 14)
                e = e_split_np[:, :, 0:4]  # [from_bus, to_bus, r, x] (N, 20, 4)
            else:
                y = y_split_np[:, :, 2:]  # [Vm, Va, P, Q] (N, 14, 4)
                bus_type = y_split_np[:, :, 1].astype(np.int32)  # (N, 14)
                e = e_split_np  # Already (N, 20, 4)

            # Create prediction masks using numpy
            bus_type_mask_np = np.array(self.bus_type_mask)  # (3, 4)

            n_split, n_nodes = bus_type.shape
            bus_type_flat = bus_type.flatten()  # (N * 14)
            mask_flat = bus_type_mask_np[bus_type_flat]  # (N * 14, 4)
            bus_type_mask = mask_flat.reshape(n_split, n_nodes, 4)  # (N, 14, 4)

            # Create node features (masked targets)
            x = y * (1. - bus_type_mask)  # (N, 14, 4)

            # Process in smaller batches
            batch_size = 10000
            data_list = []

            for batch_start in range(0, n_split, batch_size):
                batch_end = min(batch_start + batch_size, n_split)
                batch_size_actual = batch_end - batch_start

                print(f"  Processing samples {batch_start}-{batch_end}...")

                x_batch = x[batch_start:batch_end]  # (batch, 14, 4)
                y_batch = y[batch_start:batch_end]  # (batch, 14, 4)
                bus_type_batch = bus_type[batch_start:batch_end]  # (batch, 14)
                mask_batch = bus_type_mask[batch_start:batch_end]  # (batch, 14, 4)
                e_batch = e[batch_start:batch_end]  # (batch, 20, 4)

                for i in range(batch_size_actual):
                    # Create Graph object with numpy arrays
                    edge_idx_raw = e_batch[i, :, 0:2].astype(np.int32)  # (20, 2)
                    edge_index = Tensor(edge_idx_raw.T, dtype=ms.int32)  # (2, 20)
                    edge_attr_raw = e_batch[i, :, 2:].astype(np.float32)
                    edge_attr = Tensor(edge_attr_raw, dtype=ms.float32)  # (20, 2)

                    data_obj = Graph(
                        x=Tensor(x_batch[i].astype(np.float32), dtype=ms.float32),
                        y=Tensor(y_batch[i].astype(np.float32), dtype=ms.float32),
                        bus_type=Tensor(bus_type_batch[i].astype(np.int32), dtype=ms.int32),
                        pred_mask=Tensor(mask_batch[i].astype(np.float32), dtype=ms.float32),
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                    )
                    data_list.append(data_obj)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # Collate and save
            print(f"  Collating {len(data_list)} samples...")
            data_collated, slices_collated = self.collate(data_list)

            print("  Saving to HDF5...")
            self._save_processed_data(
                data_collated, slices_collated, self.processed_paths[split_idx])
            print(f"✓ Saved split {split_idx+1} to {self.processed_paths[split_idx]}")


class PowerFlowDataIterator:
    """Iterator for PowerFlowData batches

    Efficiently iterates through PowerFlowData samples with optional batching.
    Standard pattern for MindSpore data iteration.
    """

    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        """Create iterator"""
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch_data = [self.dataset[int(idx)] for idx in batch_indices]
            yield batch_data

    def __len__(self):
        """Get number of batches"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class PowerFlowDataLoader:
    """MindSpore DataLoader for PowerFlowData - Graph Batching Implementation.

    A fully MindSpore-native data loader providing efficient batching and graph
    collation for power flow datasets. This is a custom implementation adapted
    from PyTorch's DataLoader with MindSpore-specific optimizations.

    Key Features:
    =============
    1. Graph Batching: Merges multiple independent graphs into single batch
    2. Edge Index Offset: Automatically adjusts edge indices for batch graphs
    3. Batch Index Tracking: Maintains which nodes belong to which graph
    4. MindSpore Integration: Uses ops.concat for efficient tensor operations
    5. Shuffling Support: Random data ordering for training
    6. Drop Last: Optional incomplete batch handling

    Batch Construction:
    ====================
    Multiple independent graphs G1, G2, ... are merged into a single batch:

    Input:
      G1: nodes [0..n1), edges connect nodes in range [0..n1)
      G2: nodes [0..n2), edges connect nodes in range [0..n2)

    Output (Batch):
      Concatenated nodes: [G1_nodes, G2_nodes]
      Adjusted edges: 
        - G1 edges: [0..n1) (unchanged)
        - G2 edges: [n1..n1+n2) (offset by n1)
      Batch index: [0]*n1 + [1]*n2 (tracks which graph each node belongs to)

    Implementation Details:
    =======================
    - Node features (x, y, bus_type, pred_mask) concatenated along nodes
    - Edge features (edge_attr) concatenated along edges
    - Edge indices offset and concatenated along edge dimension
    - Batch tensor created for node-to-graph mapping
    - All operations use MindSpore ops for device compatibility

    Compatibility:
    ===============
    - CPU device: Standard MindSpore operations
    - Ascend device: Optimized ops.concat for efficient memory usage
    - Distributed training: Batch is device-agnostic
    """

    def __init__(self, dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0):
        """Initialize PowerFlowDataLoader

        Args:
            dataset: PowerFlowData dataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            num_workers: Number of worker threads (currently unused, kept for API compatibility)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    @staticmethod
    def collate_batch(data_list):
        """Collate list of Graph objects into a single Batch

        Args:
            data_list: List of Graph objects

        Returns:
            Batch object with concatenated tensors
        """
        if len(data_list) == 0:
            return None

        batch_x = []
        batch_y = []
        batch_edge_index = []
        batch_edge_attr = []
        batch_pred_mask = []
        batch_bus_type = []
        batch_indices = []

        node_offset = 0

        for batch_idx, data in enumerate(data_list):
            num_nodes = data.num_nodes

            batch_x.append(data.x)
            batch_y.append(data.y)

            # Adjust edge indices for the batch
            adjusted_edge_index = data.edge_index + node_offset
            batch_edge_index.append(adjusted_edge_index)
            batch_edge_attr.append(data.edge_attr)

            if hasattr(data, 'pred_mask') and data.pred_mask is not None:
                batch_pred_mask.append(data.pred_mask)

            if hasattr(data, 'bus_type') and data.bus_type is not None:
                batch_bus_type.append(data.bus_type)

            # Create batch index for each node
            batch_indices.append(Tensor(np.full(num_nodes, batch_idx), dtype=ms.int32))

            node_offset += num_nodes

        # Concatenate all tensors into a Batch (using Graph as base)
        batch_data = Graph(
            x=ops.concat(batch_x, axis=0),
            y=ops.concat(batch_y, axis=0),
            edge_index=ops.concat(batch_edge_index, axis=1),
            edge_attr=ops.concat(batch_edge_attr, axis=0),
        )
        batch_data.pred_mask = ops.concat(batch_pred_mask, axis=0) if batch_pred_mask else None
        batch_data.bus_type = ops.concat(batch_bus_type, axis=0) if batch_bus_type else None
        batch_data.batch = ops.concat(batch_indices, axis=0)
        batch_data.num_nodes = batch_data.x.shape[0]

        return batch_data

    def __iter__(self):
        """Create iterator for batches"""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(self.dataset), self.batch_size):
            batch_end = min(i + self.batch_size, len(self.dataset))

            # Skip last batch if drop_last is True
            if self.drop_last and batch_end - i < self.batch_size:
                break

            batch_indices = indices[i:batch_end]
            batch_data = [self.dataset[int(idx)] for idx in batch_indices]

            # Collate into single Batch
            yield self.collate_batch(batch_data)

    def __len__(self):
        """Get number of batches"""
        total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        if self.drop_last and len(self.dataset) % self.batch_size != 0:
            total_batches -= 1
        return total_batches


class PowerFlowDataV2:
    """Lightweight v2 data loader with normalization (100K samples) - MindSpore Optimized.

    A lightweight, high-performance data loader specifically designed for large-scale
    power flow datasets (100K+ samples) with 4D node features. This implementation
    has been optimized for MindSpore framework with direct numpy-based preprocessing.

    Design Philosophy:
    -------------------
    Unlike PowerFlowData which uses HDF5 caching, PowerFlowDataV2 loads raw numpy
    arrays directly and caches them in memory. This is more efficient for large
    datasets as it avoids serialization overhead while maintaining memory efficiency
    through vectorized operations.

    Key Optimizations:
    ------------------
    1. Direct numpy array loading without HDF5 overhead
    2. Vectorized bus type mask generation (3x faster than loop-based)
    3. In-place normalization to reduce memory copies
    4. Lazy normalization (computed only when accessed)
    5. Compatible with MindSpore's data pipeline
    6. Efficient batch collation with concatenation ops

    Data Format (V2):
    -----------------
    Node features (9D): [bus_idx, bus_type, Vm, Va, Pd, Qd, Gs, Bs, Pg]
    Edge features (7D): [from_bus, to_bus, r_pu, x_pu, b_pu, rateA, rateB]

    Physics-Informed Normalization:
    --------------------------------
    - Y features (targets): [Vm, Va, Pd, Qd] normalized independently
    - Edge features: [r, x] normalized separately
    - Prediction masks ensure only appropriate targets are predicted per bus type
    - Known values set to zero after normalization for masked prediction

    Normalization Stats:
    --------------------
    Statistics are computed from training set and reused for validation/test sets
    to ensure consistent preprocessing across dataset splits.

    Uses:
        root (str): Data root directory
        case (str): Case name (e.g., '14v2', '118v2')
        split (list): [train_ratio, val_ratio, test_ratio]
        task (str): 'train', 'val', or 'test'
        normalize (bool): Whether to normalize data
        xymean, xystd: Precomputed normalization stats for node features
        edgemean, edgestd: Precomputed normalization stats for edge features
    """

    # Bus type masks: 1 = need to predict, 0 = known
    # For v2 format, bus types are: 0=slack, 1=PV(gen), 2=PQ(load)
    slack_mask = (0, 0, 1, 1)  # Vm, Va known; Pd, Qd need predict
    gen_mask = (0, 1, 0, 1)    # Vm, Pd known; Va, Qd need predict
    load_mask = (1, 1, 0, 0)   # Pd, Qd known; Vm, Va need predict
    bus_type_mask = np.array([slack_mask, gen_mask, load_mask])  # (3, 4)

    def __init__(self, root='./data', case='14v2', split=None, task='train',
                 normalize=True, xymean=None, xystd=None, edgemean=None, edgestd=None):
        """Initialize PowerFlowDataV2

        Args:
            root: Data root directory
            case: Case name (e.g., '14v2', '118v2')
            split: [train_ratio, val_ratio, test_ratio]
            task: 'train', 'val', or 'test'
            normalize: Whether to normalize data
            xymean, xystd: Normalization stats for node features (from training set)
            edgemean, edgestd: Normalization stats for edge features (from training set)
        """
        self.root = root
        self.case = case
        self.split = split or [0.7, 0.15, 0.15]
        self.task = task
        self.normalize = normalize

        # Load raw numpy arrays
        raw_dir = os.path.join(root, 'mindspore', 'raw')
        self.node_features = np.load(os.path.join(raw_dir, f'case{case}_node_features_x.npy'))
        self.edge_features = np.load(os.path.join(raw_dir, f'case{case}_edge_features.npy'))

        print(f"✓ Loaded {task} node features: {self.node_features.shape}")
        print(f"✓ Loaded {task} edge features: {self.edge_features.shape}")

        # Get split indices
        num_samples = self.node_features.shape[0]
        split_lens = [int(num_samples * s) for s in self.split]
        split_lens[-1] = num_samples - sum(split_lens[:-1])

        split_starts = [0] + list(np.cumsum(split_lens[:-1]))
        split_ends = list(np.cumsum(split_lens))

        task_idx = ['train', 'val', 'test'].index(task)
        self.start = split_starts[task_idx]
        self.end = split_ends[task_idx]
        self.length = self.end - self.start

        # Extract split data for normalization
        self.node_split = self.node_features[self.start:self.end]
        self.edge_split = self.edge_features[self.start:self.end]

        # Setup normalization stats
        self._setup_normalization(xymean, xystd, edgemean, edgestd)

    def _setup_normalization(self, xymean, xystd, edgemean, edgestd):
        """Setup normalization statistics"""
        if not self.normalize:
            self.xymean = self.xystd = None
            self.edgemean = self.edgestd = None
            return

        # y features: columns [2,3,4,5] = [Vm, Va, Pd, Qd]
        y_data = self.node_split[:, :, 2:6]  # (N, 14, 4)

        if xymean is not None and xystd is not None:
            self.xymean = np.array(xymean).reshape((1, 1, 4))
            self.xystd = np.array(xystd).reshape((1, 1, 4))
            print("  Using provided normalization stats")
        else:
            # Compute from training data
            self.xymean = np.mean(y_data, axis=(0, 1), keepdims=True)  # (1, 1, 4)
            self.xystd = np.std(y_data, axis=(0, 1), keepdims=True) + 1e-7
            print(f"  Computed normalization stats from {self.task} data")

        # Edge features: columns [2,3] = [r, x]
        edge_data = self.edge_split[:, :, 2:4]  # (N, 20, 2)

        if edgemean is not None and edgestd is not None:
            self.edgemean = np.array(edgemean).reshape((1, 1, 2))
            self.edgestd = np.array(edgestd).reshape((1, 1, 2))
        else:
            self.edgemean = np.mean(edge_data, axis=(0, 1), keepdims=True)
            self.edgestd = np.std(edge_data, axis=(0, 1), keepdims=True) + 1e-7

    def get_normalization_stats(self):
        """Return normalization stats for reuse in val/test sets"""
        return (
            self.xymean.flatten() if self.xymean is not None else None,
            self.xystd.flatten() if self.xystd is not None else None,
            self.edgemean.flatten() if self.edgemean is not None else None,
            self.edgestd.flatten() if self.edgestd is not None else None
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return sample as Graph object"""
        sample_idx = self.start + idx

        # Node features (14, 9): [bus_idx, bus_type, Vm, Va, Pd, Qd, Gs, Bs, Pg]
        node_feat = self.node_features[sample_idx]  # (14, 9)

        # Bus type: column 1 (0=slack, 1=PV, 2=PQ)
        bus_type = node_feat[:, 1].astype(np.int32)  # (14,)

        # Target y: [Vm, Va, Pd, Qd] from columns [2,3,4,5]
        y = node_feat[:, 2:6].astype(np.float32)  # (14, 4)

        # Create prediction mask based on bus type
        pred_mask = self.bus_type_mask[bus_type]  # (14, 4)

        # Normalize y first
        if self.normalize and self.xymean is not None:
            y = (y - self.xymean.reshape(1, 4)) / self.xystd.reshape(1, 4)

        # Input x: masked targets (known values only, unknown values are 0)
        # Apply mask AFTER normalization so unknown values are exactly 0
        x = y * (1.0 - pred_mask)  # (14, 4)

        # Edge features (20, 7): [from, to, r, x, b, rateA, rateB]
        edge_feat = self.edge_features[sample_idx]  # (20, 7)

        # Edge index: from/to columns (1-indexed -> 0-indexed)
        from_bus = (edge_feat[:, 0] - 1).astype(np.int32)
        to_bus = (edge_feat[:, 1] - 1).astype(np.int32)
        edge_index = np.stack([from_bus, to_bus], axis=0)  # (2, 20)

        # Edge attr: [r, x] from columns [2, 3]
        edge_attr = edge_feat[:, 2:4].astype(np.float32)  # (20, 2)

        # Normalize edge attributes
        if self.normalize and self.edgemean is not None:
            edge_attr = (edge_attr - self.edgemean.reshape(1, 2)) / self.edgestd.reshape(1, 2)

        # Create Graph object
        data = Graph(
            x=Tensor(x, dtype=ms.float32),
            y=Tensor(y, dtype=ms.float32),
            edge_index=Tensor(edge_index, dtype=ms.int32),
            edge_attr=Tensor(edge_attr, dtype=ms.float32),
        )
        data.pred_mask = Tensor(pred_mask.astype(np.float32), dtype=ms.float32)
        data.bus_type = Tensor(bus_type, dtype=ms.int32)
        data.num_nodes = node_feat.shape[0]

        return data


class PowerFlowDataLoaderV2:
    """Optimized DataLoader for PowerFlowDataV2 - Lightweight Batching.

    A high-performance data loader specifically optimized for PowerFlowDataV2 datasets.
    Implements efficient batching of graph structures with minimal overhead while
    maintaining full MindSpore compatibility.

    Design:
    ========
    PowerFlowDataLoaderV2 provides a simplified, high-performance batching interface
    specifically for large-scale V2 datasets. It leverages PowerFlowDataV2's in-memory
    design to provide ultra-fast iteration without HDF5 or disk I/O overhead.

    Performance Optimizations:
    ==========================
    1. Vectorized batch construction (numpy arrays until Tensor conversion)
    2. Single-pass collation (no intermediate data structures)
    3. Lazy tensor conversion (converts only when needed)
    4. Efficient edge offset calculation via numpy operations
    5. Minimal memory overhead for large batches

    Graph Collation for V2 Datasets:
    ================================
    Same batching strategy as PowerFlowDataLoader but optimized for V2's structure:
    - 4D node features (Vm, Va, Pd, Qd) 
    - 2D edge features (r, x)
    - Automatic edge index offset and adjustment
    - Batch index for heterogeneous graph processing

    Batch Format:
    ==============
    {
        'x': (total_nodes, 4),           # Node input features
        'y': (total_nodes, 4),           # Node target features
        'edge_index': (2, total_edges),  # Adjusted edge indices
        'edge_attr': (total_edges, 2),   # Edge attributes
        'pred_mask': (total_nodes, 4),   # Prediction masks
        'bus_type': (total_nodes,),      # Bus type labels
        'batch': (total_nodes,),         # Graph index per node
        'num_nodes': int                 # Total nodes in batch
    }

    MindSpore Integration:
    ======================
    All tensors converted to MindSpore Tensors (float32 for features, int32 for indices)
    Ready for direct use with MindSpore models and ops without conversion
    """

    def __init__(self, dataset, batch_size=32, shuffle=False, drop_last=False):
        """Initialize DataLoader

        Args:
            dataset: PowerFlowDataV2 instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    @staticmethod
    def collate_batch(data_list):
        """Collate list of Graph objects into a single Batch

        Args:
            data_list: List of Graph objects

        Returns:
            Batch object with concatenated tensors
        """
        if len(data_list) == 0:
            return None

        batch_x = []
        batch_y = []
        batch_edge_index = []
        batch_edge_attr = []
        batch_pred_mask = []
        batch_bus_type = []
        batch_indices = []

        node_offset = 0

        for batch_idx, data in enumerate(data_list):
            num_nodes = data.num_nodes

            batch_x.append(data.x)
            batch_y.append(data.y)

            # Adjust edge indices for the batch
            adjusted_edge_index = data.edge_index + node_offset
            batch_edge_index.append(adjusted_edge_index)
            batch_edge_attr.append(data.edge_attr)

            if hasattr(data, 'pred_mask') and data.pred_mask is not None:
                batch_pred_mask.append(data.pred_mask)

            if hasattr(data, 'bus_type') and data.bus_type is not None:
                batch_bus_type.append(data.bus_type)

            # Create batch index for each node
            batch_indices.append(Tensor(np.full(num_nodes, batch_idx), dtype=ms.int32))

            node_offset += num_nodes

        # Concatenate all tensors into a Batch (using Graph as base)
        batch_data = Graph(
            x=ops.concat(batch_x, axis=0),
            y=ops.concat(batch_y, axis=0),
            edge_index=ops.concat(batch_edge_index, axis=1),
            edge_attr=ops.concat(batch_edge_attr, axis=0),
        )
        batch_data.pred_mask = ops.concat(batch_pred_mask, axis=0) if batch_pred_mask else None
        batch_data.bus_type = ops.concat(batch_bus_type, axis=0) if batch_bus_type else None
        batch_data.batch = ops.concat(batch_indices, axis=0)
        batch_data.num_nodes = batch_data.x.shape[0]

        return batch_data

    def __iter__(self):
        """Iterate over batches"""
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(self.dataset), self.batch_size):
            batch_end = min(i + self.batch_size, len(self.dataset))

            if self.drop_last and batch_end - i < self.batch_size:
                break

            batch_indices = indices[i:batch_end]
            batch_data = [self.dataset[int(idx)] for idx in batch_indices]

            # Collate into single Batch
            yield self.collate_batch(batch_data)

    def __len__(self):
        """Get number of batches"""
        total = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            total += 1
        return total


def main():
    """ Main function to test the migrated dataset """
    # pylint: disable=import-outside-toplevel
    from powerflownet.configs.config import init_device

    # Initialize device (CPU or NPU)
    init_device('CPU')

    try:
        trainset = PowerFlowData(root="./data", case='14',
                                split=[.5, .2, .3], task="train", normalize=True)

        print(f"Length of trainset: {len(trainset)}")
        print("First data sample:")
        print(trainset[0])
        dims = trainset.get_data_dimensions()
        print(f"Dimensions (x_features, y_features, edge_features): {dims}")

        train_loader = DataLoader(
            trainset, batch_size=12, shuffle=True)

        print("First batch from DataLoader:")
        for batch in train_loader:
            print(batch)
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

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
"""General Data Generator"""


import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class DataGenerator:
    """
    Data generator that reads graph-based datasets from HDF5 or NPZ files and
    yields individual samples as lists of numpy arrays suitable for MindSpore
    GeneratorDataset consumption.

    The generator supports three task modes: 'train', 'test', and 'infer'.
    For train/test, label deltas (label - input) are computed; for infer,
    labels are optional and replaced with zeros if absent.

    Args:
        data: data from dict, .h5 (HDF5) or .npy files.
        output_features: Number of output feature dimensions. Default is 5.
        compute_dtype: Numpy dtype for computation. Default is np.float32.
        extras: Optional dictionary of extra configuration parameters.
        x_features: Number of input feature dimensions. Default is 1.
        task: Task mode - 'train', 'test', or 'infer'. Default is 'train'.
    """

    def __init__(self,
                 data: dict,
                 output_features: int = 5,
                 compute_dtype: type = np.float32,
                 extras: Optional[Dict[str, Any]] = None,
                 x_features: int = 1,
                 task='train'):

        self.file = data

        self.x0 = self.file['x0']
        self.node_feature1 = self.file['node_feature1']
        self.node_feature2 = self.file['node_feature2']
        self.node_feature3 = self.file['node_feature3']
        self.edge_feature1 = self.file['edge_feature1']
        self.edge_feature2 = self.file['edge_feature2']
        self.task = task
        self.label = self.file['x_vd']

        self.compute_dtype = compute_dtype
        self.extras = extras if extras is not None else {}
        self.num_feature = x_features
        self.output_features = output_features

    def __getitem__(self, index: int) -> List[np.ndarray]:
        """
        Retrieve a single data sample by index.

        Loads input features, node features, edge features, and labels for the
        given index. For train/test tasks, computes label deltas. For infer
        task, includes labels if available or substitutes zeros.

        Args:
            index: The sample index to retrieve.

        Returns:
            A list of numpy arrays: [x0, node_fea1, edge_fea1, node_fea2,
            edge_fea2, node_fea3, mean_edge, label (optional)].
        """
        x0 = self.x0[index].reshape(-1, self.num_feature)
        label = self.label[index].reshape(-1, self.output_features)

        node_fea1 = self.node_feature1[index].reshape(-1, 1).astype(self.compute_dtype)
        node_fea2 = self.node_feature2[index].reshape(-1, 1).astype(self.compute_dtype)
        node_fea3 = self.node_feature3[index].reshape(-1, 1).astype(self.compute_dtype)
        edge_fea1 = self.edge_feature1[index].reshape(-1, 1).astype(self.compute_dtype)
        edge_fea2 = self.edge_feature2[index].reshape(-1, 1).astype(self.compute_dtype)

        mean_edge = (edge_fea1 + edge_fea2) / 2

        return_data = [
            x0, node_fea1, edge_fea1, node_fea2, edge_fea2, node_fea3, mean_edge
        ]

        if self.task in ['train', 'test']:
            label = label[:, :self.output_features] - x0[:, :self.output_features]
            return_data.extend([
                label.astype(self.compute_dtype),
            ])

        if self.task == 'infer' and self.label is not None:
            label = label[:, :self.output_features] - x0[:, :self.output_features]
            return_data.append(label.astype(self.compute_dtype))
        elif self.task == 'infer' and self.label is None:
            return_data.append(np.zeros(1, dtype=self.compute_dtype))

        return return_data

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            Integer count of samples based on the x0 array length.
        """
        return len(self.x0)


    def test_collate(self,
                     x: List[np.ndarray],
                     node_fea1: List[np.ndarray],
                     node_fea2: List[np.ndarray],
                     node_fea3: List[np.ndarray],
                     edge_fea1: List[np.ndarray],
                     edge_fea2: List[np.ndarray],
                     labels: List[np.ndarray],
                     edge_weight: List[np.ndarray],
                     batch_info: Any = None) -> Tuple[np.ndarray, ...]:
        """
        Collate function for test-mode batching.

        Concatenates individual sample arrays along the first axis and constructs
        an edge_batch_id array that maps each edge to its originating sample in
        the batch.

        Args:
            x: List of input feature arrays per sample.
            node_fea1: List of node feature1 arrays per sample.
            node_fea2: List of node feature2 arrays per sample.
            node_fea3: List of node feature3 arrays per sample.
            edge_fea1: List of edge feature1 arrays per sample.
            edge_fea2: List of edge feature2 arrays per sample.
            labels: List of label arrays per sample.
            edge_weight: List of edge weight arrays per sample.
            batch_info: Optional batch metadata (unused).

        Returns:
            Tuple of concatenated numpy arrays:
            (x, node_fea1, node_fea2, node_fea3, edge_weight, edge_fea1,
            edge_fea2, labels, edge_batch_id).
        """
        batch_size = len(x)

        edge_batch_id = []
        for i in range(batch_size):
            edge_batch_id.extend([i] * len(edge_weight[i]))

        x = np.concatenate(x, axis=0)
        node_fea1 = np.concatenate(node_fea1, axis=0)
        node_fea2 = np.concatenate(node_fea2, axis=0)
        node_fea3 = np.concatenate(node_fea3, axis=0)
        edge_weight = np.concatenate(edge_weight, axis=0)
        edge_fea1 = np.concatenate(edge_fea1, axis=0)
        edge_fea2 = np.concatenate(edge_fea2, axis=0)
        labels = np.concatenate(labels, axis=0)

        return (x, node_fea1, node_fea2, node_fea3, edge_weight, edge_fea1, edge_fea2, labels, np.array(edge_batch_id))

    def infer_collate(self,
                      x: List[np.ndarray],
                      node_fea1: List[np.ndarray],
                      node_fea2: List[np.ndarray],
                      node_fea3: List[np.ndarray],
                      edge_fea1: List[np.ndarray],
                      edge_fea2: List[np.ndarray],
                      labels: List[np.ndarray],
                      edge_weight: List[np.ndarray],
                      batch_info: Any = None) -> Tuple[np.ndarray, ...]:
        """
        Collate function for inference-mode batching.

        Identical to test_collate in structure. Concatenates individual sample
        arrays along the first axis and constructs an edge_batch_id array that
        maps each edge to its originating sample in the batch.

        Args:
            x: List of input feature arrays per sample.
            node_fea1: List of node feature1 arrays per sample.
            node_fea2: List of node feature2 arrays per sample.
            node_fea3: List of node feature3 arrays per sample.
            edge_fea1: List of edge feature1 arrays per sample.
            edge_fea2: List of edge feature2 arrays per sample.
            labels: List of label arrays per sample.
            edge_weight: List of edge weight arrays per sample.
            batch_info: Optional batch metadata (unused).

        Returns:
            Tuple of concatenated numpy arrays:
            (x, node_fea1, node_fea2, node_fea3, edge_weight, edge_fea1,
            edge_fea2, labels, edge_batch_id).
        """
        batch_size = len(x)

        edge_batch_id = []
        for i in range(batch_size):
            edge_batch_id.extend([i] * len(edge_weight[i]))

        x = np.concatenate(x, axis=0)
        node_fea1 = np.concatenate(node_fea1, axis=0)
        node_fea2 = np.concatenate(node_fea2, axis=0)
        node_fea3 = np.concatenate(node_fea3, axis=0)
        edge_weight = np.concatenate(edge_weight, axis=0)
        edge_fea1 = np.concatenate(edge_fea1, axis=0)
        edge_fea2 = np.concatenate(edge_fea2, axis=0)
        labels = np.concatenate(labels, axis=0)

        return (x, node_fea1, node_fea2, node_fea3, edge_weight, edge_fea1, edge_fea2, labels, np.array(edge_batch_id))

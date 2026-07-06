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
"""Data Generator for train bucket"""


import math
import numpy as np


BUCKET_BATCH_EXPANSION_COEFFICIENT = 1.15


class BucketDatasetGenerator:
    """
    Provide data distribution of different buckets based on x_input length.

    This class groups training samples into buckets by node count, and yields
    dynamically batched data where each bucket has its own batch size. This
    enables efficient padding and batching for graph neural network training
    where graph sizes vary significantly.

    Args:
        dataset: The training dataset returned by get_dataset.
        sample_num_by_node: A dict mapping node count to sample count,
            e.g. {50: 100, 100: 200}.
        bucket_boundaries: List of bucket upper boundaries,
            e.g. [100, 200, 300]. Buckets are [0, b[0]), [b[0], b[1]), ...
        batch_sizes: List of batch sizes for each bucket,
            e.g. [16, 8, 4]. Length must match bucket_boundaries.
        padding_indices: List of attribute indices that require padding
            to the maximum node count within a batch.
        rank_size: Number of parallel shards used in distributed training.
    """

    def __init__(self, dataset, sample_num_by_node, bucket_boundaries, batch_sizes, padding_indices, rank_size):
        assert len(batch_sizes) == len(bucket_boundaries)
        self.dataset = dataset
        self.bucket_boundaries = bucket_boundaries
        self.batch_sizes = batch_sizes
        self.num_buckets = len(self.bucket_boundaries)
        self.padding_indices = padding_indices
        total_batches = self._calculate_total_batches(sample_num_by_node, bucket_boundaries, batch_sizes, rank_size)
        self.dataset_batch_num = math.ceil(BUCKET_BATCH_EXPANSION_COEFFICIENT * total_batches)
        self.random_list = self._generate_random_list()
        self._init_variables()

    def _calculate_total_batches(self, samples_by_node: dict[int, int], bucket_bounds: list[int], bucket_batch_sizes: list[int], rank_size: int) -> int:
        """
        Calculate total available batches for the current dataset configuration.

        Args:
            samples_by_node: A dict using node number as key and corresponding sample
                number as value, e.g. {50: 100, 100: 200}.
            bucket_bounds: Upper bound of each bucket, e.g. [100, 200, 300].
                Buckets are defined as [0, b[0]), [b[0], b[1]), ..., [b[n-2], b[n-1]).
            bucket_batch_sizes: Batch size for each bucket. Length must equal
                len(bucket_bounds).
            rank_size: Number of distributed shards for data partitioning.

        Returns:
            Total batch count (sum of the ceiling of batch counts for each bucket).

        Raises:
            ValueError: When the lengths of bucket_bounds and bucket_batch_sizes do
                not match, or when data falls outside the defined bucket ranges.
        """
        assert samples_by_node, "samples_by_node is None"
        if len(bucket_bounds) != len(bucket_batch_sizes):
            raise ValueError(f"The lengths of bucket_bounds ({len(bucket_bounds)}) and bucket_batch_sizes ({len(bucket_batch_sizes)}) do not match")

        if len(bucket_bounds) == 0:
            return 0

        for i in range(1, len(bucket_bounds)):
            if bucket_bounds[i] <= bucket_bounds[i - 1]:
                raise ValueError(f"bucket_bounds must be strictly monotonically increasing, but bucket_bounds[{i}]={bucket_bounds[i]} <= bucket_bounds[{i - 1}]={bucket_bounds[i - 1]}")

        bucket_samples = [0] * len(bucket_bounds)

        for nodes, samples in samples_by_node.items():
            bucket_idx = -1
            for i, upper_bound in enumerate(bucket_bounds):
                if nodes < upper_bound:
                    bucket_idx = i
                    break

            if bucket_idx == -1:
                raise ValueError(f"The number of nodes {nodes} is beyond the maximum bucket boundary {bucket_bounds[-1]}")

            if nodes <= 0:
                raise ValueError(f"The number of nodes {nodes} must be positive")

            bucket_samples[bucket_idx] += samples

        total_batches = 0
        for i, samples in enumerate(bucket_samples):
            batch_count = math.ceil(math.ceil(samples / rank_size) / bucket_batch_sizes[i])
            total_batches += batch_count

        return total_batches

    def _generate_random_list(self):
        """
        Generate a random bucket selection list for training iteration.

        Uses a binomial distribution to create a weighted random sequence that
        determines which bucket to draw from at each iteration step.
        """
        random_list = np.random.binomial(n=(self.num_buckets - 1), p=0.55, size=self.dataset_batch_num)
        random_list = (random_list + 2) % self.num_buckets
        return random_list

    def _init_variables(self):
        """
        Initialize or reset internal bucket state variables.

        Clears all bucket data buffers, resets the iteration counter, and
        prepares the remaining data processing stage.
        """
        self.data_bucket = {i: [] for i in range(self.num_buckets)}
        self.iter = 0
        self.remaining_data = []
        self.stage = 0

    def _get_bucket_index(self, x_input):
        """
        Determine which bucket a sample belongs to based on its x_input node count.

        Args:
            x_input: The input feature array whose first dimension is the node count.

        Returns:
            The bucket index corresponding to the node count.
        """
        num_node = x_input.shape[0]
        for i, threshold in enumerate(self.bucket_boundaries):
            if num_node < threshold:
                return i
        return len(self.bucket_boundaries) - 1

    def _get_batch_size(self, bucket_index):
        """
        Get the configured batch size for a specific bucket.

        Args:
            bucket_index: The index of the target bucket.

        Returns:
            The batch size associated with the given bucket.
        """
        return self.batch_sizes[bucket_index]

    def __next__(self):
        """
        Yield the next dynamically-batched data group from the bucket iterator.

        During the main iteration stage (stage 0), samples are distributed into
        buckets and a batch is returned whenever the current bucket meets its
        batch size requirement and matches the random selection list. Once all
        source data is consumed, remaining data in buckets is processed.
        """
        if self.stage != 0:
            return self._process_remaining_data()

        for item in self.iterator:
            bucket_index = self._get_bucket_index(item[0])
            self.data_bucket[bucket_index].append(item)

            for key in self.data_bucket.keys():
                data = self.data_bucket[key]
                current_batch_size = self._get_batch_size(key)
                if len(data) >= current_batch_size and self.random_list[self.iter] == key:
                    self.data_bucket[key] = self.data_bucket[key][current_batch_size:]
                    self.iter += 1
                    return self._package_data(data, current_batch_size)

        self.stage = 1
        return self._process_remaining_data()

    def _package_data(self, data, batch_size):
        """
        Package a set of data samples into a single batch with padding.

        For attributes at padding_indices, samples are zero-padded to the maximum
        node count in the batch. Edge indices in attribute index 2 are offset by
        each sample's node count to avoid cross-sample index collision.

        Args:
            data: List of data samples (tuples of numpy arrays) to batch together.
            batch_size: The number of samples in this batch.

        Returns:
            A list of numpy arrays representing the concatenated and padded batch,
            plus edge_batch_id and batch_size as the last two elements.
        """
        max_node_num = max(sample[0].shape[0] for sample in data)
        batch_data = data[0]
        edge_batch_id = [0] * len(batch_data[1])
        for attr_index in range(len(batch_data)):
            if attr_index in self.padding_indices:
                padded_sample = []
                for i in range(batch_size):
                    sample = data[i]
                    if sample[attr_index].shape[0] < max_node_num:
                        pad_width = ((0, max_node_num - sample[attr_index].shape[0]),) + ((0, 0),) * (sample[attr_index].ndim - 1)
                        padded_sample.append(np.pad(sample[attr_index], pad_width, mode='constant', constant_values=0))
                    else:
                        padded_sample.append(sample[attr_index])
                batch_data[attr_index] = np.concatenate(padded_sample, axis=0)
            else:
                for i in range(1, batch_size):
                    if attr_index == 2:
                        data[i][attr_index] += i * max_node_num
                        edge_batch_id.extend([i] * len(data[i][1]))
                        batch_data[attr_index] = np.concatenate((batch_data[attr_index], data[i][attr_index]), axis=1)
                    else:
                        batch_data[attr_index] = np.concatenate((batch_data[attr_index], data[i][attr_index]), axis=0)

        return batch_data + [edge_batch_id, batch_size]

    def _process_remaining_data(self):
        """
        Process remaining data left in buckets after the main iteration is complete.

        First tries to form full batches from remaining data, then falls back to
        partial batches. Resets internal state and raises StopIteration when no
        data remains.
        """
        for key in self.data_bucket.keys():
            data = self.data_bucket[key]
            current_batch_size = self._get_batch_size(key)
            if len(data) >= current_batch_size:
                self.data_bucket[key] = self.data_bucket[key][current_batch_size:]
                self.iter += 1
                return self._package_data(data, current_batch_size)

        for key in self.data_bucket.keys():
            data = self.data_bucket[key]
            if len(data) > 0:
                self.data_bucket[key] = []
                self.iter += 1
                return self._package_data(data, len(data))

        self._init_variables()
        raise StopIteration

    def __iter__(self):
        """
        Initialize the iterator for bucket-based batch generation.

        Resets internal state and creates a tuple iterator from the underlying
        dataset for enumeration.
        """
        self._init_variables()
        self.iterator = self.dataset.create_tuple_iterator(output_numpy=True)
        return self

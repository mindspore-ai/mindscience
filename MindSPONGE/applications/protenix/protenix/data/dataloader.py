# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides data loaders and samplers for training and testing.
"""

import math
from typing import Iterator, Optional, Sequence

import mindspore as ms
from mindspore import mint
from mindspore.dataset import DistributedSampler, Sampler
from ml_collections.config_dict import ConfigDict

from protenix.data.dataset import get_datasets
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


class WeightedSampler(Sampler):
    """
    A weighted sampler for single node.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool,
        seed: int = 0,
    ):
        """
        Args:
            weights (list or numpy array): A list or numpy array of weights.
            num_samples (int): The number of samples to be drawn.
            replacement (bool): Whether sampling is done with replacement.
            seed (int): The seed for the random number generator.
        """
        self.weights = ms.Tensor(weights, dtype=ms.float32)
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        """
        Generates an iterator over the sampled indices.

        This method uses a random number generator to sample indices based on the provided weights.
        The generator is seeded with the current seed and epoch to ensure reproducibility.

        Returns:
            iter: An iterator over the sampled indices.
        """
        g = ms.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = mint.multinomial(
            self.weights, self.num_samples, self.replacement, generator=g
        ).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedWeightedSampler(DistributedSampler):
    """
    A distributed weighted sampler for multiple nodes.
    """

    def __init__(
        self,
        dataset,
        weights: Sequence[float],
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset (Dataset): The dataset to be loaded.
            weights (list): The weights associated with the dataset.
            num_samples (int): The total number of samples to be drawn.
            num_replicas (int, optional): The number of replicas to use for distributed sampling. Defaults to None.
            rank (int, optional): The rank of the current process in a distributed environment. Defaults to None.
            replacement (bool, optional): Whether to sample with replacement. Defaults to True.
            seed (int, optional): The random seed for reproducibility. Defaults to 0.
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.weights = ms.Tensor(weights, dtype=ms.float32)
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        self.num_samples = num_samples

        self.num_samples_per_replica = int(
            math.ceil(self.num_samples / self.num_replicas)
        )
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """
        Generates an iterator over the sampled indices for the current process in a distributed environment.

        This method uses a random number generator to sample indices based on the provided weights.
        The generator is seeded with the current seed and epoch to ensure reproducibility.
        The sampled indices are then distributed across the replicas according to the rank of the current process.

        Returns:
            iter: An iterator over the sampled indices for the current process.
        """
        g = ms.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = mint.multinomial(
            self.weights, self.num_samples, self.replacement, generator=g
        ).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class KeySumBalancedSampler(Sampler):
    """
    A sampler that balances samples across workers based on the sum of a key.
    """

    def __init__(
        self,
        dataset,
        key: str,
        value_scale: float = 1.0,
        seed: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        This method initializes the KeySumBalancedSampler.
        It calls the `get_balanced_assignments` method to distribute the
        dataset indices across workers based on the key sum.

        Args:
            dataset (Dataset): The dataset to sample from.
            key (str): The key by which data will be balanced (integer
                value).
            value_scale (float): The multiplier of key value when computing
                the worker assignment weight
            num_replicas (int, optional): Number of processes participating
                in distributed training.
            rank (int, optional): Rank of the current process within
                num_replicas.
        """
        self.dataset = dataset
        self.key = key
        self.value_scale = value_scale
        self.seed = seed
        self.num_replicas = num_replicas or dist.get_world_size()
        self.rank = rank or dist.get_rank()

        # Get indices for this process after balancing by key sum
        worker_assignments = self.get_balanced_assignments()
        self.indices = worker_assignments[self.rank]

    def get_balanced_assignments(self):
        """
        Distribute dataset indices across workers such that the sum of key values
        assigned to each worker is as balanced as possible.
        """
        if self.seed is not None:
            # deterministically shuffle based on seed
            g = ms.Generator()
            g.manual_seed(self.seed)
            indices = mint.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # pad for len(dataset) to self.num_replicas if len(dataset) < self.num_replicas
        while len(indices) < self.num_replicas:
            indices += indices[: (self.num_replicas - len(indices))]

        if isinstance(self.dataset.indices_list, list):
            # e.g. recentPDB test set
            dataset_values = [
                x[self.key].astype(int)[0]
                for x in self.dataset.indices_list
            ]
        else:
            # e.g. posebuster test set
            dataset_values = (
                self.dataset.indices_list[self.key].astype(int).to_numpy()
            )

        # Sort indices by key value
        key_value_pairs = [(idx, dataset_values[idx]) for idx in indices]
        key_value_pairs.sort(key=lambda x: x[1], reverse=True)

        # Calculate the target number of samples per worker
        num_samples_per_worker = len(self.dataset) // self.num_replicas

        # Initialize containers for worker assignments and their current key sum
        worker_assignments = [[] for _ in range(self.num_replicas)]
        worker_sums = [0] * self.num_replicas
        total_samples = num_samples_per_worker * self.num_replicas

        # Distribute samples using a greedy strategy to balance the key sum
        for idx, key_value in key_value_pairs[:total_samples]:
            # Find the worker with the smallest sum that hasn't exceeded its target sample count
            min_worker = min(
                range(self.num_replicas),
                key=lambda i: (
                    worker_sums[i]
                    if len(worker_assignments[i]) < num_samples_per_worker
                    else float("inf")
                ),
            )
            worker_assignments[min_worker].append(idx)
            worker_sums[min_worker] += key_value**2

        # Fix any discrepancies in the number of samples
        all_indices = [idx for idx, _ in key_value_pairs]

        # Assign remaining samples if the dataset isn't divisible perfectly
        if len(all_indices) > total_samples:
            for i in range(len(all_indices) - total_samples):
                worker_assignments[i % self.num_replicas].append(
                    all_indices[total_samples + i]
                )

        # Return the indices assigned to the current worker
        return worker_assignments

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloaders(
    configs: ConfigDict,
    seed: int,
    num_shards=1,
    shard_id=None,
    error_dir: Optional[str] = None,
):
    """
    Generate data loaders for training and testing based on the given
    configurations and seed.

    Args:
        configs (ConfigDict): An object containing the data configuration
            information.
        seed (int): The random seed used for data sampling.
        num_shards (int): Number of shards for data partitioning.
        shard_id (int, optional): The shard ID for this process.
        error_dir (str, optional): The directory to store error information.
            Defaults to None.

    Returns:
        tuple: A tuple containing the training dataset and test dataset.

    """
    if shard_id is None:
        shard_id = []

    train_dataset, test_datasets = get_datasets(
        configs, error_dir, num_shards, shard_id
    )

    # Note: train_sampler is created but not returned
    # Keep for potential future use
    _ = WeightedSampler(
        weights=train_dataset.merged_datapoint_weights,
        num_samples=configs.data.epoch_size,
        replacement=True,
        seed=seed,
    )

    test_dataset_sizes = {}
    test_dataset = None
    for test_name, test_dataset in test_datasets.items():
        test_dataset_sizes[test_name] = len(test_dataset)
        # Note: test_sampler is created but not returned
        # Keep for potential future use
        # Commenting out to avoid undefined 'world_size' variable
        # _ = (
        #     KeySumBalancedSampler(
        #         test_dataset, key="num_tokens", seed=configs.seed
        #     )
        #     if world_size > 1
        #     else None
        # )

    logger.info(
        f"train data size: {len(train_dataset)}, "
        f"test size: {test_dataset_sizes}"
    )
    return train_dataset, test_dataset

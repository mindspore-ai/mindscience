# Copyright 2023 Huawei Technologies Co., Ltd
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
r"""Common utilities for loading datasets."""
from typing import List, Iterable, Union, Tuple
from abc import ABC, abstractmethod
import bisect

import numpy as np
from numpy.typing import NDArray
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset, BatchDataset, TupleIterator
from mindspore.communication import get_rank, get_group_size

from .env import DATASET_INDEXED, int_dtype


class Dataset(ABC):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting
    fetching a data sample for a given key. Subclasses could also optionally
    overwrite :meth:`__len__`, which is expected to return the size of the
    dataset.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Union[NDArray, Tuple[NDArray]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        if len(self.datasets) == 0:  # pylint: disable=C1801
            raise ValueError("'datasets' should not be an empty iterable.")
        dataset_sizes = [len(dataset) for dataset in self.datasets]
        self.cumulative_sizes = np.cumsum(dataset_sizes).tolist()

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Tuple[NDArray[float]]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


def concat_datasets(datasets: List[Dataset]) -> Dataset:
    r"""
    Obtain the concatenation of multiple datasets. For the case when there is
    only one dataset, return this dataset without applying concatenation.
    """
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)
    return dataset


class IndexedDataset(Dataset):
    r"""Add data indices to the data loaded from an existing dataset."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, idx_data: int) -> Tuple[NDArray]:
        data_tuple = self.dataset[idx_data]
        return_tuple = (*data_tuple, int_dtype(idx_data))
        return return_tuple

    def __len__(self) -> int:
        return len(self.dataset)


def split_data_tuple(data_tuple: Tuple[Tensor]) -> Tuple:
    r"""
    Split the data tuple.

    Args:
        data_tuple (Tuple[Tensor]): Data tuple provided by the dataset.

    Returns:
        input_tuple (Tuple[Tensor]): Input to the model.
        u_label (Tensor[float]): Label, expected model output.
        data_idx (Tensor[int]): Indices of the data samples in the dataset
            for the current data batch.
    """
    input_tuple = data_tuple[:-2]  # tuple of tensors
    u_label = data_tuple[-2]  # tensor
    data_idx = data_tuple[-1]  # tensor
    return input_tuple, u_label, data_idx


def datasets2loader(datasets: List[Dataset],
                    batch_size: int,
                    shuffle: bool,
                    num_workers: int,
                    create_iter: bool = False,
                    ) -> Union[BatchDataset, TupleIterator]:
    r"""
    Construct a dataloader for a list of datasets. In MindSpore, this will be
    an instance of the `BatchDataset` object if `create_iter` is False,
    and a `TupleIterator` object otherwise.
    """
    column_names = datasets[0].DATA_COLUMN_NAMES
    dataset = concat_datasets(datasets)
    if DATASET_INDEXED:
        column_names = column_names + ["idx_data"]
        dataset = IndexedDataset(dataset)

    try:  # data parallel case
        rank_id = get_rank()
        rank_size = get_group_size()
        bsz_device, residual = divmod(batch_size, rank_size)
    except RuntimeError:
        rank_id, rank_size = None, None
        bsz_device, residual = batch_size, 0

    if residual != 0:
        raise ValueError(
            f"The value of 'total_batch_size' ({batch_size}) should be "
            f"devisible by the number of devices ({rank_size}).")

    dataloader = GeneratorDataset(
        source=dataset,
        shuffle=shuffle,
        column_names=column_names,
        num_parallel_workers=num_workers,
        python_multiprocessing=False,
        num_shards=rank_size, shard_id=rank_id)
    dataloader = dataloader.batch(bsz_device)

    if create_iter:
        dataloader = dataloader.create_tuple_iterator()

    return dataloader

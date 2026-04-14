# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
meta
"""

from abc import ABC
from dataclasses import dataclass
from copy import deepcopy
from typing import Union

import numpy as np
from numpy import random
import mindspore as ms
from mindspore import ops
from mindspore.dataset import Sampler
from mindspore.dataset import GeneratorDataset
from .. import utils
from ..configs import Registry as R


def obj_from_dict(data_dict: dict):
    """
    Initialize an object based on the data contained in the dict. If the key contains `.`,
    an object should be constructed with given data as property based on the key like the pattern of `**.cls_name`.

    Args:
        data_dict (dict): An dictionary for construction of objects.

    Returns:
        obj_dict (dict): An dictionary for construction of object

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    obj_dict = {}

    for key, value in data_dict.items():
        if '.' in key:
            obj, prop = key.split('.')
            if obj not in obj_dict:
                obj_dict[obj] = {}
            obj_dict.get(obj)[prop] = value
        else:
            obj_dict[key] = value
    for key, props in obj_dict.items():
        if isinstance(props, dict) and 'cls_name' in props:
            cls_name = str(props.get('cls_name'))
            clazz = R.get(cls_name)
            obj_dict[key] = clazz(**props)
    return obj_dict


@R.register('data.MetaData')
@dataclass
class MetaData(ABC):
    """
    The abstract class that all of data class should be inherited.

    Args:
        cls_name (str): the registered name of each data class.
        detach (bool): if True, all of the matrix value of property should be np.ndarray,
                       otherwise they should be ms.Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    cls_name: str = None
    detach: bool = True

    def __post_init__(self):
        if self.cls_name is None:
            self.cls_name = type(self).cls_name
        if isinstance(self.cls_name, ms.Tensor):
            self.detach = False
        int32 = np.int32 if self.detach else ms.int32
        float32 = np.float32 if self.detach else ms.float32
        for key, value in self.__dict__.items():
            if isinstance(value, (np.ndarray, ms.Tensor)):
                if value.dtype in [np.floating, ms.float_]:
                    value = value.astype(float32)
                if value.dtype in [np.integer, ms.int_]:
                    value = value.astype(int32)
                if 0 in value.shape:
                    value = None
            setattr(self, key, value)

    @classmethod
    def keys(cls):
        """
        Obtained the keys of data fields of this class

        Returns:
            key (dict_keys): keys of data fields
        """
        keys = cls.__dataclass_fields__.keys()
        return keys

    @classmethod
    def from_dict(cls, **kwargs):
        """
        Initialization of data class with given parameters.
        With this method, the parameter not in the data fields could also be set.
        """
        appendix = {}
        for key, value in kwargs.items():
            if key not in cls.keys():
                appendix[key] = value
        for key in appendix:
            kwargs.pop(key)
        obj = cls(**kwargs)
        for key, value in appendix.items():
            setattr(obj, key, value)
        return obj

    def stop_gradient(self):
        """
        Stop the gradient calculation of its properties with the value of Tensor.
        """
        for key, value in self.__dict__.items():
            if isinstance(value, ms.Tensor):
                setattr(self, key, ops.stop_gradient(value))
        return self

    def to_tensor(self):
        """
        Transform the data type from np.ndarray to ms.Tensor for each of its property.
        """
        self.detach = False
        for key, value in self.__dict__.items():
            value = utils.to_tensor(value)
            self.__dict__[key] = value
        return self

    def to_array(self):
        """
        Transform the data type from ms.Tensor to np.ndarray for each of its property.
        """
        self.detach = True
        for key, value in self.__dict__.items():
            value = utils.to_array(value)
            self.__dict__[key] = value
        return self

    def to_dict(self):
        """

        Extract all of the value of the property into a dictionary.

        Returns:
            kwargs (dict): The dictionary contains the value of all properties.
        """
        kwargs = self.__dict__
        for key, value in kwargs.items():
            if value is None:
                value = np.array([]) if self.detach else ms.Tensor([])
            kwargs[key] = value
        return kwargs


class BatchSampler(Sampler):
    """
    Sampling methood for construction of the batch of a dataset. This class is specifcally used for GeneratorDataset.

    Args:
        data_size (int): The size of dataset
        batch_size (_type_): The size of each batch during iteration.
        drop_last (bool, optional): If drop the last batch whose size is smaller than the batch_size. Defaults to False.
        shuffle (bool, optional): For each iteration, if the order of the data is shuffled. Defaults to False.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, data_size: int, batch_size: int, drop_last=False, shuffle=False):
        super().__init__()
        self.data_size = data_size
        self.source = list(range(data_size))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.epoch = 1

    def __iter__(self):
        self.epoch += 1
        if self.shuffle:
            random.shuffle(self.source)
        for i in range(0, self.data_size, self.batch_size):
            # Drop reminder
            if i + self.batch_size <= self.data_size:
                yield self.source[i: i + self.batch_size]
            elif not self.drop_last:
                yield self.source[i:]


@R.register('dataset.base')
class BaseDataset:
    """
    The basic class of Dataset in Aichemist

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    _caches = []
    _seps = {'csv': ',', 'txt': '\t', 'tsv': '\t'}

    def __init__(self, **kwargs):
        if not hasattr(self, 'columns'):
            self.columns = self._caches
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        size = -1
        for cache in self._caches:
            size_ = len(getattr(self, cache))
            if size == -1:
                size = size_
            if size != size_:
                raise RuntimeError('The size of each data is not equal!')
        return size

    def __getitem__(self, index):
        output = []
        index = self._standarize_index(index)
        for cache in self._caches:
            data = getattr(self, cache)
            if isinstance(data, (np.ndarray, ms.Tensor)) or isinstance(index, (int, np.integer)):
                output.append(data[index])
            else:
                output.append([data[i] for i in index])

        return output

    def split(self, sizes: Union[tuple, list], randomize: bool = False):
        """
        Split The dataset into several subsets.

        Args:
            sizes (Union[tuple, list]): The size of each subset.
            randomize (bool, optional): If true the dataset will be split randomly, otherwise sequentially.
                                        Defaults to False.

        Raises:
            ValueError: The summation of the sizes should be equal to either 0 or its data size.
                        And all of number in sizes should be no small than 0.

        Returns:
            splits (List): The subsets split from itself based on the given sizes.
        """
        if isinstance(sizes, (tuple, list, np.ndarray)):
            sizes = np.array(sizes)
            assert (sizes >= 0).all()
            if sizes.dtype == float:
                assert np.sum(sizes) == 1
                assert (sizes <= 1).all()
                partitions = np.cumsum(sizes) * len(self)
            elif sizes.dtype == int:
                assert np.sum(sizes) == len(self)
                partitions = np.cumsum(sizes)
        elif isinstance(sizes, int):
            assert sizes < len(self)
            sizes = np.array([1/sizes] * sizes)
            partitions = np.cumsum(sizes) * len(self)
        else:
            raise ValueError('Unsuported data type of ratios!')
        start = 0
        order_ = np.arange(len(self))
        random.shuffle(order_)
        splits = []
        for part in partitions:
            end = int(part)
            if randomize:
                index = order_[start:end]
            else:
                index = list(range(start, end))
            splits.append(self.subset(index))
            start = end
        return splits

    def subset(self, indices: Union[int, list, np.integer, np.ndarray, slice]):
        """ Generate the subset of itself based on the given indices.

        Args:
            indices (Union[int, list, np.integer, np.ndarray, slice]):
                the indics of data that will be used to extract from itself.

        Returns:
            ds (BaseDataset): generated subset.
        """
        ds = deepcopy(self)
        for cache_ in self._caches:
            cache = getattr(ds, cache_)
            if isinstance(cache, (tuple, list)):
                sub = [cache[i] for i in indices]
            else:
                sub = cache[indices]
            setattr(ds, cache_, sub)
        return ds

    def dict_iterator(self, batch_size: int, drop_last: bool = False, shuffle: bool = False):
        """
        Generate the dict iterator for data iteration.

        Args:
            batch_size (int): The size of each batch during iteration.
            drop_last (bool, optional): If drop the last batch whose size is smaller than the batch_size.
                                        Defaults to False.
            shuffle (bool, optional): For each iteration, if the order of the data is shuffled. Defaults to False.

        Returns:
            iterator: Dict iterator created by GeneratorDataset.
                      During iteration, a dict will be generated that contains batch of dataset.
        """
        sampler = BatchSampler(len(self), batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
        iterator = GeneratorDataset(self, column_names=self.columns, sampler=sampler).create_dict_iterator()
        return iterator

    def tuple_iterator(self, batch_size, drop_last=False, shuffle=False):
        """
        Generate the tuple iterator for data iteration.

        Args:
            batch_size (int): The size of each batch during iteration.
            drop_last (bool, optional): If drop the last batch whose size is smaller than the batch_size.
                                        Defaults to False.
            shuffle (bool, optional): For each iteration, if the order of the data is shuffled. Defaults to False.

        Returns:
            iterator: Tuple iterator created by GeneratorDataset.
                      During iteration, a tuple will be generated that contains batch of dataset.
        """
        sampler = BatchSampler(len(self), batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
        iterator = GeneratorDataset(self, column_names=self.columns, sampler=sampler).create_tuple_iterator()
        return iterator

    def _standarize_index(self, index: Union[int, list, np.integer, np.ndarray, slice]):
        """
        Standarize indices

        Args:
            index (Union[int, list, np.integer, np.ndarray, slice]):
                the indics of data that will be used to extract from itself.

        Raises:
            ValueError: The type of index does not support.

        Returns:
            index (list): Stadard indices.
        """
        if isinstance(index, np.ndarray):
            index = index.tolist()
        elif isinstance(index, ms.Tensor):
            index = index.asnumpy().tolist()
        elif not isinstance(index, (list, int, np.integer, slice)):
            raise ValueError(f"Unknown index `{index}`")
        return index

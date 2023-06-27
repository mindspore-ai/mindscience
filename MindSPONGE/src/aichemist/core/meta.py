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
from typing import Any
from typing import Union
import numpy as np
from numpy import random
import mindspore as ms
from .. import util
from .config import Registry as R


def args_to_dict(*args, **kwargs):
    """
    args_to_dict
    """
    args_ = list(args)
    for i, arg in enumerate(args_):
        if isinstance(arg, MetaData):
            args_[i] = arg.to_dict()
        elif isinstance(arg, (tuple, list)):
            args_[i] = args_to_dict(*arg)[0]
        elif isinstance(arg, dict):
            args_[i] = args_to_dict(**arg)[1]
        else:
            pass
    for key, value in kwargs.items():
        if isinstance(value, MetaData):
            kwargs[key] = value.to_dict()
        elif isinstance(value, (tuple, list)):
            kwargs[key] = args_to_dict(*value)[0]
        elif isinstance(value, dict):
            kwargs[key] = args_to_dict(**value)[1]
        else:
            pass
    return args_, kwargs


def args_from_dict(*args, **kwargs):
    """
    args_from_dict
    """
    args_ = list(args)
    for i, arg in enumerate(args_):
        if isinstance(arg, dict) and '__class__' in arg:
            class_id = arg.get('__class__')
            clazz = R.order[class_id]
            args_[i] = clazz.from_dict(**arg)
        elif isinstance(arg, (tuple, list)):
            args_[i] = args_from_dict(*arg)[0]
        elif isinstance(arg, dict):
            args_[i] = args_from_dict(**arg)[1]
        else:
            pass
    for key, value in kwargs.items():
        if isinstance(value, dict) and '__class__' in value:
            class_id = value.get('__class__')
            clazz = R.order[class_id]
            kwargs[key] = clazz.from_dict(**value)
        elif isinstance(value, (tuple, list)):
            kwargs[key] = args_from_dict(*value)[0]
        elif isinstance(value, dict):
            kwargs[key] = args_from_dict(**value)[1]
        else:
            pass
    return args_, kwargs


@R.register('data.MetaData')
class MetaData(ABC):
    """
    MetaData
    """
    _caches = {}

    def __init__(self, detach=True, **kwargs) -> None:
        super().__init__()
        if '__class__' in kwargs:
            kwargs.pop('__class__')
        self.detach = detach

    def __getattribute__(self, name: str) -> Any:
        is_cache = False
        if name == '_caches':
            return super().__getattribute__(name)
        for suffix, cache in self._caches.items():
            if name.startswith(suffix):
                start = len(suffix)
                cache = getattr(self, cache)
                attr = cache.get(name[start:], None)
                is_cache = True
                break
        if not is_cache:
            attr = super().__getattribute__(name)
        return attr

    def __setattr__(self, name: str, value: Any):
        cache_ = None
        start = 0
        for suffix, cache in self._caches.items():
            if name.startswith(suffix):
                start = len(suffix)
                cache_ = getattr(self, cache)
            elif name.startswith(cache) and name != cache:
                start = len(cache + '_')
                cache_ = getattr(self, cache)
        if cache_ is not None:
            cache_[name[start:]] = value
        else:
            self.__dict__[name] = value

    @classmethod
    def from_dict(cls, **kwargs):
        """
        MetaData.from_dict
        """
        clazz = R.order[kwargs.get('__class__')]
        return clazz(**kwargs)

    def to_dict(self):
        """
        MetaData.to_dict
        """
        kwargs = {'__class__': R.get_id(self.registry_key_)}
        for param, values in self.__dict__.items():
            if param.startswith('_'):
                continue
            # pylint: disable=E1133
            if param in self._caches.values():
                for key in values:
                    kwargs[param + '_' + key] = values[key]
            if util.is_numeric(values):
                kwargs[param] = values
        return kwargs

    def to_tensor(self):
        """
        MetaData.to_tensor
        """
        self.detach = False
        for key in dir(self):
            if key.startswith('_'):
                continue
            value = getattr(self, key)
            if key in set(self._caches.values()):
                value = util.batch_to_device(value, detach=self.detach)
            else:
                value = util.to_tensor(value)
            self.__dict__[key] = value
        return self

    def to_array(self):
        """
        MetaData.to_array
        """
        self.detach = True
        for key in dir(self):
            value = getattr(self, key)
            value = util.to_array(value)
            self.__dict__[key] = value
        return self


@R.register('dataset.loader')
class DataLoader:
    """
    DataLoader
    """
    _caches = []
    _seps = {'csv': ',', 'txt': '\t', 'tsv': '\t'}

    def __init__(self, batch_size=32, verbose=True, shuffle=True, **kwargs):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.iterator = 0
        self.initialize(**kwargs)

    def __iter__(self):
        self.iterator = 0
        if self.shuffle:
            random.shuffle(self._order)
        return self

    def __len__(self):
        return len(self._order)

    def __getitem__(self, index):
        index = self._standarize_index(index)
        index = self._order[index].tolist()
        output = []
        for cache in self._caches:
            data = getattr(self, cache)
            if isinstance(index, int) or isinstance(data, (np.ndarray, ms.Tensor)):
                output.append(data[index])
            else:
                output.append([data[i] for i in index])
        return output

    def __next__(self):
        if self.iterator >= len(self):
            raise StopIteration
        end = self.iterator + self.batch_size
        if end > len(self):
            output = self[self.iterator:]
        else:
            output = self[self.iterator:end]
        self.iterator = end
        return output

    def initialize(self, **kwargs):
        """
        DataLoader.initialize
        """
        size = -1
        for key, value in kwargs.items():
            if key in self._caches:
                if size == -1:
                    size = len(value)
                else:
                    assert size == len(value)
            setattr(self, key, value)
        if size == -1:
            self._order = np.array([])
        else:
            self._order = np.arange(size)

    def split(self, ratios=None):
        """
        DataLoader.split
        """
        if isinstance(ratios, (tuple, list, np.ndarray)):
            ratios = np.array(ratios)
            assert (ratios >= 0).all()
            if ratios.dtype == float:
                assert np.sum(ratios) == 1
                assert (ratios <= 1).all()
                partitions = np.cumsum(ratios) * len(self)
            elif ratios.dtype == int:
                assert np.sum(ratios) == len(self)
                partitions = np.cumsum(ratios)
        elif isinstance(ratios, int):
            assert ratios < len(self)
            ratios = np.array([1/ratios] * ratios)
            partitions = np.cumsum(ratios) * len(self)
        else:
            raise ValueError('Unsuported data type of ratios!')
        start = 0
        splits = []
        for part in partitions:
            end = int(part)
            index = self._order[start:end]
            splits.append(self.subset(index))
            start = end
        return splits

    def subset(self, indices: Union[list, tuple, np.ndarray]):
        """
        DataLoader.subset
        """
        kwargs = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                pass
            elif key in self._caches:
                if isinstance(value, (list, tuple)):
                    kwargs[key] = [value[i] for i in indices]
                elif isinstance(value, (np.ndarray, ms.Tensor)):
                    kwargs[key] = value[indices]
                else:
                    raise ValueError('Unsported data type!')
            else:
                kwargs[key] = value
        subset = type(self)(**kwargs)
        return subset

    def _standarize_index(self, index, count=None):
        """
        standarize_index
        """
        if count is None:
            count = len(self)
        if isinstance(index, int):
            index = index
        elif isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif isinstance(index, np.ndarray):
            index = index.tolist()
        elif not isinstance(index, list):
            raise ValueError(f"Unknown index `{index}`")
        return index

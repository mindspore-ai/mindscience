# Copyright 2021 Huawei Technologies Co., Ltd
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
#pylint: disable=W0223
#pylint: disable=W0221
#pylint: disable=W0212
"""
Combine pde/ic/bc datasets together
"""
from __future__ import absolute_import
import copy

import mindspore.dataset as ds
from mindspore import log as logger

from .data_base import Data, ExistedDataConfig
from .existed_data import ExistedDataset
from .equation import Equation
from .boundary import BoundaryIC, BoundaryBC
from ..geometry import Geometry

_geomdata_dict = {
    "domain": Equation,
    "IC": BoundaryIC,
    "BC": BoundaryBC,
}


class Dataset(Data):
    r"""
    Combine datasets together.

    Parameters:
        geometry_dict (dict, optional): specifies geometry datasets to be merged. The key is geometry instance and
            value is a list of type of geometry. For example, geometry_dict = {geom : ["domain", "BC", "IC"]}.
            Default: None.
        existed_data_list (Union[list, tuple, ExistedDataConfig], optional): specifies existed datasets to be merged.
            For example, existed_data_list = [ExistedDataConfig_Instance1, ExistedDataConfig_Instance2]. Default: None.
        dataset_list (Union[list, tuple, Data], optional): specifies instances of data to be merged. For example,
            dataset_list=[BoundaryIC_Instance, Equation_Instance, BoundaryBC_Instance and ExistedData_Instance].
            Default: None.

    Raises:
        ValueError: If geometry_dict, existed_data_list and dataset_list are all None.
        TypeError: If the type of geometry_dict is not dict.
        TypeError: If the type of key of geometry_dict is not instance of class Geometry.
        TypeError: If the type of existed_data_list is not list, tuple or instance of ExistedDataConfig.
        TypeError: If the element of existed_data_list is not instance of ExistedDataConfig.
        TypeError: If the element of dataset_list is not instance of class Data.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import Rectangle, create_config_from_edict
        >>> from mindelec.data import Dataset
        >>> rectangle_mesh = edict({'domain': edict({'random_sampling': False, 'size': [50, 25]})})
        >>> rect_space = Rectangle("rectangle", coord_min=[0, 0], coord_max=[5, 5],
        ...                        sampling_config=create_config_from_edict(rectangle_mesh))
        >>> geom_dict = {rect_space: ["domain"]}
        >>> dataset = Dataset(geometry_dict=geom_dict)
    """
    def __init__(self, geometry_dict=None, existed_data_list=None, dataset_list=None):
        super(Dataset, self).__init__()
        if geometry_dict is None and existed_data_list is None and dataset_list is None:
            raise ValueError("Dataset should have at least one sub-dataset, but got None")

        if geometry_dict is not None:
            _check_type(geometry_dict, "geometry_dict", dict)
            _check_key_value(geometry_dict)

        if existed_data_list is not None:
            if isinstance(existed_data_list, ExistedDataConfig):
                existed_data_list = [existed_data_list]
            _check_type(existed_data_list, "existed_data_list", (list, tuple))

            for data_config in existed_data_list:
                _check_type(data_config, "the element in existed_data_list", ExistedDataConfig)

        if dataset_list is not None:
            if isinstance(dataset_list, Data):
                dataset_list = [dataset_list]
            _check_type(dataset_list, "dataset_list", (list, tuple))
            for dataset in dataset_list:
                _check_type(dataset, "the element in dataset_list", Data)

        self.existed_data_list = existed_data_list
        self.geometry_dict = geometry_dict
        self.dataset_list = dataset_list
        self.all_datasets = [] if dataset_list is None else dataset_list
        self.columns_list = None
        self._iterable_datasets = None

        self.num_dataset = len(dataset_list) if dataset_list is not None else 0
        if existed_data_list is not None:
            self.num_dataset += len(existed_data_list)
        if geometry_dict is not None:
            for geom in geometry_dict:
                self.num_dataset += len(geometry_dict[geom])
        logger.info("Total datasets number: {}".format(self.num_dataset))

        self.dataset_columns_map = {}
        self.column_index_map = {}
        self.dataset_constraint_map = {}

    def _create_dataset_from_geometry(self, geometry, geom_type="domain"):
        """create dataset from geometry."""
        dataset_instance = _geomdata_dict[geom_type](geometry)
        return dataset_instance

    def _get_all_datasets(self):
        """get all datasets"""
        if self.geometry_dict is not None:
            for geom, types in self.geometry_dict.items():
                for geom_type in types:
                    dataset = self._create_dataset_from_geometry(geom, geom_type)
                    self.all_datasets.append(dataset)

        if self.existed_data_list is not None:
            for data_config in self.existed_data_list:
                dataset = ExistedDataset(data_config=data_config)
                self.all_datasets.append(dataset)

    def _merge_all_datasets(self, shuffle=True, num_parallel_workers=1, num_shards=1, shard_id=0,
                            python_multiprocessing=False):
        """merge all datasets"""
        self._iterable_datasets = _IterableDatasets(self.all_datasets)
        dataset = ds.GeneratorDataset(source=self._iterable_datasets,
                                      column_names=self.columns_list,
                                      shuffle=shuffle,
                                      num_parallel_workers=num_parallel_workers,
                                      num_shards=num_shards,
                                      shard_id=shard_id,
                                      python_multiprocessing=python_multiprocessing
                                      )
        return dataset

    def create_dataset(self,
                       batch_size=1,
                       preprocess_fn=None,
                       input_output_columns_map=None,
                       shuffle=True,
                       drop_remainder=True,
                       prebatched_data=False,
                       num_parallel_workers=1,
                       num_shards=None,
                       shard_id=None,
                       python_multiprocessing=False):
        """
        create the final mindspore type dataset to merge all the sub-datasets.

        Args:
            batch_size (int, optional): An int number of rows each batch is created with. Default: 1.
            preprocess_fn (Union[list[TensorOp], list[functions]], optional): List of operations to be
                applied on the dataset. Operations are applied in the order they appear in this list. Default: None.
            input_output_columns_map (dict, optional): specifies which columns to replace and to what.
                The key is the column name to be replaced and the value is the name you want to replace with.
                There's no need to set this argument if all columns are not changed after mapping. Default: None.
            shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is
                required. Default: True, expected order behavior shown in the table.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node. Default: True.
            prebatched_data (bool, optional): Generate pre-batched data before create mindspore dataset. If True,
                pre-batched data will be returned when get each sub-dataset data by index. Else, the batch operation
                will be done by mindspore dataset interface: dataset.batch. When batch_size is very large, it's
                recommended to set this option to be True in order to improve performance on host. Default: False.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: 1.
            num_shards (int, optional): Number of shards that the dataset will be divided into.
                Random accessible input is required. When this argument is specified, `num_samples` reflects the
                maximum sample number of per shard. Default: None.
            shard_id (int, optional): The shard ID within num_shards. This argument must be specified
                only when num_shards is also specified. Random accessible input is required. Default: None.
            python_multiprocessing (bool, optional): Parallelize Python function per_batch_map with multi-processing.
                This option could be beneficial if the function is computational heavy. Default: False.

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> data = dataset.create_dataset()
        """
        self._get_all_datasets()
        _check_type(prebatched_data, "prebatched_data", bool)
        _check_type(drop_remainder, "drop_remainder", bool)
        _check_type(shuffle, "shuffle", bool)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("The type of batch_size should be int, but got {}".format(type(batch_size)))

        if prebatched_data and not drop_remainder:
            raise ValueError("prebatched_data is not supported when drop_remained is set to be False")
        for dataset in self.all_datasets:
            prebatch_size = batch_size if prebatched_data else 1
            prebatch_shuffle = shuffle if prebatched_data else False
            dataset._initialization(batch_size=prebatch_size, shuffle=prebatch_shuffle)
            self.columns_list = dataset.columns_list if self.columns_list is None else self.columns_list + \
                                                                                       dataset.columns_list
            logger.info("Check initial all dataset, dataset: {}, columns_list: {}, data_size: {}".format(
                dataset.name, dataset.columns_list, len(dataset)))

        dataset = self._merge_all_datasets(shuffle=False if prebatched_data else shuffle,
                                           num_parallel_workers=num_parallel_workers,
                                           num_shards=num_shards,
                                           shard_id=shard_id,
                                           python_multiprocessing=python_multiprocessing)
        logger.info("Initial dataset size: {}".format(dataset.get_dataset_size()))
        logger.info("Get all dataset columns names: {}".format(self.columns_list))

        self.dataset_columns_map, self.dataset_constraint_map, self.column_index_map = self._create_trace_maps()
        logger.info("Dataset columns map: {}".format(self.dataset_columns_map))
        logger.info("Dataset column index map: {}".format(self.column_index_map))
        logger.info("Dataset constraints map: {}".format(self.dataset_constraint_map))

        if preprocess_fn is not None:
            input_columns = copy.deepcopy(self.columns_list)
            _check_type(input_output_columns_map, "input_output_columns_map", (type(None), dict))
            if input_output_columns_map is not None:
                new_columns_list, new_dataset_columns_map = self._update_columns_list(input_output_columns_map)
                self.columns_list = new_columns_list
                self.dataset_columns_map = new_dataset_columns_map
                self.column_index_map = {}
                for i in range(len(self.columns_list)):
                    self.column_index_map[self.columns_list[i]] = i
                logger.info("Dataset columns map after preprocess: {}".format(self.dataset_columns_map))
                logger.info("Dataset column index after preprocess: {}".format(self.column_index_map))
                logger.info("Dataset constraints after preprocess: {}".format(self.dataset_constraint_map))
            output_columns = self.columns_list

            dataset = dataset.map(operations=preprocess_fn,
                                  input_columns=input_columns,
                                  output_columns=output_columns,
                                  column_order=output_columns,
                                  num_parallel_workers=num_parallel_workers,
                                  python_multiprocessing=python_multiprocessing)
            logger.info("Get all dataset columns names after preprocess: {}".format(self.columns_list))

        if not prebatched_data:
            dataset = dataset.batch(batch_size=batch_size,
                                    drop_remainder=drop_remainder,
                                    num_parallel_workers=num_parallel_workers)
        logger.info("Final dataset size: {}".format(dataset.get_dataset_size()))
        return dataset

    def _update_columns_list(self, input_output_columns_map):
        """update columns list"""
        new_dataset_columns_map = {}
        for dataset in self.all_datasets:
            columns_list = dataset.columns_list
            new_dataset_columns_map[dataset.name] = []
            for column in columns_list:
                if column in input_output_columns_map.keys():
                    new_column = input_output_columns_map[column]
                    if isinstance(new_column, list):
                        new_dataset_columns_map[dataset.name] += new_column
                    else:
                        new_dataset_columns_map[dataset.name].append(new_column)
                else:
                    new_dataset_columns_map[dataset.name].append(column)

        new_columns_list = []
        for name in new_dataset_columns_map:
            new_columns_list += new_dataset_columns_map[name]
        return new_columns_list, new_dataset_columns_map

    def _create_trace_maps(self):
        """create trace maps"""
        dataset_columns_map = {}
        dataset_constraint_map = {}
        column_index_map = {}
        for dataset in self.all_datasets:
            name = dataset.name
            dataset_columns_map[name] = dataset.columns_list
            dataset_constraint_map[name] = dataset.constraint_type

        for i in range(len(self.columns_list)):
            column_index_map[self.columns_list[i]] = i
        return dataset_columns_map, dataset_constraint_map, column_index_map

    def get_columns_list(self):
        """get columns list

        Args:

        Returns:
            list[str]. column names list of the final unified dataset.

        Examples:
            >>> columns_list = dataset.get_columns_list()
        """
        if self.columns_list is None:
            raise ValueError("Please call create_dataset() first before get final columns list to avoid unexpected "
                             "error")
        return self.columns_list

    def set_constraint_type(self, constraint_type="Equation"):
        """set constraint type of dataset

        Args:
            constraint_type (Union[str, dict): The constraint type of specified dataset. If is string, the constraint
                type of all subdataset will be set to the same one. If is dict, the subdataset and it's constraint type
                is specified by the pair (key, value). Default: "Equation".

        Examples:
            >>> dataset.set_constraint_type("Equation")
        """
        if isinstance(constraint_type, str):
            logger.warning("Argument constraint_type: {} is str, the same type will be set for all of the sub-datasets"
                           .format(constraint_type))
            for datasets in self.all_datasets:
                datasets.set_constraint_type(constraint_type)
        elif isinstance(constraint_type, dict):
            for dataset in constraint_type.keys():
                if dataset not in self.all_datasets:
                    raise ValueError("Unknown dataset: {}. All sub-dataset are: {}".format(
                        dataset, [data.name for data in self.all_datasets]))
                dataset.set_constraint_type(constraint_type[dataset])
        else:
            raise TypeError("the type of constraint_type should be dict or str but got {}"
                            .format(type(constraint_type)))

    def __getitem__(self, index):
        if self._iterable_datasets is None:
            raise ValueError("Please call create_dataset() first before getting item by index to avoid unexpected "
                             "error")
        return self._iterable_datasets[index]

    def __len__(self):
        if self._iterable_datasets is None:
            raise ValueError("Please call create_dataset() first before getting item by index to avoid unexpected "
                             "error")
        return len(self._iterable_datasets)


class _IterableDatasets():
    """get data iteratively"""
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        dataset_size = [len(dataset) for dataset in dataset_list]
        logger.info("Get all dataset sizes: {}".format(dataset_size))
        self.longest = max(dataset_size)

    def __getitem__(self, index):
        col_data = None
        for dataset_instance in self.dataset_list:
            item = dataset_instance[index]
            col_data = item if col_data is None else col_data + item
        return col_data

    def __len__(self):
        return self.longest


def _check_type(param, param_name, param_type):
    """check type"""
    if not isinstance(param, param_type):
        raise TypeError("The type of {} should be instance of {}, but got {}".format(
            param_name, param_type, type(param)))


def _check_key_value(geometry_dict):
    """check key and value of specified dict"""
    for geom in geometry_dict.keys():
        logger.info("check type: {}".format(type(geom)))
        _check_type(geom, "key of geometry_dict", Geometry)

        for geom_type in geometry_dict[geom]:
            if geom_type not in _geomdata_dict.keys():
                raise KeyError("Unknown geom_type: {}, only {} are supported".format(
                    geom_type, ", ".join(_geomdata_dict.keys())))

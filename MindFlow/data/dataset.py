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


from mindscience.data.flow.geometry import Geometry
from mindscience.utils.check_func import check_param_type, check_dict_type_value

from .data_base import Data, ExistedDataConfig
from .existed_data import ExistedDataset
from .equation import Equation
from .boundary import BoundaryIC, BoundaryBC

_geomdata_dict = {
    "domain": Equation,
    "IC": BoundaryIC,
    "BC": BoundaryBC,
}


class Dataset(Data):
    r"""
    Combine datasets together.
    """

    def __init__(self, geometry_dict=None, existed_data_list=None, dataset_list=None):
        super().__init__()
        if all((geometry_dict is None, existed_data_list is None, dataset_list is None)):
            raise ValueError("Dataset should have at least one sub-dataset, but got None")

        if geometry_dict is not None:
            check_param_type(geometry_dict, "geometry_dict", data_type=dict)
            check_dict_type_value(
                geometry_dict,
                "geometry_dict",
                key_type=Geometry,
                value_type=str,
                value_value=list(_geomdata_dict.keys())
            )

        if existed_data_list is not None:
            if isinstance(existed_data_list, ExistedDataConfig):
                existed_data_list = [existed_data_list]
            check_param_type(existed_data_list, "existed_data_list", (list, tuple))

            for data_config in existed_data_list:
                check_param_type(data_config, "element in existed_data_list", ExistedDataConfig)

        if dataset_list is not None:
            if isinstance(dataset_list, Data):
                dataset_list = [dataset_list]
            check_param_type(dataset_list, "dataset_list", (list, tuple))
            for dataset in dataset_list:
                check_param_type(dataset, "element in dataset_list", Data)

        self.existed_data_list = existed_data_list
        self.geometry_dict = geometry_dict
        self.dataset_list = dataset_list
        self.all_datasets = dataset_list if dataset_list else []
        self.columns_list = None
        self._iterable_datasets = None

        self.num_dataset = len(dataset_list) if dataset_list else 0
        if existed_data_list:
            self.num_dataset += len(existed_data_list)
        if geometry_dict:
            for geom in geometry_dict:
                self.num_dataset += len(geometry_dict[geom])

        logger.info(f"Total datasets number: {self.num_dataset}")
        self.dataset_columns_map = {}
        self.column_index_map = {}
        self.dataset_constraint_map = {}

    def _create_dataset_from_geometry(self, geometry, geom_type="domain"):
        """create dataset from geometry."""
        dataset_instance = _geomdata_dict.get(geom_type)(geometry)
        return dataset_instance

    def _get_all_datasets(self):
        """get all datasets"""
        if self.geometry_dict:
            for geom, types in self.geometry_dict.items():
                for geom_type in types:
                    dataset = self._create_dataset_from_geometry(geom, geom_type)
                    self.all_datasets.append(dataset)

        if self.existed_data_list:
            for data_config in self.existed_data_list:
                dataset = ExistedDataset(data_config=data_config)
                self.all_datasets.append(dataset)

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
                       python_multiprocessing=False,
                       sampler=None):
        """
        create the final mindspore type dataset to merge all the sub-datasets.
        """
        self._get_all_datasets()
        check_param_type(prebatched_data, "prebatched_data", data_type=bool)
        check_param_type(drop_remainder, "drop_remainder", data_type=bool)
        check_param_type(shuffle, "shuffle", data_type=bool)
        check_param_type(batch_size, "batch_size", data_type=int, exclude_type=bool)

        if prebatched_data and not drop_remainder:
            raise ValueError(
                "prebatched_data is not supported when drop_remainder is set to be False"
            )

        for dataset in self.all_datasets:
            prebatch_size = batch_size if prebatched_data else 1
            prebatch_shuffle = shuffle if prebatched_data else False
            dataset._initialization(batch_size=prebatch_size, shuffle=prebatch_shuffle)
            self.columns_list = (
                dataset.columns_list if not self.columns_list
                else self.columns_list + dataset.columns_list
            )
            logger.info(
                f"Check initial all dataset, dataset: {dataset.name}, "
                f"columns_list: {dataset.columns_list}, data_size: {len(dataset)}"
            )

        dataset = self._merge_all_datasets(
            shuffle=False if prebatched_data else shuffle,
            num_parallel_workers=num_parallel_workers,
            num_shards=num_shards,
            shard_id=shard_id,
            python_multiprocessing=python_multiprocessing
        )
        logger.info(f"Initial dataset size: {dataset.get_dataset_size()}")
        logger.info(f"Get all dataset columns names: {self.columns_list}")

        self.dataset_columns_map, self.dataset_constraint_map, self.column_index_map = \
            self._create_trace_maps()
        logger.info(f"Dataset columns map: {self.dataset_columns_map}")
        logger.info(f"Dataset column index map: {self.column_index_map}")
        logger.info(f"Dataset constraints map: {self.dataset_constraint_map}")

        if sampler:
            logger.info("Dataset uses sampler")
            dataset.use_sampler(sampler)

        if preprocess_fn:
            input_columns = copy.deepcopy(self.columns_list)
            check_param_type(
                input_output_columns_map,
                "input_output_columns_map",
                (type(None), dict)
            )
            if input_output_columns_map:
                new_columns_list, new_dataset_columns_map = self._update_columns_list(
                    input_output_columns_map
                )
                self.columns_list = new_columns_list
                self.dataset_columns_map = new_dataset_columns_map
                self.column_index_map = {}
                for idx, name in enumerate(self.columns_list):
                    self.column_index_map[name] = idx
                logger.info(
                    f"Dataset columns map after preprocess: {self.dataset_columns_map}"
                )
                logger.info(
                    f"Dataset column index after preprocess: {self.column_index_map}"
                )
                logger.info(
                    f"Dataset constraints after preprocess: {self.dataset_constraint_map}"
                )
            output_columns = self.columns_list

            dataset = dataset.map(
                operations=preprocess_fn,
                input_columns=input_columns,
                output_columns=output_columns,
                num_parallel_workers=num_parallel_workers,
                python_multiprocessing=python_multiprocessing
            )
            dataset = dataset.project(output_columns)
            logger.info(
                f"Get all dataset columns names after preprocess: {self.columns_list}"
            )

        if not prebatched_data:
            dataset = dataset.batch(
                batch_size=batch_size,
                drop_remainder=drop_remainder,
                num_parallel_workers=num_parallel_workers
            )
        logger.info(f"Final dataset size: {dataset.get_dataset_size()}")
        return dataset

    def _merge_all_datasets(self, shuffle=True, num_parallel_workers=1, num_shards=1,
                            shard_id=0, python_multiprocessing=False):
        """merge all datasets"""
        self._iterable_datasets = _IterableDatasets(self.all_datasets)
        dataset = ds.GeneratorDataset(
            source=self._iterable_datasets,
            column_names=self.columns_list,
            shuffle=shuffle,
            num_parallel_workers=num_parallel_workers,
            num_shards=num_shards,
            shard_id=shard_id,
            python_multiprocessing=python_multiprocessing
        )
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
                        new_dataset_columns_map.get(dataset.name).append(new_column)
                else:
                    new_dataset_columns_map.get(dataset.name).append(column)

        new_columns_list = []
        for _, columns in new_dataset_columns_map.items():
            new_columns_list += columns
        return new_columns_list, new_dataset_columns_map

    def get_columns_list(self):
        """
        get columns list

        Returns:
            list[str]. column names list of the final unified dataset.
        """
        if not self.columns_list:
            raise ValueError(
                "Please call create_dataset() first before get final columns list to avoid unexpected error"
            )
        return self.columns_list

    def _create_trace_maps(self):
        """create trace maps"""
        dataset_columns_map = {}
        dataset_constraint_map = {}
        column_index_map = {}
        for dataset in self.all_datasets:
            name = dataset.name
            dataset_columns_map[name] = dataset.columns_list
            dataset_constraint_map[name] = dataset.constraint_type

        for i, column in enumerate(self.columns_list):
            column_index_map[column] = i
        return dataset_columns_map, dataset_constraint_map, column_index_map

    def __getitem__(self, index):
        if not self._iterable_datasets:
            raise ValueError(
                "Call create_dataset() before getting item by index to avoid unexpected error"
            )
        return self._iterable_datasets[index]

    def set_constraint_type(self, constraint_type="Equation"):
        """set constraint type of dataset"""
        if isinstance(constraint_type, str):
            logger.warning(
                f"Argument constraint_type: {constraint_type} is str, "
                "the same type will be set for all of the sub-datasets"
            )
            for datasets in self.all_datasets:
                datasets.set_constraint_type(constraint_type)
        elif isinstance(constraint_type, dict):
            for dataset in constraint_type.keys():
                if dataset not in self.all_datasets:
                    raise ValueError(
                        f"Unknown dataset: {dataset}. All sub-dataset are: "
                        f"{[data.name for data in self.all_datasets]}"
                    )
                dataset.set_constraint_type(constraint_type[dataset])
        else:
            raise TypeError(
                f"the type of constraint_type should be dict or str but got {type(constraint_type)}"
            )

    def __len__(self):
        if not self._iterable_datasets:
            raise ValueError(
                "Call create_dataset() before getting item by index to avoid unexpected error"
            )
        return len(self._iterable_datasets)


class _IterableDatasets:
    """get data iteratively"""

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        dataset_size = [len(dataset) for dataset in dataset_list]
        logger.info(f"Get all dataset sizes: {dataset_size}")
        self.longest = max(dataset_size)

    def __getitem__(self, index):
        col_data = None
        for dataset_instance in self.dataset_list:
            item = dataset_instance[index]
            col_data = col_data + item if col_data else item
        return col_data

    def __len__(self):
        return self.longest

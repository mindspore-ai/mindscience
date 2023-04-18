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

import mindspore.dataset as ds
from mindspore import log as logger

from .data_base import Data, CONSTRAINT_TYPES
from ..utils.check_func import check_param_type, check_param_value, check_dict_type, check_dict_type_value


class MindDataset(Data):
    """
    Create dataset from MindRecord-type data.

    Args:
        dataset_files (Union[str, list[str]]): If `dataset_file` is a str, it represents for
            a file name of one component of a mindrecord source, other files with identical source
            in the same path will be found and loaded automatically. If `dataset_file` is a list,
            it represents for a list of dataset files to be read directly.
        dataset_name (str, optional): name of dataset. Default: ``"dataset_name"``.
        constraint_type (str, optional): constraint type of the specified dataset to get it's corresponding loss
            function. Default: ``"Label"``. Other supported types can be found in `mindflow.data.Dataset`.
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            If `shuffle` is ``False``, no shuffling will be performed.
            If `shuffle` is ``True``, performs global shuffle. Default: True.
            Otherwise, there are two levels of shuffling:

            - ``Shuffle.GLOBAL``: Shuffle both the files and sample.
            - ``Shuffle.FILES``: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None``.
            When this argument is specified, 'num_samples' reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards`. Default: ``None``. This
            argument can only be specified when `num_shards` is also specified.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset. Default: ``None``, sampler is exclusive
            with shuffle and block_reader. Support list: ``SubsetRandomSampler``,
            ``PkSampler``, ``RandomSampler``, ``SequentialSampler``, ``DistributedSampler``.
        num_samples (int, optional): The number of samples to be included in the dataset. Default: ``None``,
            all samples.
        num_parallel_workers (int, optional): The number of readers. Default: ``None``.

    Raises:
        ValueError: If `dataset_files` are not valid or do not exist.
        TypeError: If `dataset_name` is not string.
        ValueError: If `constraint_type.lower()` not in [``"equation"``, ``"bc"``, ``"ic"``, ``"label"``,
            ``"function"``, ``"custom"``].
        RuntimeError: If `num_shards` is specified but `shard_id` is ``None``.
        RuntimeError: If `shard_id` is specified but `num_shards` is ``None``.
        ValueError: If `shard_id` is invalid (< 0 or >= `num_shards`).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.data import MindDataset
        >>> dataset_files = ["./data_dir"] # contains 1 or multiple MindRecord files
        >>> dataset = MindDataset(dataset_files=dataset_files)
    """
    def __init__(self, dataset_files, dataset_name="dataset", constraint_type="Label", shuffle=True,
                 num_shards=None, shard_id=None, sampler=None, num_samples=None, num_parallel_workers=None):
        super(MindDataset, self).__init__()
        check_param_type(dataset_name, "dataset_name", data_type=str)
        check_param_type(constraint_type, "constraint_type", data_type=str)
        check_param_value(constraint_type.lower(), "constraint_type", CONSTRAINT_TYPES)
        self.mind_dataset = ds.MindDataset(dataset_files,
                                           num_parallel_workers=num_parallel_workers,
                                           shuffle=shuffle,
                                           num_shards=num_shards,
                                           shard_id=shard_id,
                                           sampler=sampler,
                                           num_samples=num_samples)
        self.columns_list = self.mind_dataset.get_col_names()
        logger.info("Get MindRecord dataset with columns: {}".format(self.columns_list))
        self.dataset_columns_map = {dataset_name: self.columns_list}
        for i in range(len(self.columns_list)):
            self.column_index_map[self.columns_list[i]] = i
        self.dataset_constraint_map = {dataset_name: constraint_type}
        self.dataset_name = [dataset_name]

    def split_dataset(self, dataset_dict, constraint_dict=None):
        """split the original dataset in order to set difference loss functions.

        Args:
            dataset_dict (dict): dictionary of each sub-dataset, the key is the labeled name while the value
                refers to the specified columns contained in the sub-dataset.
            constraint_dict (Union[None, str, dict]): The constraint type of specified dataset. If ``None``,
                "Label" will be set for all. If is string, all will be set to the same one. If is dict,
                the subdataset and it's constraint type is specified by the pair (key, value). Default: ``None``.

        Examples:
            >>> dataset.split_dataset({"Equation" : "inner_points", "BC" : "bc_points"})
        """
        check_dict_type_value(dataset_dict, "sub-dataset dict", key_type=str, value_type=str,
                              value_value=self.columns_list)
        if constraint_dict:
            check_dict_type(constraint_dict, "sub-constraint dict", key_type=str, value_type=str)
            for key, value in constraint_dict.items():
                if value.lower() not in CONSTRAINT_TYPES:
                    raise ValueError("constraint type for dataset {} should be in {}, but got {}".format(
                        key, CONSTRAINT_TYPES, value))
            if dataset_dict.keys() != constraint_dict.keys():
                raise ValueError("The sub-dataset name should be consistent, but got dataset_dict: {},"
                                 "while constraint_dict: {}".format(dataset_dict.keys(), constraint_dict.keys()))

        self.dataset_columns_map = dataset_dict
        if constraint_dict:
            self.dataset_constraint_map = constraint_dict
        else:
            self.dataset_columns_map.clear()
            for sub_data in self.dataset_columns_map.keys():
                self.dataset_constraint_map[sub_data] = "Label"
        self.dataset_name = list(dataset_dict.keys())

    def set_constraint_type(self, constraint_type="Equation"):
        """set constraint type of dataset

        Args:
            constraint_type (Union[str, dict]): The constraint type of specified dataset. If is string, the constraint
                type of all subdataset will be set to the same one. If is dict, the subdataset and it's constraint type
                is specified by the pair (key, value).

        Examples:
            >>> dataset.set_constraint_type("Equation")
        """
        check_param_type(constraint_type, "constraint_type", data_type=(str, dict))
        if isinstance(constraint_type, str):
            check_param_value(constraint_type.lower(), "constraint_type", CONSTRAINT_TYPES)
            for dataset in self.dataset_name:
                self.dataset_constraint_map[dataset] = constraint_type
        else:
            for dataset in self.dataset_name:
                if dataset not in constraint_type:
                    raise ValueError("constraint type of dataset {} was not defined in constraint_type {}".format(
                        dataset, constraint_type))
                self.dataset_columns_map[dataset] = constraint_type[dataset]

    def create_dataset(self,
                       batch_size=1,
                       preprocess_fn=None,
                       updated_columns_list=None,
                       drop_remainder=True,
                       prebatched_data=False,
                       num_parallel_workers=1,
                       python_multiprocessing=False):
        """
        create the final mindspore type dataset.

        Args:
            batch_size (int, optional): An int number of rows each batch is created with. Default: ``1``.
            preprocess_fn (Union[list[TensorOp], list[functions]], optional): List of operations to be
                applied on the dataset. Operations are applied in the order they appear in this list.
                Default: ``None``.
            updated_columns_list (list, optional): List of columns to be applied on the dataset.
                Default: ``None``.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size. If ``True``, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node. Default: ``True``.
            prebatched_data (bool, optional): Generate pre-batched data before data preprocessing.
                Default: ``False``.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel.
                Default: 1.
            python_multiprocessing (bool, optional): Parallelize Python function per_batch_map with multi-processing.
                This option could be beneficial if the function is computational heavy.
                Default: ``False``.

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> data = dataset.create_dataset()
        """
        check_param_type(prebatched_data, "prebatched_data", data_type=bool)
        check_param_type(drop_remainder, "drop_remainder", data_type=bool)
        check_param_type(batch_size, "batch_size", data_type=int, exclude_type=bool)

        dataset = self.mind_dataset
        if prebatched_data:
            dataset = dataset.batch(batch_size=batch_size,
                                    drop_remainder=drop_remainder,
                                    num_parallel_workers=num_parallel_workers)

        if preprocess_fn:
            dataset = dataset.map(operations=preprocess_fn,
                                  input_columns=self.columns_list,
                                  output_columns=updated_columns_list,
                                  column_order=updated_columns_list,
                                  num_parallel_workers=num_parallel_workers,
                                  python_multiprocessing=python_multiprocessing)
            if updated_columns_list:
                self.columns_list = updated_columns_list
                dataset_name = "data"
                self.dataset_columns_map = {dataset_name: self.columns_list}
                for i in range(len(self.columns_list)):
                    self.column_index_map[self.columns_list[i]] = i

        if not prebatched_data:
            dataset = dataset.batch(batch_size=batch_size,
                                    drop_remainder=drop_remainder,
                                    num_parallel_workers=num_parallel_workers)
        logger.info("Final dataset size: {}".format(dataset.get_dataset_size()))
        logger.info("Dataset columns map after preprocess: {}".format(self.dataset_columns_map))
        logger.info("Dataset column index after preprocess: {}".format(self.column_index_map))
        logger.info("Dataset constraints after preprocess: {}".format(self.dataset_constraint_map))
        return dataset

    def get_columns_list(self):
        """get columns list

        Returns:
            list[str]. column names list of the final unified dataset.

        Examples:
            >>> columns_list = dataset.get_columns_list()
        """
        return self.columns_list

    def __getitem__(self, index):
        return list(self.mind_dataset.create_tuple_iterator())[index]

    def __len__(self):
        return self.mind_dataset.get_dataset_size()

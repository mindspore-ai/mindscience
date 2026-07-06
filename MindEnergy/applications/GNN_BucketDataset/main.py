#!/usr/bin/env python3

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
"""Main entry for bucket dataset generator"""

import h5py
import numpy as np
import mindspore.dataset as ds
from mindspore import set_seed
from mindspore.communication import get_rank, get_group_size
from typing import Dict, Optional, Any
from src import BucketDatasetGenerator, DataGenerator
from src.tools import load_yaml, load_npy

set_seed(0)
np.random.seed(0)

CURRENT_MODE = 'TEST'


def get_dataset(dataset_cfg: Dict[str, Any],
                model_cfg: Dict[str, Any],
                parameter_cfg: Optional[Dict[str, Any]] = None,
                parallel: bool = False,
                prefetch_size: int = 256) -> ds.GeneratorDataset:
    """
    Create a MindSpore GeneratorDataset based on the provided configuration.

    For training task, the dataset is wrapped with BucketDatasetGenerator to enable
    dynamic bucket batching. For test/infer tasks, standard fixed-size batching is used.

    Args:
        dataset_cfg: Configuration dictionary for the dataset.
        model_cfg: Model configuration dictionary.
        parameter_cfg: Optional parameter configuration dictionary.
        parallel: Whether to use distributed parallel data loading. If True, data is
            sharded across ranks using MindSpore communication APIs.
        prefetch_size: Number of data items to prefetch for performance optimization.

    Returns:
        A MindSpore GeneratorDataset (or bucket-based dataset for training) ready for
        iteration.
    """
    print('Start Load Dataset')

    file_path = dataset_cfg["path"]
    columns = list(dataset_cfg["columns"])
    task = dataset_cfg["role"]
    num_workers = int(dataset_cfg["num_workers"])
    batch_size = dataset_cfg["batch_size"]
    shuffle = dataset_cfg["shuffle"]
    drop_remainder = dataset_cfg["drop_remainder"]
    dataset_extras = dataset_cfg.get("extras", {})
    x_features = model_cfg["x_features"]
    output_features = model_cfg["output"]
    parameter_cfg = parameter_cfg or {}

    if parallel:
        rank_id = get_rank()
        rank_size = get_group_size()
    else:
        rank_id = 0
        rank_size = 1

    ds.config.set_prefetch_size(prefetch_size)

    if CURRENT_MODE == 'TEST':
        data = get_dict_data()
    else:
        data = load_data(file_path)

    generator = DataGenerator(
        data=data,
        task=task,
        x_features=x_features,
        output_features=output_features,
        extras=dataset_extras
    )

    dataset = ds.GeneratorDataset(
        source=generator,
        column_names=columns,
        shard_id=rank_id,
        num_shards=rank_size,
        num_parallel_workers=num_workers,
        shuffle=shuffle
    )

    print(f'Task: {generator.task}')

    columns = columns + ['batch_id']

    if task == 'train':
        columns = columns + ['batch_size']
        sample_num_by_node = dataset_cfg["sample_num_by_node"]
        bucket_boundaries = dataset_cfg["bucket_boundaries"]
        bucket_batch_size = dataset_cfg["bucket_batch_size"]
        padding_indices = dataset_cfg["padding_indices"]

        bucket_dataset = BucketDatasetGenerator(dataset, sample_num_by_node, bucket_boundaries, bucket_batch_size,
                                                padding_indices, rank_size)
        bucket_dataloader = ds.GeneratorDataset(bucket_dataset, column_names=columns, shuffle=False)

        print(
            f'Dataset Bucket Boundary: {bucket_dataset.bucket_boundaries}, Batch Num(estimated): {bucket_dataset.dataset_batch_num}')
        return bucket_dataloader
    elif task == 'test':
        dataset = dataset.batch(
            batch_size,
            drop_remainder=drop_remainder,
            per_batch_map=generator.test_collate,
            output_columns=columns
        )
    else:
        dataset = dataset.batch(
            batch_size,
            drop_remainder=drop_remainder,
            per_batch_map=generator.infer_collate,
            output_columns=columns
        )
    return dataset


def load_data(file_path):
    is_h5_file = file_path.endswith('.h5')
    if is_h5_file:
        file = h5py.File(file_path, 'r')
    else:
        file = load_npy(file_path)

    return file


def get_dict_data():
    result = {}
    sample_shape = {5: 20, 11: 50, 15: 50, 25: 80}
    result['x0'] = [np.random.random((samples, 7 * nodes)).tolist() for nodes, samples in sample_shape.items()]
    result['node_feature1'] = [np.random.random((samples, nodes)).tolist() for nodes, samples in sample_shape.items()]
    result['node_feature2'] = [np.random.random((samples, nodes)).tolist() for nodes, samples in sample_shape.items()]
    result['node_feature3'] = [np.random.random((samples, nodes)).tolist() for nodes, samples in sample_shape.items()]
    result['edge_feature1'] = [np.random.random((samples, nodes)).tolist() for nodes, samples in sample_shape.items()]
    result['edge_feature2'] = [np.random.random((samples, nodes)).tolist() for nodes, samples in sample_shape.items()]
    result['x_vd'] = [np.random.random((samples, 3 * nodes)).tolist() for nodes, samples in sample_shape.items()]

    return result


if __name__ == "__main__":
    yaml_path = "./configs/config.yaml"
    params = load_yaml(yaml_path)

    model_params = params['model']
    parameter_params = params['parameter']
    train_ds_cfg = params['data']['train']
    test_ds_cfg = params['data']['test']
    infer_ds_cfgs = params['data']['infer']

    train_dataset = get_dataset(
        train_ds_cfg,
        model_params,
        parameter_params,
        parallel=True
    )

    print("Create bucket train dataset successfully")

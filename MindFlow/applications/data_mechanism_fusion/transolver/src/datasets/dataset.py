# Copyright 2025 Huawei Technologies Co., Ltd
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
Dataset loading and processing for Darcy Flow problems.
"""
import os
import numpy as np
import scipy.io as scio

import mindspore.dataset as ds
from mindspore import log as logger


class DarcyDataset:
    """
    A custom dataset class to load Darcy Flow data from .mat files.
    """
    def __init__(self, file_path: str, ntrain: int, subsampling: int, resolution: int):
        self.file_path = file_path
        self.ntrain = ntrain
        self.subsampling = subsampling
        self.resolution = resolution

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found at: {file_path}. "
                "Please download the 'piececonst_r421_N1024_smooth1.mat' dataset first"
            )

        try:
            data = scio.loadmat(file_path)
            raw_coeff = data['coeff']
            raw_sol = data['sol']

            self.coeff = raw_coeff[:ntrain, ::subsampling, ::subsampling][:, :resolution, :resolution]
            self.coeff = self.coeff.reshape(ntrain, -1, 1)

            self.solution = raw_sol[:ntrain, ::subsampling, ::subsampling][:, :resolution, :resolution]
            self.solution = self.solution.reshape(ntrain, -1, 1)

            x_grid = np.linspace(0, 1, resolution)
            y_grid = np.linspace(0, 1, resolution)
            grid_x, grid_y = np.meshgrid(x_grid, y_grid)

            pos = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
            self.pos = np.tile(pos[np.newaxis, ...], (ntrain, 1, 1))

            logger.info(f"Dataset loaded successfully from {file_path}")

        except Exception as error:
            logger.error(f"Failed to load dataset: {error}")
            raise

    def __getitem__(self, index):
        return (
            self.pos[index].astype(np.float32),
            self.coeff[index].astype(np.float32),
            self.solution[index].astype(np.float32)
        )

    def __len__(self):
        return self.ntrain


def create_dataset(file_path: str,
                   batch_size: int = 8,
                   ntrain: int = 1000,
                   subsampling: int = 5,
                   resolution: int = 64,
                   shuffle: bool = True,
                   num_workers: int = 2):
    """
    Create a MindSpore dataset for Darcy Flow training.
    """
    dataset_generator = DarcyDataset(file_path, ntrain, subsampling, resolution)

    ds_loader = ds.GeneratorDataset(
        source=dataset_generator,
        column_names=["pos", "x", "y"],
        shuffle=shuffle,
        num_parallel_workers=num_workers
    )

    ds_loader = ds_loader.batch(batch_size, drop_remainder=True)

    return ds_loader

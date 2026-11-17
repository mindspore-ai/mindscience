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
"""Data handling utilities"""

import os
import yaml
from typing import Dict
import numpy as np


def load_yaml(file_path: str) -> Dict:
    """
    Load and parse a YAML configuration file.

    Args:
        file_path: Path to the YAML file to be loaded.

    Returns:
        A dictionary containing the parsed YAML configuration.
    """
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def load_npy(dataset_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all .npy files from a directory into a dictionary.

    Each file is keyed by its filename (without extension), and the value
    is the numpy array loaded from that file.

    Args:
        dataset_dir: Path to the directory containing .npy files.

    Returns:
        A dictionary mapping filename stems to their corresponding numpy arrays.
    """
    data_dict = {}

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")

    for filename in os.listdir(dataset_dir):
        if filename.endswith('.npy'):
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(dataset_dir, filename)
            data_dict[key] = np.load(file_path)

    return data_dict

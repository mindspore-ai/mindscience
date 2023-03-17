# Copyright 2022 Huawei Technologies Co., Ltd
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
utility functions
"""
import os
import yaml
import numpy as np


EPS = 1e-8
np.random.seed(0)


def _make_paths_absolute(dir_, config):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Args:
        dir_ (str): The path of yaml configuration file.
        config (dict): The yaml for configuration file.

    Returns:
        Dict. The configuration information in dict format.
    """
    for key in config.keys():
        if key.endswith("_path"):
            config[key] = os.path.join(dir_, config[key])
            config[key] = os.path.abspath(config[key])
        if isinstance(config[key], dict):
            config[key] = _make_paths_absolute(dir_, config[key])
    return config


def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): The path of yaml configuration file.

    Returns:
        Dict. The configuration information in dict format.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``
    """
    # Read YAML experiment definition file
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config = _make_paths_absolute(os.path.dirname(file_path), config)
    return config

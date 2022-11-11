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


def make_paths_absolute(dir_, config):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    config : dict

    Returns
    -------
    config : dict
    """
    for key in config.keys():
        if key.endswith("_path"):
            config[key] = os.path.join(dir_, config[key])
            config[key] = os.path.abspath(config[key])
        if isinstance(config[key], dict):
            config[key] = make_paths_absolute(dir_, config[key])
    return config


def load_config(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    config : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        config = yaml.safe_load(stream)
    config = make_paths_absolute(os.path.join(os.path.dirname(yaml_filepath), ".."), config)
    return config
# Copyright 2023 Huawei Technologies Co., Ltd
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

import datetime
import os
import yaml

import mindspore


def make_dir(path):
    '''make_dir'''
    if os.path.exists(path):
        return
    try:
        permissions = os.R_OK | os.W_OK | os.X_OK
        os.umask(permissions << 3 | permissions)
        mode = permissions << 6
        os.makedirs(path, mode=mode, exist_ok=True)
    except PermissionError as e:
        mindspore.log.critical("No write permission on the directory(%r), error = %r", path, e)
        raise TypeError("No write permission on the directory.") from e
    finally:
        pass


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
    config = _make_paths_absolute(os.path.join(
        os.path.dirname(file_path), ".."), config)
    return config


def get_datapath_from_date(start_date, idx):
    """
    Get data file name of given start date and index of the data.

    Args:
        start_date (datetime.datetime): The start date of data.
        idx (int): The index of data.

    Returns:
        date_file_name (str). The data file name.
        static_file_name (str). The static file name.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> from mindearth.utils import get_datapath_from_date
        >>> date = datetime.datetime(2019, 1, 1, 0, 0, 0)
        >>> idx = 1
        >>> date_file_name, static_file_name = get_datapath_from_date(date, idx)
        >>> print(f"date_file_name: {date_file_name}, static_file_name: {static_file_name}")
        date_file_name: 2019/2019_01_01_2.npy, static_file_name: 2019/2019.npy
    """
    t0 = start_date
    t = t0 + datetime.timedelta(hours=idx)
    year = t.year
    month = t.month
    day = t.day
    hour = t.hour + 1
    date_file_name = f'{year}/{year}_{str(month).zfill(2)}_{str(day).zfill(2)}_{hour}.npy'
    static_file_name = f'{year}/{year}.npy'
    return date_file_name, static_file_name

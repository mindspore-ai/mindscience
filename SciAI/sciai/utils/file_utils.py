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
# ==============================================================================
"""file utils"""
import os


def make_sciai_dirs():
    """
    Create directories for a sciai project. It creates `checkpoints`, `data`, `figures`, `logs` if one doesn't exist.
    `checkpoints` for model checkpoints; `data` for dataset and generated data; `figures` for model plots; `logs` for
    training or evaluation logs.
    """
    dir_names = ["checkpoints", "data", "figures", "logs"]
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)


def _isfile(file_path):
    """
    Returns True if is a file, else returns False
    """
    return '.' in os.path.split(file_path)[1]


def _get_automodel_module_name(model_name):
    """
    Calculate the model name according to
    """
    return f"auto_model_{model_name}"


def _is_folder_non_empty(folder_path):
    """
    Recursively judge whether a folder is non-empty.
    """
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            return True
        if os.path.isdir(item_path) and _is_folder_non_empty(item_path):
            return True
    return False

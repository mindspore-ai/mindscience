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
r"""This module provides a function to load a configuration file."""
from typing import Tuple
from omegaconf import OmegaConf


def load_config(file_path: str) -> Tuple[dict, str]:
    r"""
    Load a configuration file.

    Args:
        file_path (str): The path of yaml configuration file.

    Returns:
        Tuple[dict, str]: The configuration dictionary and its string representation.
    """
    if not file_path.endswith(".yaml"):
        raise ValueError("The configuration file must be a yaml file")

    config = OmegaConf.load(file_path)

    base_config_path = config.get("base_config", "none")
    if base_config_path.lower() != "none":
        config_custom = config
        config = OmegaConf.load(base_config_path)
        config.merge_with(config_custom)

    config_str = OmegaConf.to_yaml(config)

    return config, config_str

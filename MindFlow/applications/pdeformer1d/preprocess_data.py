#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
r"""Preprocessing custom multi_pde data."""
import argparse

from src.data.utils_multi_pde import gen_dag_info_all
from src.utils.load_yaml import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDEformer data.")
    parser.add_argument("--config_file_path", type=str,
                        default="configs/config_yzh_grammar.yaml")
    args = parser.parse_args()
    config, _ = load_config(args.config_file_path)
    gen_dag_info_all(config)

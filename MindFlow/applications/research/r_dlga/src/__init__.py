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
"""init"""
from .utils import random_data, evaluate, cal_grads, cal_terms, pinn_loss_func
from .utils import get_dict_name, get_dicts, calculate_coef, LossArgs
from .utils import get_lefts, update_lib, gene_algorithm, TrainArgs
from .dataset import create_dataset, create_pinn_dataset
from .meta_data import produce_meta_data
from .generalized_ga import BurgersGeneAlgorithm, CylinderFlowGeneAlgorithm, PeriodicHillGeneAlgorithm

__all__ = [
    "random_data",
    "evaluate",
    "cal_grads",
    "cal_terms",
    "pinn_loss_func",
    "get_dict_name",
    "get_dicts",
    "calculate_coef",
    "get_lefts",
    "update_lib",
    "create_dataset",
    "create_pinn_dataset",
    "produce_meta_data",
    "BurgersGeneAlgorithm",
    "CylinderFlowGeneAlgorithm",
    "PeriodicHillGeneAlgorithm",
    "gene_algorithm",
    "TrainArgs",
    "LossArgs"
]

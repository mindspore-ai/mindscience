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
from .dataset import create_dataset
from .utils import calculate_lp_loss_error, make_dir, scheduler, get_param_dic, init_model
from .utils import plot_coe, get_label_coe, plot_test_error, plot_extrapolation_error

__all__ = [
    "create_dataset",
    "calculate_lp_loss_error",
    "make_dir",
    "scheduler",
    "get_param_dic",
    "init_model",
    "plot_coe",
    "get_label_coe",
    "plot_test_error",
    "plot_extrapolation_error"
]

# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""Data"""

from .elements import elements, element_dict, element_name, element_set, atomic_mass
from .hyperparam import str_to_tensor, tensor_to_str
from .hyperparam import get_class_parameters, get_hyper_parameter, get_hyper_string
from .hyperparam import set_class_parameters, set_hyper_parameter, set_class_into_hyper_param
from .hyperparam import load_checkpoint, load_hyperparam, load_hyper_param_into_class
from .template import get_template, get_template_index
from .parameters import ForceFieldParameters

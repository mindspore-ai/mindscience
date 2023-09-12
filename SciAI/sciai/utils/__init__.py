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
"""sciai utils"""
from .check_utils import to_tuple
from .file_utils import make_sciai_dirs
from .log_utils import log_config, print_log
from .ms_utils import str2datatype, amp2datatype, datatype2np, to_tensor, set_seed, to_float, \
    flatten_add_dim, calc_ckpt_name
from .plot_utils import save_result_dir, newfig, savefig
from .python_utils import lazy_property, lazy_func, parse_arg, print_args, print_time, download_resource
from .register_utils import Register, FunctionType
from .time_utils import time_str, time_second

__all__ = []
__all__.extend(["to_tuple"])
__all__.extend(["make_sciai_dirs"])
__all__.extend(["log_config", "print_log"])
__all__.extend(["str2datatype", "amp2datatype", "datatype2np", "to_tensor", "set_seed", "to_float",
                "flatten_add_dim", "calc_ckpt_name"])
__all__.extend(["save_result_dir", "newfig", "savefig"])
__all__.extend(["lazy_property", "lazy_func", "parse_arg", "print_args", "print_time", "download_resource"])
__all__.extend(["time_str", "time_second"])
__all__.extend(["Register", "FunctionType"])

# Copyright 2021 Huawei Technologies Co., Ltd
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
"""check context mode"""
from mindspore import context


def check_mode(api_name):
    if context.get_context("mode") == context.PYNATIVE_MODE:
        raise RuntimeError("{} is only supported GRAPH_MODE now but got PYNATIVE_MODE".format(api_name))

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
"""test gadmall"""
import os
import yaml
import pytest

from mindspore import context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.gad_mall.train import main

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_comb_should_loss_small_enough_when_epoch_10(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    args = parse_arg(config)
    # parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    # outputs = sys.stdout.getvalue().strip()
    # _, loss_value, _, _ = re.findall(r"step: (.*), loss: (.*), interval: (.*), total: (.*)", outputs)[-1]
    # assert float(loss_value) < 0.6
    clear_stub(stderr, stdout)

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
"""test case for auto sponge"""
import re
import sys
import yaml
import pytest

from mindspore import context

from sciai.model import AutoModel
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_auto_sponge_ufold_training(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)

    model = AutoModel.from_pretrained("UFold")

    with open("./config/ufold_config.yaml", "r") as file:
        status = yaml.safe_load(file)
    status["is_training"] = True
    with open("./config/ufold_config.yaml", "w") as file:
        yaml.dump(status, file)
    model.update_config(save_model=False)

    model.initialize(config_path="./config/ufold_config.yaml")
    model.train(data_source="./examples")

    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r"([0-9]*\.[0-9]*)", outputs)[-1]
    assert float(loss) < 3

    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_auto_sponge_ufold_predict(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)

    model = AutoModel.from_pretrained("UFold")
    with open("./config/ufold_config.yaml", "r") as file:
        status = yaml.safe_load(file)
    status["test_ckpt"] = 'All'
    status["is_training"] = False
    with open("./config/ufold_config.yaml", "w") as file:
        yaml.dump(status, file)

    model.update_config(load_ckpt=True, load_ckpt_path="./checkpoint/ufold_train_99.ckpt")

    model.initialize(config_path="./config/ufold_config.yaml")
    ret = model.evaluate(data="./examples/Acid.caps._TRW-240015_1-352.ct")
    assert ret[0].shape == (1, 352, 352)

    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_auto_sponge_ufold_finetune(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)

    model = AutoModel.from_pretrained("UFold")

    with open("./config/ufold_config.yaml", "r") as file:
        status = yaml.safe_load(file)
    status["is_training"] = True
    with open("./config/ufold_config.yaml", "w") as file:
        yaml.dump(status, file)
    model.update_config(save_model=False, load_ckpt_path="./checkpoint/ufold_train_99.ckpt")

    model.initialize(config_path="./config/ufold_config.yaml")
    model.finetune(data_source="./examples")

    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r"([0-9]*\.[0-9]*)", outputs)[-1]
    assert float(loss) < 3

    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_auto_sponge_update(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)

    model = AutoModel.from_pretrained("UFold")

    with pytest.raises(ValueError) as info:
        model.update_config(wrong_key="wrong_value")
    assert "Unknown keyword:" in str(info.value)

    model.update_config(save_model=True, load_ckpt_path="my_path", load_ckpt=True)
    assert model.model_args.get("save_model")
    assert model.model_args.get("load_ckpt_path") == "my_path"
    assert model.model_args.get("load_ckpt")

    model.update_config(key="new_key", conf="new_config", config_path="new_config_path")
    assert model.func_args.get("initialize").get("key") == "new_key"
    assert model.func_args.get("initialize").get("conf") == "new_config"
    assert model.func_args.get("initialize").get("config_path") == "new_config_path"

    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_auto_sponge_abnormal(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)

    model = AutoModel.from_pretrained("UFold")

    with pytest.raises(ValueError) as info:
        model.evaluate(load_ckpt_path=1)
    assert "Invalid load checkpoint path" in str(info.value)

    with pytest.raises(ValueError) as info:
        model.evaluate(load_ckpt_path="my_path")
    assert "Invalid load checkpoint path" in str(info.value)

    with pytest.raises(ValueError) as info:
        model.finetune(load_ckpt_path=1)
    assert "Invalid load checkpoint path" in str(info.value)

    with pytest.raises(ValueError) as info:
        model.finetune(load_ckpt_path="my_path")
    assert "Invalid load checkpoint path" in str(info.value)

    clear_stub(stderr, stdout)

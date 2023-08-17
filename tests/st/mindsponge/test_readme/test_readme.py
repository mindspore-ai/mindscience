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
# ============================================================================
"""Test README EXAMPLES."""
import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_md_run():
    """
    Feature: test md
    Description: None
    Expectation: finish simulation
    """
    os.environ['MS_JIT_MODULES'] = 'sponge'
    from sponge import ForceField, Molecule, Sponge
    from sponge.callback import RunInfo, WriteH5MD
    from mindspore import context
    from mindspore.nn import Adam

    context.set_context(mode=context.GRAPH_MODE)
    system = Molecule(template='water.spce.yaml')

    system.reduplicate([0.3, 0, 0])
    system.reduplicate([0, 0.3, 0])
    new_sys = system.copy([0, 0, -0.3])
    system.reduplicate([0, 0, 0.3])
    system.append(new_sys)

    potential = ForceField(system, parameters='SPCE')

    opt = Adam(system.trainable_params(), 1e-3)

    md = Sponge(system, potential, opt)

    run_info = RunInfo(10)
    cb_h5md = WriteH5MD(system, 'tutorial_b03.h5md', save_freq=10)

    md.run(1000, callbacks=[run_info, cb_h5md])

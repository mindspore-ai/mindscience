# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test parallel attention"""

import os
import pytest

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_tensor_parallel_attention():
    """
    Feature: Tensor-parallel attention

    Description: Launch msrun for the tensor-parallel attention example with 8 workers
    and verify the process exits successfully and logs contain no errors.

    Expectation: msrun returns exit code 0 and worker logs do not contain ERROR lines.
    """
    scripts_name = "run_parallel_attention.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --parallel_type tp"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_tensor_parallel_attention "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_tensor_parallel_attention/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_tensor_parallel_attention/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_ulysses_context_parallel_attention():
    """
    Feature: Ulysses context-parallel attention

    Description: Launch msrun for the ulysses_cp parallel_type with 8 workers
    and verify the process exits successfully and logs contain no errors.

    Expectation: msrun returns exit code 0 and worker logs do not contain ERROR lines.
    """
    scripts_name = "run_parallel_attention.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --parallel_type ulysses_cp"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_ulysses_context_parallel_attention "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_ulysses_context_parallel_attention/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_ulysses_context_parallel_attention/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_colossal_context_parallel_attention():
    """
    Feature: Colossal context-parallel attention

    Description: Launch msrun for the colossal_cp parallel_type with 8 workers
    and verify the process exits successfully and logs contain no errors.

    Expectation: msrun returns exit code 0 and worker logs do not contain ERROR lines.
    """
    scripts_name = "run_parallel_attention.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --parallel_type colossal_cp"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_colossal_context_parallel_attention "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_colossal_context_parallel_attention/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_colossal_context_parallel_attention/worker_*.log"

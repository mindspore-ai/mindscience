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
"""test ColumnParallelLinear and RowParallelLinear"""

import os
import pytest

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_column_parallel_linear():
    """
    Feature: Column-parallel linear layer.

    Description: Launches an 8-worker msrun job that runs the column-parallel linear example.

    Expectation: msrun exits with code 0 and worker logs contain no ERROR entries.
    """
    scripts_name = "run_column_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --gather_output"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_column_parallel_linear "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_column_parallel_linear/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_column_parallel_linear/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_column_parallel_linear_bias():
    """
    Feature: Column-parallel linear layer with bias.

    Description: Runs the example with bias enabled across 8 workers via msrun.

    Expectation: Job exits successfully (code 0) and logs show no ERROR lines.
    """
    scripts_name = "run_column_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --gather_output --bias"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_column_parallel_linear_bias "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_column_parallel_linear_bias/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_column_parallel_linear_bias/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_column_parallel_linear_not_gather_output():
    """
    Feature: Column-parallel linear without gathering output.

    Description: Runs the column-parallel example without gathering outputs from workers.

    Expectation: msrun completes with return code 0 and no ERROR in worker logs.
    """
    scripts_name = "run_column_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num}"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_column_parallel_linear_not_gather_output "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_column_parallel_linear_not_gather_output/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_column_parallel_linear_not_gather_output/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_column_parallel_linear_use_sequence_parallel():
    """
    Feature: Column-parallel linear with sequence parallelism.

    Description: Executes the example using sequence parallelism across 8 workers.

    Expectation: Job finishes with exit code 0 and worker logs contain no ERROR entries.
    """
    scripts_name = "run_column_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --gather_output --use_sequence_parallel"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_column_parallel_linear_use_sequence_parallel "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' \
              {sh_path}/msrun_column_parallel_linear_use_sequence_parallel/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_column_parallel_linear_use_sequence_parallel/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_column_parallel_linear_use_sequence_parallel_not_gather_output():
    """
    Feature: Sequence-parallel column linear without gathering output.

    Description: Runs sequence-parallel example and keeps outputs per-worker (no gather).

    Expectation: msrun returns 0 and worker logs are free of ERROR lines.
    """
    scripts_name = "run_column_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --use_sequence_parallel"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_column_parallel_linear_use_sequence_parallel_not_gather_output "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' \
              {sh_path}/msrun_column_parallel_linear_use_sequence_parallel_not_gather_output/worker_0.log -C 3")
    assert ret == 0, \
        "msrun failed, please check msrun_column_parallel_linear_use_sequence_parallel_not_gather_output/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_row_parallel_linear():
    """
    Feature: Row-parallel linear layer.

    Description: Launches an 8-worker msrun job running the row-parallel linear example.

    Expectation: Process exits with code 0 and no ERROR entries appear in logs.
    """
    scripts_name = "run_row_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num}"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_row_parallel_linear "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_row_parallel_linear/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_row_parallel_linear/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_row_parallel_linear_bias():
    """
    Feature: Row-parallel linear layer with bias.

    Description: Runs the row-parallel example with bias enabled across 8 workers.

    Expectation: msrun exits successfully (code 0) and worker logs contain no ERROR.
    """
    scripts_name = "run_row_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --bias"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_row_parallel_linear_bias "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_row_parallel_linear_bias/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_row_parallel_linear_bias/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_row_parallel_linear_input_is_parallel():
    """
    Feature: Row-parallel linear where input is already parallel.

    Description: Executes the row-parallel example with input pre-partitioned across workers.

    Expectation: Job completes with return code 0 and logs show no ERROR lines.
    """
    scripts_name = "run_row_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --input_is_parallel"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_row_parallel_linear_input_is_parallel "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_row_parallel_linear_input_is_parallel/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_row_parallel_linear_input_is_parallel/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_row_parallel_linear_use_sequence_parallel():
    """
    Feature: Row-parallel linear with sequence parallelism.

    Description: Runs the row-parallel example enabling sequence parallel across workers.

    Expectation: msrun returns 0 and worker logs are free of ERROR entries.
    """
    scripts_name = "run_row_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --use_sequence_parallel"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_row_parallel_linear_use_sequence_parallel "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' \
              {sh_path}/msrun_row_parallel_linear_use_sequence_parallel/worker_0.log -C 3")
    assert ret == 0, "msrun failed, please check msrun_row_parallel_linear_use_sequence_parallel/worker_*.log"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_row_parallel_linear_use_sequence_parallel_input_is_parallel():
    """
    Feature: Sequence-parallel row linear with parallel input.

    Description: Runs the example combining sequence parallelism and parallel input across workers.

    Expectation: Process exits with code 0 and worker logs contain no ERROR lines.
    """
    scripts_name = "run_row_parallel_linear.py"
    device_num = 8

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    scripts_path = os.path.join(sh_path, scripts_name)

    scripts_cmd = f"{scripts_path} --num_workers {device_num} --input_is_parallel --use_sequence_parallel"
    cmd = (
        f"msrun --worker_num={device_num} "
        + f"--local_worker_num={device_num} "
        + "--master_port=8191 "
        + "--log_dir=msrun_row_parallel_linear_use_sequence_parallel_input_is_parallel "
        + "--join=True "
        + "--cluster_time_out=300 "
        + f"{scripts_cmd}"
    )
    ret = os.system(cmd)
    os.system(f"grep -E 'ERROR|error' \
              {sh_path}/msrun_row_parallel_linear_use_sequence_parallel_input_is_parallel/worker_0.log -C 3")
    assert ret == 0, \
        "msrun failed, please check msrun_row_parallel_linear_use_sequence_parallel_input_is_parallel/worker_*.log"

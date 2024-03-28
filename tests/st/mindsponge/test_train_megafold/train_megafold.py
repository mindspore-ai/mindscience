
# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Megafold train test"""
import argparse
import os
import re
import stat


def time_calculate(forward_time_list, backward_time_list):
    '''calculate time cost'''
    total_time = float(backward_time_list[0])
    for i in range(4):
        total_time += float(forward_time_list[i])
    return total_time

def find_info(file_path):
    '''read the information from log'''
    read = open(file_path, "r")
    loss_content = read.read()
    read.close()
    forward_time_list = re.findall(r"forward time :  (\d+\.?\d*)", loss_content)
    backward_time_list = re.findall(r"backward time :  (\d+\.?\d*)", loss_content)
    # pylint: disable=W0621
    compile_time = time_calculate(forward_time_list, backward_time_list)
    execuate_time = time_calculate(forward_time_list[::-1], backward_time_list[::-1])
    loss_list = re.findall(r"total_loss: (\d+\.?\d*)", loss_content)
    last_loss = float(loss_list[-1])
    return compile_time, execuate_time, last_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train test')
    parser.add_argument('--device_type', default="910A", help='device type')
    parser.add_argument('--device_id', default=0, help='device id')
    arguments = parser.parse_args()
    device_type = arguments.device_type
    device_id = arguments.device_id
    os.environ['DEVICE_ID'] = str(device_id)
    cmd_copy = "cp -r ../../../../MindSPONGE/applications/model_cards/examples/MEGA-Protein ./"
    os.system(cmd_copy)
    os.system(f"python3 run.py --device_type {device_type} > megefold_res.log")
    compile_time_res, execuate_time_res, last_loss_res = find_info("megefold_res.log")
    out_res = f"loss: {last_loss_res} \n" \
              f"compile_time: {compile_time_res} \n" \
              f"execute_time: {execuate_time_res} \n"
    os_flags = os.O_RDWR | os.O_CREAT
    os_modes = stat.S_IRWXU
    res_path = f'./Megafold_train_result1.log'
    with os.fdopen(os.open(res_path, os_flags, os_modes), 'w') as fout:
        fout.write(out_res)
    fout.close()

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
"""utilities for sciai tests"""
import os
import subprocess

import mindspore as ms

from sciai.common.train_cell import to_tuple
from sciai.utils import print_log
from sciai.utils.check_utils import _check_type


def tensor_equal(tensor1, tensor2):
    return (tensor1 == tensor2).all()


def tuple_tensor_equal(tuple_tensor1, tuple_tensor2):
    return len(tuple_tensor1) == len(tuple_tensor2) \
           and all(tensor_equal(tuple_tensor1[i], tuple_tensor2[i]) for i in range(len(tuple_tensor1)))


def find_card(space_lb=10000, card_total_num=8, cards=None):
    """
    Find best card with the largest remaining memory.
    Args:
        space_lb (int): Space lower bound. Default: 10000.
        card_total_num (int): Card number on the machine. Default: 8.
        cards (Union[int, tuple[int], None]): Cards to select from. Default: None.
    Returns:
        int: Best card index.
    """
    if cards is not None:
        _check_type(cards, "cards", (int, tuple))
        cards = to_tuple(cards)
        [_check_type(_, "element in cards", int) for _ in cards]  # pylint: disable=W0106
    on_ascend = ms.get_context("device_target") == "Ascend"
    if on_ascend:
        param_dict = {
            "cmd": ["npu-smi", "info"], "start_flag": "+===", "mem_used_start_pos": 53,
            "mem_total_start_pos": 60, "hp_used_start_pos": 71, "hp_total_start_pos": 78, "step_len": 3
        }
    else:
        param_dict = {
            "cmd": "nvidia-smi", "start_flag": "|===", "mem_used_start_pos": 35, "mem_total_start_pos": 46,
            "hp_used_start_pos": 35, "hp_total_start_pos": 46, "step_len": 4
        }
    try:
        res = subprocess.Popen(param_dict["cmd"], stdout=subprocess.PIPE, shell=False).communicate(timeout=10)
    except Exception as _:  # pylint: disable=W0703
        print_log("failed to execute cmd, will use default card 0", enable_log=False)
        return 0
    res_list = str(res[0], 'utf-8').strip().split("\n")
    i = 0
    for line in res_list:
        if line.startswith(param_dict["start_flag"]):
            break
        i += 1
    else:
        print_log("return info format is incorrect! will use default card 0", enable_log=False)
        return 0
    card_space_remain_dict = {}
    for j in range(i + 1, i + card_total_num * param_dict["step_len"], param_dict["step_len"]):
        card_index = int(res_list[j].split()[1])
        mem_space_remain = _find_source(res_list[j + 1], param_dict, "mem_used_start_pos", "mem_total_start_pos")
        hp_space_remain = _find_source(res_list[j + 1], param_dict, "hp_used_start_pos", "hp_total_start_pos")
        card_space_remain_dict[card_index] = mem_space_remain, hp_space_remain
    if cards is None:
        card_space_remain_dict_filtered = card_space_remain_dict
    else:
        card_space_remain_dict_filtered = {k: v for k, v in card_space_remain_dict.items() if k in cards}
    sorted_remain_tuples = sorted(list(card_space_remain_dict_filtered.items()), key=lambda _: -(_[1][0] + _[1][1] / 2))
    best_card, best_remain = sorted_remain_tuples[0]
    if best_remain[0] < space_lb or best_remain[1] < space_lb:
        print_log(f"No card has free space more than {space_lb} MB.", enable_log=False)
    print_log(f"found best card '{best_card}' with free memory/HBM {best_remain}MB.", enable_log=False)
    return best_card


def _find_source(card_space_line, param_dict, used_start_key, total_start_key):
    """
    Find the remaining memory source according to the printing format of `GPU` or `Ascend`.
    Args:
        card_space_line (str): String line containing card space information.
        param_dict (dict): Parameter dictionary.
        used_start_key (str): Used space start flag key in param_dict.
        total_start_key (str): Total space start flag key in param_dict.
    Returns:
         int: Remaining space for the given card.
    """
    used_start_ind = param_dict[used_start_key]
    total_start_index = param_dict[total_start_key]
    card_space_used = float(card_space_line[used_start_ind: used_start_ind + 5])
    card_space_total = float(card_space_line[total_start_index: total_start_index + 5])
    card_space_remain = card_space_total - card_space_used
    return card_space_remain


DATA_PATH = os.getenv("DATASET_PATH", "/home/workspace/mindspore_dataset")
DEFAULT_CI_BASE_PATH = os.path.join(DATA_PATH, "sciai_data")


def copy_dataset(cur_dir, origin_data_path=DEFAULT_CI_BASE_PATH):
    dir_path, model_dir_name = os.path.split(cur_dir)
    origin_data_dir = os.path.join(origin_data_path, model_dir_name)
    cmd_copy = ['cp', '-rf', origin_data_dir, dir_path]
    subprocess.Popen(cmd_copy, stdout=subprocess.PIPE, shell=False).communicate(timeout=100)
    print_log(f"Data copy to current directory successfully.")

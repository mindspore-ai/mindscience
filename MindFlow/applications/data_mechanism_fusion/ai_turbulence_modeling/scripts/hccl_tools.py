# ============================================================================
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
"""
hccl_tools
"""
import json
import os
import socket
import sys
from argparse import ArgumentParser
from typing import Dict, Any


def parse_args():
    """
    parse args .

    Args:

    Returns:
        args.
    """
    parser = ArgumentParser(description="mindspore distributed training launch "
                                        "helper utility that will generate hccl"
                                        " config file")
    parser.add_argument("--device_num", type=str, default="[0,8)",
                        help="The number of the Ascend accelerators used. "
                        "please note that the Ascend accelerators"
                        "used must be continuous, such [0,4) means using four chips "
                        "0，1，2，3; [0,1) means using chip 0; In the most Ascend system, "
                        "the first four chips belong to one group, "
                        "and the last four chips belong to another one."
                        "Only full chips are allowed to cross-group such as [0,8),"
                        " other cross-group such as [3,6) are prohibited.")
    parser.add_argument("--visible_devices", type=str, default="0,1,2,3,4,5,6,7",
                        help="The visible devices according to the software system. "
                        "Usually used in the virtual system or docker container "
                        "that makes the device_id dismatch logic_id. "
                        "--device_num uses logic_id. "
                        "For example \"4,5,6,7\" means the system has 4 logic chips "
                        "which are actually the last 4 chips in hardware "
                        "while `--device_num` could only be set to \"[0, 4)\" "
                        "instead of \"[4, 8)\"")
    parser.add_argument("--server_ip", type=str, default="",
                        help="Set the server_ip manually, to avoid errors in auto detection.")
    args = parser.parse_args()
    return args


def get_host_ip():
    """
    get host ip
    """
    host_ip = None

    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
    except EOFError:
        pass

    return host_ip


def main():
    """main"""
    print("start", __file__)
    args = parse_args()

    # visible_devices
    visible_devices = args.visible_devices.split(',')
    print(f'visible_devices:{visible_devices}')

    # server_id
    host_ip = get_host_ip()
    if args.server_ip:
        server_id = args.server_ip
    elif host_ip:
        server_id = host_ip
    else:
        raise ValueError("please input server ip!")
    print(f'server_id:{server_id}')

    # device_num
    first_num = int(args.device_num[1])
    last_num = int(args.device_num[3])
    if first_num < 0 or last_num > 8:
        raise ValueError(f"device num {args.device_num} must be in range [0,8] !")
    if first_num > last_num:
        raise ValueError(f"First num {first_num} of device num \
                         {args.device_num} must less than last num {last_num} !")
    if first_num < 4 < last_num:
        if first_num == 0 and last_num == 8:
            pass
        else:
            raise ValueError(f"device num {args.device_num} must \
                             be in the same group of [0,4] or [4,8] !")

    device_num_list = list(range(first_num, last_num))
    print("device_num_list:", device_num_list)

    assert len(visible_devices) >= len(device_num_list)

    # construct hccn_table
    device_ips: Dict[Any, Any] = {}
    try:
        for device_id in device_num_list:
            ret = os.popen(f"hccn_tool -i {device_id} -ip -g").readlines()
            device_ips[str(device_id)] = ret[0].split(":")[1].replace('\n', '')
    except IndexError:
        print("Failed to call hccn_tool, try to read /etc/hccn.conf instead")
        try:
            with open('/etc/hccn.conf', 'r', encoding="utf-8") as fin:
                for hccn_item in fin.readlines():
                    if hccn_item.strip().startswith('address_'):
                        device_id, device_ip = hccn_item.split('=')
                        device_id = device_id.split('_')[1]
                        device_ips[device_id] = device_ip.strip()
        except OSError as exc:
            print("Failed to read /etc/hccn.conf")
            raise SystemError("Failed to find information for hccl") from exc

    hccn_table = {'version': '1.0',
                  'server_count': '1',
                  'server_list': []}
    device_list = []
    rank_id = 0
    for instance_id in device_num_list:
        device_id = visible_devices[instance_id]
        device_ip = device_ips[device_id]
        device = {'device_id': device_id,
                  'device_ip': device_ip,
                  'rank_id': str(rank_id)}
        print(f'rank_id:{rank_id}, device_id:{device_id}, device_ip:{device_ip}')
        rank_id += 1
        device_list.append(device)
    hccn_table['server_list'].append({
        'server_id': server_id,
        'device': device_list,
        'host_nic_ip': 'reserve'
    })
    hccn_table['status'] = 'completed'

    # save hccn_table to file
    file_path = os.path.abspath(__file__)
    table_path = os.path.dirname(file_path)
    table_fn = os.path.join(table_path,
                            f'rank_table_{len(device_num_list)}pcs.json')
    with open(table_fn, 'w', encoding="utf-8") as table_fp:
        json.dump(hccn_table, table_fp, indent=4)
    sys.stdout.flush()
    print("Completed: hccl file was save in :", table_fn)


if __name__ == "__main__":
    main()

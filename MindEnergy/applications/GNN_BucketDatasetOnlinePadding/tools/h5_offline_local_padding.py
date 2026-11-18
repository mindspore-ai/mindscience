#!/usr/bin/env python3

# Copyright 2026 Huawei Technologies Co., Ltd
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
"""offline global padding to local padding for h5 dataset"""



import h5py
import os
from pathlib import Path


# 数据形状：(50000, 460, 特征数)
h5_path_list =[
    "/home/work/user-job-dir/node/train_50_150.h5",
    "/home/work/user-job-dir/node/train_100_200.h5",
    "/home/work/user-job-dir/node/train_150_250.h5",
    "/home/work/user-job-dir/node/train_200_300.h5",
    "/home/work/user-job-dir/node/train_250_350.h5",
    "/home/work/user-job-dir/node/train_300_400.h5",
    "/home/work/user-job-dir/node/train_350_450.h5",
    "/home/work/user-job-dir/node/train_400_500.h5"
]


node_attrs = ['LP_shunt', 'bus_name', 'node_type', 'shunt_limit', 'theta', 'x0', 'x_vd']



# h5的数据结构为 f['key'][num_sample][num_node][num_feature]
# 若数据尚未全部读入内存，用 h5py 做懒加载切片，只读取需要的 150 维，内存占用瞬间降至原来的 150/460 ≈ 32.6%
def global_padding_to_local_padding(h5_path, max_padding):
    dst_h5_dir = os.path.dirname(h5_path)
    dst_h5_dir = os.path.join(dst_h5_dir, 'local_padding')
    if not os.path.exists(dst_h5_dir):
        os.makedirs(dst_h5_dir, exist_ok=True)

    p = Path(h5_path)
    new_name = f"{p.stem}_padding{max_padding}{p.suffix}"
    print(f"new_name={new_name}")
    dst_h5_path = os.path.join(dst_h5_dir, new_name)
    print(f"dst_h5_path={dst_h5_path}")

    global_padding_data = h5py.File(h5_path, 'r')
    print(f"global_padding_data['node_type'].shape={global_padding_data['node_type'].shape}")

    with h5py.File(dst_h5_path, 'w', libver='latest', rdcc_nslots=100000, rdcc_nbytes=1024 ** 3) as f:
        dt = h5py.special_dtype(vlen=float)

        sliced_data = {}
        for key in global_padding_data.keys():
            print(f"key={key}")
            if key in ['branch_data_dict', 'bus_data_dict', 'node_num_dict']:
                continue

            if key not in node_attrs:
                f.create_dataset(key, data=global_padding_data[key])
                print(f'h5 edge attr {key} copy successfully')
            else:
                # 只针对节点特征进行h5py切片，而对于边的特征直接拷贝
                # 这是个视图，没有拷贝数据
                sliced_data[key] = global_padding_data[key][:, :max_padding, :]
                print(sliced_data[key].shape)
                f.create_dataset(key, data=sliced_data[key])
                print(f'h5 node attr {key} convert successfully')

        print(list(sliced_data.keys()))
        print(len(sliced_data.keys()))

        f.close()
        print("Convert_h5_dataset successfully!")



def extract_number_from_path(path: str) -> int:
    """
    从文件路径中提取最后一个 '_' 与 '.h5' 之间的数字。

    参数:
        path: 文件路径字符串，例如 '/home/train_50_150.h5'

    返回:
        提取到的整数，若未找到或转换失败则返回 None
    
    测试示例：

    print(extract_number_from_path("/home/train_50_150.h5"))  # 输出: 150
    print(extract_number_from_path("data_42.h5"))             # 输出: 42
    print(extract_number_from_path("no_underscore.h5"))       # 输出: None (非数字)
    print(extract_number_from_path("test_abc.h5"))            # 输出: None (非数字)
    """
    last_underscore = path.rfind('_')
    dot_h5 = path.rfind('.h5')

    # 检查下划线和 .h5 是否存在且顺序正确
    if last_underscore == -1 or dot_h5 == -1 or last_underscore > dot_h5:
        return None

    number_str = path[last_underscore + 1:dot_h5]
    try:
        return int(number_str)
    except ValueError:
        return None




if __name__ == "__main__":

    for path in h5_path_list:
        current_max_padding = extract_number_from_path(path)
        print(f"current h5 path is {path}, max_padding={current_max_padding}")
        global_padding_to_local_padding(path, current_max_padding)



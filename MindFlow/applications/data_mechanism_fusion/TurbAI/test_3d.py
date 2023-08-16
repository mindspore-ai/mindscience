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
test_3d
"""
import argparse
import os

import numpy as np

from mindspore import Tensor, context, load_checkpoint
from mindspore.common import dtype as mstype
from mindflow.utils import load_yaml_config

from src import ResMLP
from src import plt_error_distribute
from src import PostProcess3DMinMax
from src import get_datalist_from_txt


def get_post_process_3d(reynolds, predict, data_path):
    """get_post_process_3d"""
    df_max = np.loadtxt(data_path + '/3d_max.dat')
    df_min = np.loadtxt(data_path + '/3d_min.dat')
    sca_max = Tensor(df_max, mstype.float32).reshape(1, -1)
    sca_min = Tensor(df_min, mstype.float32).reshape(1, -1)
    post_op = PostProcess3DMinMax(reynolds, sca_max[:, -1], sca_min[:, -1])
    pred_ori = post_op(predict).asnumpy().astype("float32")
    return pred_ori


def test(config):
    """test_3d"""
    test_path = config["data_path"] + '/test.txt'
    df_data = get_datalist_from_txt(test_path)
    test_re = Tensor(df_data['Re'].values, mstype.float32)
    test_label_ori = df_data["Mut"]
    test_dis = df_data["dis"]
    test_x = df_data["X"]
    test_set = np.load(config["data_path"] + '/test_data_3d.npy').astype(np.float32)
    test_data, test_label = test_set[:, 0:10], test_set[:, 10].reshape(-1, 1)
    print("load data test")
    # 读取模型
    # 实例化前向网络
    best_model = ResMLP(input_num=test_data.shape[1], width=64, depth=10, output_num=1)
    load_checkpoint(config["model_path"], net=best_model)

    # 测试集测试
    test_predict_list = best_model(Tensor(test_data))
    test_pred_ori = get_post_process_3d(test_re, test_predict_list, config["data_path"])
    test_predict_list = [data[0] for data in test_predict_list.asnumpy()]
    test_label = [data[0] for data in test_label]
    prefix = os.path.join(config["data_path"], "3d_network_example")
    plt_error_distribute(config["visualization"], test_x, test_pred_ori, test_label_ori,
                         test_dis, test_predict_list, test_label, prefix, "test")


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=7)
    parser.add_argument('--config_file_path', type=str, default="./configs/TurbAI_3D_ResMLP.yaml")
    input_args = parser.parse_args()
    return input_args


if __name__ == '__main__':
    # 参数设置
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, device_target="Ascend")
    test_config = load_yaml_config(args.config_file_path)
    test(test_config)

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
test_2d
"""
import argparse
import os

from mindspore import Tensor, load_checkpoint, context
from mindspore.common import dtype as mstype
from mindflow.utils import load_yaml_config

from src import MLP
from src import get_mean_std_data_from_txt, get_min_max_data_from_txt
from src import plt_error_distribute
from src import PostProcess2DMinMax, PostProcess2DStd
from src import get_tensor_data, get_datalist_from_txt


def get_post_process(reynolds, dis, predict, label_norm, file_path):
    """get_post_process"""
    if label_norm == "MinMax":
        df_max, df_min = get_min_max_data_from_txt(file_path)
        post_op = PostProcess2DMinMax(reynolds, dis, df_max[:, -1], df_min[:, -1])
        pred_ori = post_op(predict).asnumpy().astype("float32")
        return pred_ori
    if label_norm == "Std":
        df_mean, df_std = get_mean_std_data_from_txt(file_path)
        post_op = PostProcess2DStd(reynolds, dis, df_mean[:, -1], df_std[:, -1])
        pred_ori = post_op(predict).asnumpy().astype("float32")
        return pred_ori
    return []


def test(config):
    """test_2d"""
    # 测试集生成
    test_path = config["data_path"] + '/test.txt'
    df_data = get_datalist_from_txt(test_path)
    test_dis = df_data["dis"]
    test_dis_tensor = Tensor(test_dis.values, mstype.float32)
    test_re = Tensor(df_data['Re'].values, mstype.float32)
    test_label_ori = df_data["Mut"]
    test_x = df_data["X"]
    test_data, test_label, _, _ = get_tensor_data(df_data, config["feature_norm"],
                                                  config["label_norm"], config["data_path"])
    test_data = test_data.asnumpy().astype("float32")
    test_label = test_label.asnumpy().astype("float32").reshape(-1, 1)

    # 读取模型
    best_model = MLP(config["MLP"])
    load_checkpoint(config["model_path"], net=best_model)

    # 测试集测试
    test_predict_list = best_model(Tensor(test_data))
    test_pred_ori = get_post_process(test_re, test_dis_tensor, test_predict_list,
                                     config["label_norm"], config["data_path"])
    test_predict_list = [data[0] for data in test_predict_list.asnumpy()]
    test_label = [data[0] for data in test_label]
    prefix = os.path.join(config["data_path"], "2d_network_example")
    plt_error_distribute(config["visualization"], test_x, test_pred_ori, test_label_ori,
                         test_dis, test_predict_list, test_label, prefix, "test")


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=7)
    parser.add_argument('--config_file_path', type=str, default="./configs/TurbAI_2D_MLP.yaml")
    input_args = parser.parse_args()
    return input_args


if __name__ == '__main__':
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, device_target="Ascend")
    test_config = load_yaml_config(args.config_file_path)
    test(test_config)

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
"""CAE and CNN train"""

# Python标准库导入

# 第三方库导入
from sciai.context import init_project

# 本地项目模块导入
import src.process as process
from src.cae_3d_ms import get_cae_data, get_cae_model, train_cae_model
from src.cnn_3d_ms import get_cnn_data, prepare, train_cnn_model2, train_cnn_model1

def main(args):
    x_train, x_test = get_cae_data(args)
    model, train_data, es, mc, re = \
        get_cae_model(x_train, x_test, args)
    train_cae_model(model, train_data, es, mc, re, args)


    train_dataset, test_dataset, matrix = get_cnn_data(args)
    loss_fn, total_epochs, config_ck, ckpoint_cb, es, ls = prepare(args)
    train_cnn_model1(train_dataset, test_dataset, matrix, loss_fn, total_epochs, config_ck, ckpoint_cb, es, ls)
    train_cnn_model2(train_dataset, test_dataset, matrix, loss_fn, total_epochs, config_ck, ckpoint_cb, es, ls)


if __name__ == "__main__":
    args_ = process.prepare()
    init_project(args=args_[0])
    main(*args_)

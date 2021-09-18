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
# ==============================================================================
"""
eval
"""
import os
import argparse
import numpy as np

from mindspore.common import set_seed
from mindspore import context
from mindspore import load_checkpoint
import mindspore.dataset as ds

from src.model import S11Predictor
from src.metric import l2_error_np
from src.config import config

set_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--label_path', type=str)
parser.add_argument('--data_config_path', default='./src/data_config.npz', type=str)
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--model_path', help='checkpoint directory')
parser.add_argument('--output_path', default="./")

opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=opt.device_num)

def evaluation():
    """evaluation"""

    model_net = S11Predictor(input_dim=config["input_channels"])
    load_checkpoint(opt.model_path, model_net)
    print("model loaded")

    data_config = np.load(opt.data_config_path)
    scale_s11 = data_config["scale_s11"]
    input_data = np.load(opt.input_path)
    label_data = np.load(opt.label_path)
    label_predict = np.zeros(label_data.shape)

    def create_dataset(net_input, net_label, batch_size=16, num_workers=4, shuffle=False):
        data = (net_input, net_label)
        dataset = ds.NumpySlicesDataset(data, column_names=["data", "label"], shuffle=shuffle)
        dataset = dataset.batch(batch_size=batch_size, num_parallel_workers=num_workers)
        return dataset

    data_loader = create_dataset(input_data, label_data, batch_size=config['batch_size'])

    model_net.set_train(False)

    label_predict = np.zeros(label_data.shape)
    i = 0
    for iter_data in data_loader.create_dict_iterator():
        test_input_space, test_label = iter_data["data"], iter_data["label"]
        test_predict = model_net(test_input_space).asnumpy()
        test_label = test_label.asnumpy()
        for i in range(test_label.shape[0]):
            predict_tmp = test_predict[i, :]
            label_tmp = test_label[i, :]
            label_real_tmp = 1.0 - np.power(10, label_tmp * scale_s11)
            predict_real_tmp = 1.0 - np.power(10, predict_tmp * scale_s11)

            print("error for %i input: %s" %(i, l2_error_np(label_real_tmp, predict_real_tmp)))
            label_predict[i] = label_real_tmp
            i += 1

    np.save(os.path.join(opt.output_path, "predicted_label.npy"), label_predict)
    print("predicted S parameter saved in %s" %opt.output_path)

if __name__ == '__main__':
    evaluation()

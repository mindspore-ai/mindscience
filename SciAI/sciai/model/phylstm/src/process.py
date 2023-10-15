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

"""process"""
import os
import math
import yaml

import numpy as np
from mindspore.dataset import GeneratorDataset
from scipy import io
from plot import plot_your_figure

from sciai.utils import parse_arg, print_log


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def get_data(args, mode):
    """get data from bouc-wen"""
    # load original data
    mat = io.loadmat(args.load_data_path)
    ag_data = np.expand_dims(np.concatenate([mat['input_tf'], mat['input_pred_tf']]), -1).astype(np.float32)
    u_data = np.expand_dims(np.concatenate([mat['target_X_tf'], mat['target_pred_X_tf']]), -1).astype(np.float32)
    ut_data = np.expand_dims(np.concatenate([mat['target_Xd_tf'], mat['target_pred_Xd_tf']]), -1).astype(np.float32)
    utt_data = np.expand_dims(np.concatenate([mat['target_Xdd_tf'], mat['target_pred_Xdd_tf']]),
                              1).astype(np.float32)
    # calculate phi
    t = mat['time']
    dt = t[0, 1] - t[0, 0]
    n = u_data.shape[1]
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3])])
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([n - 3]), np.array([1 / 2, -2, 3 / 2])])
    phi_t0 = 1 / dt * np.concatenate(
        [np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])])
    phi_t0 = np.reshape(phi_t0, [1, n, n]).astype(np.float32)

    if mode == "train":
        ag_train = ag_data[0:10]
        u_train = u_data[0:10]
        ut_train = ut_data[0:10]
        utt_train = utt_data[0:10]
        g_train = -utt_train - ag_train
        phi = np.repeat(phi_t0, ag_train.shape[0], axis=0)
        return ag_train, u_train, ut_train, utt_train, g_train, phi
    if mode == "addition":
        ag_c_train = ag_data[0:50]
        lift_train = -ag_c_train
        phi = np.repeat(phi_t0, ag_c_train.shape[0], axis=0)
        return ag_c_train, lift_train, phi
    if mode == "test":
        ag_test = ag_data[10:]
        u_test = u_data[10:]
        ut_test = ut_data[10:]
        utt_test = utt_data[10:]
        g_test = -utt_test - ag_test
        phi = np.repeat(phi_t0, ag_test.shape[0], axis=0)
        return ag_test, u_test, ut_test, utt_test, g_test, phi
    return None


class BoucWenTrain:
    """train dataset of bouc-wen"""
    def __init__(self, args):
        self.ag_train, self.u_train, self.ut_train, self.utt_train, self.g_train, self.phi = get_data(args, "train")

    def __getitem__(self, index):
        return self.ag_train[index], self.u_train[index], self.ut_train[index], self.utt_train[index], \
               self.g_train[index], self.phi[index]

    def __len__(self):
        # return self.ag_train.shape[0]
        return len(self.ag_train)


class BoucWenAddition:
    """addition dataset of bouc-wen"""
    def __init__(self, args):
        self.ag_c_train, self.lift_train, self.phi = get_data(args.load_data_path, "addition")

    def __getitem__(self, index):
        return self.ag_c_train[index], self.lift_train[index], self.phi[index]

    def __len__(self):
        return len(self.ag_c_train)


class BoucWenTest:
    """test dataset of bouc-wen"""
    def __init__(self, args):
        self.ag_test, self.u_test, self.ut_test, self.utt_test, self.g_test, self.phi = get_data(args.load_data_path,
                                                                                                 "test")

    def __getitem__(self, index):
        return self.ag_test[index], self.u_test[index], self.ut_test[index], self.utt_test[index], self.g_test[index], \
               self.phi[index]

    def __len__(self):
        return len(self.ag_test)


def prepare_dataset(args):
    """prepare dateset"""
    if args.mode == 'train':
        train_dataset = GeneratorDataset(source=BoucWenTrain(args),
                                         column_names=['ag_train', 'u_train', 'ut_train', 'utt_train', 'g_train',
                                                       'phi']).batch(batch_size=10)
        print_log('Successfully loaded train_dataset!')
        addition_dataset = GeneratorDataset(source=BoucWenAddition(args),
                                            column_names=['ag_c_train', 'lift_train', 'phi']).batch(batch_size=50)
        print_log('Successfully loaded addition_dataset!')
        return train_dataset, addition_dataset
    if args.mode == 'val':
        test_dataset = GeneratorDataset(source=BoucWenTest(args),
                                        column_names=['ag_test', 'u_test', 'ut_test', 'utt_test', 'g_test',
                                                      'phi']).batch(batch_size=90)
        print_log('Successfully loaded test_dataset!')
        return test_dataset
    return None


def post_process(args, network, dataset):
    """post process"""
    for _, (ag_test, u_test, ut_test, utt_test, g_test, phi) in enumerate(dataset.create_tuple_iterator()):
        z_1, z_1_dot, _, z_2_dot, _, lift = network.predict(ag_test, phi)
        g = -z_2_dot + lift

        # Using correlation coefficients to measure the performance of output <z_1>
        output_1 = 'z_1'
        gamma_list = []
        for i in range(len(u_test)):
            # Calculate Pearson correlation coefficient
            gamma = np.corrcoef(u_test[i, :, 0].asnumpy(), z_1[i, :, 0].asnumpy())[0, 1]
            # Truncate to two decimal places
            gamma = math.floor(gamma * 100) / 100
            gamma_list.append(gamma)
        mean = np.mean(gamma_list)
        print_log("Mean correlation coefficient of z_1:{}".format(mean))
        plot_your_figure(args, output_1, gamma_list)

        # Using correlation coefficients to measure the performance of output <z_1_dot>
        output_2 = 'z_1_dot'
        gamma_list = []
        for i in range(len(ut_test)):
            # Calculate Pearson correlation coefficient
            gamma = np.corrcoef(ut_test[i, :, 0].asnumpy(), z_1_dot[i, :, 0].asnumpy())[0, 1]
            # Truncate to two decimal places
            gamma = math.floor(gamma * 100) / 100
            gamma_list.append(gamma)
        mean = np.mean(gamma_list)
        print_log("Mean correlation coefficient of z_1_dot:{}".format(mean))
        plot_your_figure(args, output_2, gamma_list)

        # Using correlation coefficients to measure the performance of output <z_2_dot>
        output_3 = 'z_2_dot'
        gamma_list = []
        for i in range(len(utt_test)):
            # Calculate Pearson correlation coefficient
            gamma = np.corrcoef(utt_test[i, :, 0].asnumpy(), z_2_dot[i, :, 0].asnumpy())[0, 1]
            # Truncate to two decimal places
            gamma = math.floor(gamma * 100) / 100
            gamma_list.append(gamma)
        mean = np.mean(gamma_list)
        print_log("Mean correlation coefficient of z_2_dot:{}".format(mean))
        plot_your_figure(args, output_3, gamma_list)

        # Using correlation coefficients to measure the performance of output <g>
        output_4 = 'g'
        gamma_list = []
        for i in range(len(g_test)):
            # Calculate Pearson correlation coefficient
            gamma = np.corrcoef(g_test[i, :, 0].asnumpy(), g[i, :, 0].asnumpy())[0, 1]
            # Truncate to two decimal places
            gamma = math.floor(gamma * 100) / 100
            gamma_list.append(gamma)
        mean = np.mean(gamma_list)
        print_log("Mean correlation coefficient of z_1:{}".format(mean))
        plot_your_figure(args, output_4, gamma_list)

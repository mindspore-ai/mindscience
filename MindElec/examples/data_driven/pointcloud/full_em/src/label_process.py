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
# ============================================================================
"""label process"""
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Electromagnetic Simulation')
parser.add_argument('--label_path', default='/input/path/')
parser.add_argument('--processed_label_path', default='./processed_label.npy')
parser.add_argument('--label_scale_path', default='./label_scale.npy')
parser.add_argument('--x_solution', type=int, default=50)
parser.add_argument('--y_solution', type=int, default=50)
parser.add_argument('--z_solution', type=int, default=8)
parser.add_argument('--t_solution', type=int, default=100)
parser.add_argument('--is_train', type=bool, default=True, help='train label or eval label')
opt = parser.parse_args()

labels = np.load(opt.label_path)
X, Y, Z, T = opt.x_solution, opt.y_solution, opt.z_solution, opt.t_solution
inputs_label = np.ones((labels.shape[0], T, X, Y, Z, 6), dtype=np.float32)

if opt.is_train:
    scale_eh = np.ones((6, T)).astype(np.float32)
    for i in range(T):
        scale_eh[0][i] = np.max([np.abs(np.max(labels[:, i, :, :, :, 0])), np.abs(np.min(labels[:, i, :, :, :, 0]))])
        scale_eh[1][i] = np.max([np.abs(np.max(labels[:, i, :, :, :, 1])), np.abs(np.min(labels[:, i, :, :, :, 1]))])
        scale_eh[2][i] = np.max([np.abs(np.max(labels[:, i, :, :, :, 2])), np.abs(np.min(labels[:, i, :, :, :, 2]))])
        scale_eh[3][i] = np.max([np.abs(np.max(labels[:, i, :, :, :, 3])), np.abs(np.min(labels[:, i, :, :, :, 3]))])
        scale_eh[4][i] = np.max([np.abs(np.max(labels[:, i, :, :, :, 4])), np.abs(np.min(labels[:, i, :, :, :, 4]))])
        scale_eh[5][i] = np.max([np.abs(np.max(labels[:, i, :, :, :, 5])), np.abs(np.min(labels[:, i, :, :, :, 5]))])

        scale_eh[0][i] = 1.0 if scale_eh[0][i] == 0.0 else scale_eh[0][i]
        scale_eh[1][i] = 1.0 if scale_eh[1][i] == 0.0 else scale_eh[1][i]
        scale_eh[2][i] = 1.0 if scale_eh[2][i] == 0.0 else scale_eh[2][i]
        scale_eh[3][i] = 1.0 if scale_eh[3][i] == 0.0 else scale_eh[3][i]
        scale_eh[4][i] = 1.0 if scale_eh[4][i] == 0.0 else scale_eh[4][i]
        scale_eh[5][i] = 1.0 if scale_eh[5][i] == 0.0 else scale_eh[5][i]

        labels[:, i, :, :, :, 0] = labels[:, i, :, :, :, 0] / scale_eh[0][i]
        labels[:, i, :, :, :, 1] = labels[:, i, :, :, :, 1] / scale_eh[1][i]
        labels[:, i, :, :, :, 2] = labels[:, i, :, :, :, 2] / scale_eh[2][i]
        labels[:, i, :, :, :, 3] = labels[:, i, :, :, :, 3] / scale_eh[3][i]
        labels[:, i, :, :, :, 4] = labels[:, i, :, :, :, 4] / scale_eh[4][i]
        labels[:, i, :, :, :, 5] = labels[:, i, :, :, :, 5] / scale_eh[5][i]
    path = opt.label_scale_path
    np.save(path, scale_eh)

# fusion time axis
processed_label = np.reshape(labels, (labels.shape[0] * T, X, Y, Z, labels.shape[-1]))

# save label
path = opt.processed_label_path
np.save(path, processed_label)
print("save finished!")

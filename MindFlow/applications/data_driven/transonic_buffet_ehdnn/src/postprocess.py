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
"""Post-processing """
import os
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.io import loadmat


def plot_train_loss(train_loss, plot_dir, epochs):
    """Plot change of loss during training"""
    t_loss = plt.scatter(list(range(epochs)), train_loss, s=0.2)
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.legend([t_loss], ['train'], loc='upper right')
    plt.savefig(f'{plot_dir}/train_loss.png')


class PostProcess:
    r"""
    Tools of PostProcess of prediction (convert Tensor data into flow field contours)

    Args:
        foil_file (str): The file path of the oat15a airfoil geometry file.
        size_field (int): The size of flow field contour snapshot.
        x_range (list): The range of x-coordinate of flow field contour.[start_x,end_x]
        y_range (list): The range of y-coordinate of flow field contour.[start_y,end_y]

    Inputs:
        prediction_dir(str): The directory where the flow field contour snapshot is saved.
        flow_type(int): The type of flow field contour snapshot.(0 or 1 or 2)

    outputs:
        storage the flow field contours
        (The data format of flow field contours is Tecplot Data,which can be opened with Tecplot).
    """

    def __init__(self,
                 foil_path,
                 size_field,
                 x_range,
                 y_range):
        self.foil_path = foil_path
        self.size_field = size_field
        self.x_range = x_range
        self.y_range = y_range
        self.interval_xy = round(((self.x_range[1] - self.x_range[0]) / (self.size_field - 1)), 4)

    def foil_cut(self):
        """The foil wall processing"""
        foil_xy = np.load(self.foil_path)
        x = foil_xy[:, 0]
        y = foil_xy[:, 1]
        m = np.ones((len(foil_xy)))

        ti = np.arange(self.x_range[0], self.x_range[1], self.interval_xy)
        tj = np.arange(self.y_range[0], self.y_range[1], self.interval_xy)
        xx, yy = np.meshgrid(ti, tj)
        mm = griddata((x, y), m, (xx, yy), method='cubic')

        for i in range(self.size_field):
            for j in range(self.size_field):
                if np.isnan(mm[i, j]):
                    mm[i, j] = 0
        mm = mm * 100
        return mm.T

    def plot_flow_field(self, prediction_dir, flow_type):
        """
        Convert Tensor data into flow field contours

        "flow_field_class" : ['real','prediction','abs_error']
        "flow_field_type" : ['P',    'U',  'V']
        "flow_class : the value is 0 or 1 or 2 , means flow field_class[flow_class]
                  flow_field_class[0] ('real')
                  flow_field_class[1] ('prediction')
                  flow_field_class[2] ('abs_error')
        "flow_type" : the value is 0 or 1 or 2 , means flow field_type[flow_type]
                  flow_field_type[0] ('P')
                  flow_field_type[1] ('U')
                  flow_field_type[2] ('V')
        """
        flow_field_class = ['real', 'prediction', 'abs_error']
        flow_field_type = ['P', 'U', 'V']
        flow_field_data = loadmat(f'{prediction_dir}/prediction_data.mat')

        flow_dir = os.path.join(prediction_dir, flow_field_type[flow_type])
        if not os.path.exists(flow_dir):
            os.mkdir(flow_dir)

        foil = self.foil_cut()
        for c in flow_field_class:
            flow_field = flow_field_data[f"{c}"]
            for n in range(len(flow_field)):
                file = os.open(f"{flow_dir}/{c}_{n + 1}.dat", "w+")
                file.write(f"VARIABLES = X, Y,{flow_field_type[flow_type]}_{c}\n")
                file.write(f"zone i={self.size_field} j={self.size_field}\n")
                single_flow = np.squeeze(flow_field[n, flow_type, :, :])
                single_flow = single_flow + foil
                for i in range(self.size_field):
                    x = self.x_range[0] + i * self.interval_xy
                    for j in range(self.size_field):
                        y = self.y_range[0] + j * self.interval_xy
                        file.write(f"{x} {y} {single_flow[i, j]}\n")
                file.close()

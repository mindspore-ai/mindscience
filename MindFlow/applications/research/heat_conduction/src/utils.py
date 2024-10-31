# Copyright 2024 Huawei Technologies Co., Ltd
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
utils
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from mindspore import ops, jit_class, Tensor
from mindspore import dtype as mstype
from mindflow.utils import load_yaml_config
from mindflow.cell import UNet2D


def init_model(backbone, data_params, model_params, compute_dtype=mstype.float32):
    """initial_data_and_model"""
    _ = data_params
    _ = compute_dtype
    if backbone == "unet2d":
        model = UNet2D(in_channels=model_params["in_channels"],
                       out_channels=model_params["out_channels"],
                       base_channels=model_params["unet2d"]["base_channels"],
                       kernel_size=model_params["unet2d"]["kernel_size"],
                       stride=model_params["unet2d"]["stride"])
    return model


def check_file_path(file_path):
    """check_file_path"""
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def count_params(params):
    """count_params"""
    count = 0
    for p in params:
        t = 1
        for i in range(len(p.shape)):
            t = t * p.shape[i]
        count += t
    return count


def plot_image(tensor, sample_no):
    image = tensor.asnumpy()[sample_no, :, :, 0]
    plt.imshow(image, cmap='rainbow')
    plt.axis('off')
    plt.show()


def plot_image_first(tensor, sample_no):
    image = tensor.asnumpy()[sample_no, :, :, 1]
    plt.imshow(image, cmap='rainbow')
    plt.axis('off')
    plt.show()


@jit_class
class Trainer():
    r"""
    Trainer

    Args:
        model (Model): The Unet2D/FNO2D model.
        data_params (dict): The data parameters loaded from yaml file.
        loss_fn (Tensor): The loss function.
        means (list): The mean value of every input channel.
        stds (list): The standard deviation value of every input channel.

    Inputs:
            - inputs (Tensor) - Tensor of shape :math:`(batch\_size*T_in, resolution, resolution, channels)`.
            - labels (Tensor) - Tensor of shape :math:`(batch\size, resolution, resolution, channels)`.

    Outputs:
            - loss (float) - The average loss calculated by average of test step losses.
            - loss_full (float) - The average loss directly calculated for the current batch.
            - pred (Tensor) - Tensor of shape :math:`(batch\size, resolution, resolution, channels)`.
            - step_losses (list) - The list of step losses with length of T_out
    """

    def __init__(self, model, data_params, loss_fn, means, stds, config_path="./configs/combined_methods.yaml"):
        self.config = load_yaml_config(config_path)
        self.variation_params = self.config["variation"]
        self.model = model
        self.test_steps = data_params["T_out"]
        self.loss_fn = loss_fn
        self.mean = Tensor(means, dtype=mstype.float32)
        self.std = Tensor(stds, dtype=mstype.float32)
        self.hatch_extent = self.variation_params["hatch_extent"]
        self.loss1_list = []
        self.loss2_list = []
        self.loss1_scale = self.variation_params["loss1_scale"]
        self.max_scale = self.variation_params["max_scale"]

    def _build_features(self, inputs):
        return inputs.astype(mstype.float32)

    def get_loss(self, inputs, labels):
        """get loss"""

        embeds = self._build_features(inputs)
        y = labels[:, :, :, :]

        batch_no = embeds.shape[0]
        loss = 0
        pred = 0
        step_losses = []

        embeds = ops.cast(embeds, mstype.float32)
        bcs = embeds[:, :, :, 0:1]
        mask_no_bcout = embeds[:, 1:127, 1:127, 1:2]

        mask_no_bcout_transposed = ops.transpose(mask_no_bcout, (0, 3, 1, 2))
        mask_with_bcout_transposed = ops.pad(mask_no_bcout_transposed, [1, 1, 1, 1], mode='constant', value=1.0)
        mask_with_bcout = ops.transpose(mask_with_bcout_transposed, (0, 2, 3, 1))

        im_org = self.model(embeds)
        im_org = ops.cast(im_org, mstype.float32)
        pred = im_org * (1 - mask_with_bcout) + bcs

        l_data = self.loss_fn(pred, y)

        pred_tensor = ops.transpose(pred, (0, 3, 1, 2))

        desired_weight_x = np.array([[[[0, 1.0, 0], [0, -2.0, 0], [0, 1.0, 0]]]])
        weight_x = Tensor(desired_weight_x, dtype=mstype.float32)
        desired_weight_y = np.array([[[[0, 0, 0], [1.0, -2.0, 1.0], [0, 0, 0]]]])
        weight_y = Tensor(desired_weight_y, dtype=mstype.float32)
        conv2d = ops.Conv2D(out_channel=1, kernel_size=3)
        dtd2 = conv2d(pred_tensor, weight_x) + conv2d(pred_tensor, weight_y)

        dtd2 = ops.transpose(dtd2, (0, 2, 3, 1))
        dtd2 = dtd2 * (1 - mask_no_bcout)

        np_array = np.zeros((batch_no, 126, 126, 1))
        tensor_zeros = Tensor(np_array, dtype=mstype.float32)
        l_phy = self.loss_fn(dtd2, tensor_zeros)

        l_total = self.loss1_scale * l_data + l_phy

        step_losses.append(l_total)
        loss += l_total

        return loss, l_data, l_phy, inputs, pred, labels, step_losses

    def renew_loss_lists(self, loss1, loss2):
        self.loss1_list.append(loss1.asnumpy())
        self.loss2_list.append(loss2.asnumpy())
        if len(self.loss1_list) > self.hatch_extent:
            self.loss1_list.pop(0)
            self.loss2_list.pop(0)

    def adjust_hatchs(self):
        """adjust_hatchs"""
        if len(self.loss1_list) >= self.hatch_extent:
            sliding_avg_loss1 = np.mean(self.loss1_list[-self.hatch_extent:])
            sliding_avg_loss2 = np.mean(self.loss2_list[-self.hatch_extent:])
        else:
            sliding_avg_loss1 = np.mean(self.loss1_list)
            sliding_avg_loss2 = np.mean(self.loss2_list)

        if sliding_avg_loss2 > 0:
            present_ratio = sliding_avg_loss2 / (sliding_avg_loss1 * self.loss1_scale)
            self.loss1_scale *= present_ratio
            self.loss1_scale = min(self.loss1_scale, self.max_scale)

    def _denormalize(self, x):
        return x * self.std + self.mean

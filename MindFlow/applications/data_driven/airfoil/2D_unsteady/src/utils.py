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
utils
"""
import os

from mindspore import ops, jit_class, Tensor
from mindspore import dtype as mstype

from .unet import Unet2D
from .fno2d import FNO2D


def init_model(backbone, data_params, model_params, compute_dtype=mstype.float32):
    """initial_data_and_model"""
    if backbone == "fno2d":
        model = FNO2D(in_channels=model_params["in_channels"] * data_params['T_in'],
                      out_channels=model_params["out_channels"],
                      resolution=model_params["resolution"],
                      modes=model_params["fno2d"]["modes"],
                      channels=model_params["fno2d"]["channels"],
                      depths=model_params["fno2d"]["depths"],
                      mlp_ratio=model_params["fno2d"]["mlp_ratio"],
                      compute_dtype=compute_dtype)
    else:
        model = Unet2D(in_channels=model_params["in_channels"] * data_params['T_in'],
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

    def __init__(self, model, data_params, loss_fn, means, stds):
        self.model = model
        self.test_steps = data_params["T_out"]
        self.loss_fn = loss_fn
        self.mean = Tensor(means)
        self.std = Tensor(stds)

    def _build_features(self, inputs):
        inputs = ops.transpose(inputs, (0, 2, 3, 1, 4))
        batch_size, height, width, _, _ = inputs.shape
        inputs = ops.reshape(inputs, (batch_size, height, width, -1))
        return inputs

    def get_loss(self, inputs, labels):
        """get loss"""
        input_dim_per_step = 3
        embeds = self._build_features(inputs)
        batch_size = embeds.shape[0]

        loss = 0
        pred = 0
        step_losses = []
        for t in range(self.test_steps):
            y = labels[:, t: t + 1, :, :, :]
            embeds = ops.cast(embeds, mstype.float32)
            im_org = self.model(embeds)
            im = ops.expand_dims(im_org, 1)
            im_inverse = self._denormalize(im)
            l = self.loss_fn(ops.reshape(im_inverse, (batch_size, -1)),
                             ops.reshape(y, (batch_size, -1)))
            step_losses.append(l)
            loss += l
            pred = im if t == 0 else ops.concat((pred, im), 1)
            im = ops.cast(ops.squeeze(im, 1), mstype.float32)
            embeds = ops.concat((embeds[..., input_dim_per_step:], im), -1)
        pred = self._denormalize(pred)
        return loss, pred, labels, step_losses

    def _denormalize(self, x):
        return x * self.std + self.mean

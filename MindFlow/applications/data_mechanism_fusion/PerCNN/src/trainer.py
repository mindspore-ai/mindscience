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
"""trainer"""
import numpy as np
import scipy.io as sio

from mindspore import nn, ops, Tensor, float32, jit_class

from .constant import dx_2d_op, dy_2d_op, lap_2d_op


@jit_class
class Trainer:
    """Trainer"""

    def __init__(self, model, time_steps=400, dx=1.0/100, dt=0.00025, nu=1/200, compute_dtype=float32):
        self.model = model
        self.loss = nn.MSELoss()
        self.dx = dx
        self.dt = dt
        self.nu = nu
        self.compute_dtype = compute_dtype
        self.pec = 0.05

        mat = sio.loadmat('./data/Burgers_2001x2x100x100_[dt=00025].mat')['uv']
        self.truth_clean = mat[100:1901]
        uv = self.add_noise(self.truth_clean)

        self.truth = Tensor(uv[:time_steps+1], dtype=self.compute_dtype)
        self.init_state_low = Tensor(
            uv[0:1, :, ::2, ::2], dtype=self.compute_dtype)

        self.dx_kernel = Tensor(np.array(dx_2d_op) /
                                self.dx, self.compute_dtype)
        self.dy_kernel = Tensor(np.array(dy_2d_op) /
                                self.dx, self.compute_dtype)
        self.lap_kernel = Tensor(
            np.array(lap_2d_op) / self.dx**2, self.compute_dtype)

    def add_noise(self, truth):
        noise = np.random.normal(size=(truth.shape))
        std = np.std(truth)
        return truth + self.pec*std*noise

    def get_ic_loss(self):
        init_state_bicubic = ops.interpolate(
            self.init_state_low, size=(100, 100), mode='bicubic')
        ini_state_pred = self.model.upconv_block(self.init_state_low)
        return self.loss(ini_state_pred, init_state_bicubic)

    def get_phy_loss(self, output):
        """calculate the phy loss"""
        output = ops.concat(
            (output[:, :, :, -2:], output, output[:, :, :, 0:3]), axis=3)
        output = ops.concat(
            (output[:, :, -2:, :], output, output[:, :, 0:3, :]), axis=2)

        laplace_u = ops.conv2d(output[0:-2, 0:1, :, :], self.lap_kernel)
        laplace_v = ops.conv2d(output[0:-2, 1:2, :, :], self.lap_kernel)

        u_x = ops.conv2d(output[0:-2, 0:1, :, :], self.dx_kernel)
        u_y = ops.conv2d(output[0:-2, 0:1, :, :], self.dy_kernel)
        v_x = ops.conv2d(output[0:-2, 1:2, :, :], self.dx_kernel)
        v_y = ops.conv2d(output[0:-2, 1:2, :, :], self.dy_kernel)

        u_t = (output[1:-1, 0:1, 2:-2, 2:-2] -
               output[0:-2, 0:1, 2:-2, 2:-2]) / self.dt
        v_t = (output[1:-1, 1:2, 2:-2, 2:-2] -
               output[0:-2, 1:2, 2:-2, 2:-2]) / self.dt

        u = output[0:-2, 0:1, 2:-2, 2:-2]
        v = output[0:-2, 1:2, 2:-2, 2:-2]

        f_u = u_t - self.nu*laplace_u + u*u_x + v*u_y
        f_v = v_t - self.nu*laplace_v + u*v_x + v*v_y

        return self.loss(f_u, ops.zeros_like(f_u)) + self.loss(f_v, ops.zeros_like(f_v))

    def get_loss(self):
        """get loss"""
        output, _ = self.model(self.init_state_low)
        output = ops.concat(tuple(output), axis=0)

        pred = output[0:-1:40, :, ::2, ::2]
        gt = self.truth[::40, :, ::2, ::2]
        idx = int(pred.shape[0] * 0.9)

        pred_tra, pred_val = pred[:idx], pred[idx:]  # prediction
        gt_tra, gt_val = gt[:idx], gt[idx:]  # ground truth

        loss_data = self.loss(pred_tra, gt_tra)
        loss_valid = self.loss(pred_val, gt_val)
        loss_ic = self.get_ic_loss()

        loss_phy = self.get_phy_loss(output)

        return 1.0*loss_data + 2.0*loss_ic, loss_data, loss_ic, loss_phy, loss_valid
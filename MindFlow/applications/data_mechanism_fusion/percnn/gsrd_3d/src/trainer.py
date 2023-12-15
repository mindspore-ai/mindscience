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

from .constant import lap_3d_op


@jit_class
class Trainer:
    """Trainer"""

    def __init__(self, upconv, recurrent_cnn, timesteps_for_train, dx, grid_size,
                 dt, mu, data_path, compute_dtype=float32):
        self.upconv = upconv
        self.recurrent_cnn = recurrent_cnn
        self.loss = nn.MSELoss()
        self.dx = dx / grid_size
        self.dt = dt
        self.mu = mu
        self.compute_dtype = compute_dtype
        self.pec = 0.1
        self.timesteps_for_train = timesteps_for_train

        mat = sio.loadmat(data_path)['uv']
        uv = np.transpose(mat, (1, 0, 2, 3, 4)).astype(np.float32)
        print("shape of uv is ", uv.shape)
        self.truth_clean = Tensor(uv)
        uv = self.add_noise(uv)
        ic = uv[1000:1001, :, :, :, :]
        print("shape of ic is ", ic.shape)

        self.truth = Tensor(
            uv[1000:self.timesteps_for_train + 1001], dtype=self.compute_dtype)
        self.init_state_low = Tensor(
            ic[0:1, :, ::2, ::2, ::2], dtype=self.compute_dtype)

        print("shape of init_state_low is ", self.init_state_low.shape)

        self.dt_kernel = Tensor(np.array([[[-1, 1, 0]]]) /
                                self.dt, self.compute_dtype)

        self.lap_kernel = Tensor(
            np.array(lap_3d_op) / self.dx**2, self.compute_dtype)

    def add_noise(self, truth):
        res = []
        for i in range(truth.shape[1]):
            u = truth[:, i: i + 1, ...]
            noise = np.random.normal(size=(u.shape))
            std = np.std(u)
            res.append(u + self.pec*std*noise/np.std(noise))
        return np.concatenate(res, axis=1)

    def get_output(self, infer_step):
        x = self.upconv(self.init_state_low)
        res = [x]
        for _ in range(infer_step):
            x = self.recurrent_cnn(x)
            res.append(x)
        return ops.cat(res, axis=0)

    def get_ic_loss(self):
        init_state_bicubic = ops.interpolate(
            self.init_state_low, size=(48, 48, 48), mode='trilinear')
        ini_state_pred = self.upconv(self.init_state_low)
        return self.loss(ini_state_pred, init_state_bicubic)

    def get_phy_loss(self, output):
        """calculate the phy loss"""
        output = ops.concat(
            (output[:, :, :, :, -2:], output, output[:, :, :, :, 0:3]), axis=4)
        output = ops.concat(
            (output[:, :, :, -2:, :], output, output[:, :, :, 0:3, :]), axis=3)
        output = ops.concat(
            (output[:, :, -2:, :, :], output, output[:, :, 0:3, :, :]), axis=2)

        laplace_u = ops.conv3d(output[0:-2, 0:1, :, :, :], self.lap_kernel)
        laplace_v = ops.conv3d(output[0:-2, 1:2, :, :, :], self.lap_kernel)

        u = output[:, 0:1, 2:-2, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        lenz = u.shape[4]
        # [height(Y), width(X), depth, c, step]
        u_conv1d = u.transpose(2, 3, 4, 1, 0)
        u_conv1d = u_conv1d.reshape(lenx*leny*lenz, 1, lent)
        u_t = ops.conv1d(u_conv1d, self.dt_kernel)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, lenz, 1, lent-2)
        u_t = u_t.transpose(4, 3, 0, 1, 2)  # [step-2, c, height(Y), width(X)]

        # temporal derivatives - v
        v = output[:, 1:2, 2:-2, 2:-2, 2:-2]
        v_conv1d = v.transpose(2, 3, 4, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny*lenz, 1, lent)
        v_t = ops.conv1d(v_conv1d, self.dt_kernel)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, lenz, 1, lent-2)
        v_t = v_t.transpose(4, 3, 0, 1, 2)  # [step-2, c, height(Y), width(X)]

        # [step, c, height(Y), width(X), depth]
        u = output[0:-2, 0:1, 2:-2, 2:-2, 2:-2]
        # [step, c, height(Y), width(X)]
        v = output[0:-2, 1:2, 2:-2, 2:-2, 2:-2]

        # GS eqn
        du = 0.2
        dv = 0.1
        f = 0.025
        k = 0.055
        # compute residual
        f_u = (du*laplace_u - u*v**2 + f*(1-u) - u_t)
        f_v = (dv*laplace_v + u*v**2 - (f+k)*v - v_t)
        return self.loss(f_u, ops.zeros_like(f_u)) + self.loss(f_v, ops.zeros_like(f_v))

    def get_loss(self):
        """get loss"""
        output = self.get_output(self.timesteps_for_train)

        pred = output[::15, :, ::2, ::2, ::2]
        gt = self.truth[::15, :, ::2, ::2, ::2]
        idx = int(pred.shape[0] * 0.9)

        pred_tra, pred_val = pred[:idx], pred[idx:]  # prediction
        gt_tra, gt_val = gt[:idx], gt[idx:]  # ground truth

        loss_data = self.loss(pred_tra, gt_tra)
        loss_valid = self.loss(pred_val, gt_val)
        loss_ic = self.get_ic_loss()

        loss_phy = self.get_phy_loss(output)

        return 1.0*loss_data + 2.0*loss_ic, loss_data, loss_ic, loss_phy, loss_valid

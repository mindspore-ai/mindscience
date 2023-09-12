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
"""network definition"""
import numpy as np
from mindspore import ops, nn
import mindspore.numpy as mnp
from mindspore.dataset import Dataset

from sciai.architecture.basic_block import MLPShortcut
from sciai.common import TrainCellWithCallBack
from sciai.operators import grad
from sciai.utils import to_tensor


class OperatorNet(nn.Cell):
    """Cell for operator"""
    def __init__(self, branch_net, trunk_net):
        super(OperatorNet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def construct(self, u, x, t):
        """Network forward pass"""
        y = ops.stack([x, t], axis=-1)
        b_ = self.branch_net(u)
        t_ = self.trunk_net(y)
        outputs = ops.sum(b_ * t_, dim=-1)
        return outputs


class ResidualNet(nn.Cell):
    """Cell for residual calculation"""
    def __init__(self, operator_net, d, k):
        super(ResidualNet, self).__init__()
        self.d = d
        self.k = k
        self.operator_net = operator_net
        self.grad_s_net = grad(self.operator_net)
        self.s_xx_net = grad(grad(self.operator_net), output_index=1, input_index=1)

    def construct(self, u, x, t):
        """Network forward pass"""
        s = self.operator_net(u, x, t)
        _, _, s_t = self.grad_s_net(u, x, t)
        s_xx = self.s_xx_net(u, x, t)

        res = s_t - self.d * s_xx - self.k * s ** 2
        return res


# # Data Sampler

class DataGenerator(Dataset):
    """Data generator"""
    def __init__(self, u, y, s, dtype, batch_size=64):
        'Initialization'
        super(DataGenerator, self).__init__()
        self.u = u.asnumpy()
        self.y = y.asnumpy()
        self.s = s.asnumpy()
        self.n = u.shape[0]
        self.batch_size = batch_size
        self.dtype = dtype

    def __getitem__(self, index):
        idx = np.random.choice(self.n, self.batch_size)
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        u, y, outputs = to_tensor((u, y, s), self.dtype)
        return u, y, outputs


class LossNet(nn.Cell):
    """Cell for loss calculation"""
    def __init__(self, operator_net, residual_net):
        super(LossNet, self).__init__()
        self.operator_net = operator_net
        self.residual_net = residual_net

    def construct(self, *inputs):
        """Network forward pass"""
        u_ic, y_ic, outputs_ic, u_bc, y_bc, outputs_bc, u_res, y_res, outputs_res = inputs
        loss_ics = self.loss_ics(u_ic, y_ic, outputs_ic)
        loss_bcs = self.loss_bcs(u_bc, y_bc, outputs_bc)
        loss_res = self.loss_res(u_res, y_res, outputs_res)
        return loss_ics, loss_bcs, loss_res

    def loss_ics(self, u, y, outputs):
        s_pred = self.operator_net(u, y[:, 0], y[:, 1])
        loss = mnp.mean((outputs.flatten() - s_pred) ** 2)
        return loss

    def loss_bcs(self, u, y, outputs):
        # Compute forward pass
        s_pred = self.operator_net(u, y[:, 0], y[:, 1])
        # Compute loss
        loss = mnp.mean((outputs.flatten() - s_pred) ** 2)
        return loss

    def loss_res(self, u, y, outputs):
        # Compute forward pass
        pred = self.residual_net(u, y[:, 0], y[:, 1])
        # Compute loss
        loss = mnp.mean((outputs.flatten() - pred) ** 2)
        return loss


class PiDeepONet(nn.Cell):
    """Cell for pi-deeponet"""
    def __init__(self, branch_layers, trunk_layers, d, k):
        super(PiDeepONet, self).__init__()

        # Network initialization
        self.branch_net = MLPShortcut(branch_layers, weight_init="xavier_trunc_normal", bias_init="zeros",
                                      activation=ops.Tanh())
        self.trunk_net = MLPShortcut(trunk_layers, weight_init="xavier_trunc_normal", bias_init="zeros",
                                     activation=ops.Tanh())
        self.operator_net = OperatorNet(self.branch_net, self.trunk_net)
        self.residual_net = ResidualNet(self.operator_net, d, k)
        self.loss_net = LossNet(self.operator_net, self.residual_net)

        # Logger
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Optimize parameters in a loop
    def train(self, ics_dataset, bcs_dataset, res_dataset, args):
        """Train procedure"""
        exponential_decay_lr = nn.ExponentialDecayLR(args.lr, 0.9, 5000, is_stair=True)
        optimizer = nn.Adam(self.loss_net.trainable_params(), learning_rate=exponential_decay_lr)

        train_cell = TrainCellWithCallBack(self.loss_net, optimizer,
                                           time_interval=args.print_interval, loss_interval=args.print_interval,
                                           loss_names=("ic_loss", "bc_loss", "res_loss"),
                                           amp_level=args.amp_level)
        # Main training loop
        for it in range(args.epochs):
            u_ic, y_ic, outputs_ic = ics_dataset[it]
            u_bc, y_bc, outputs_bc = bcs_dataset[it]
            u_res, y_res, outputs_res = res_dataset[it]

            loss_ics, loss_bcs, loss_res = train_cell(u_ic, y_ic, outputs_ic, u_bc, y_bc, outputs_bc, u_res, y_res,
                                                      outputs_res)

            if it % 10 == 0:
                loss = loss_ics + loss_bcs + loss_res
                self.loss_log.append(loss.asnumpy())
                self.loss_ics_log.append(loss_ics.asnumpy())
                self.loss_bcs_log.append(loss_bcs.asnumpy())
                self.loss_res_log.append(loss_res.asnumpy())

    # Evaluates predictions at test points
    def predict_s(self, u_star, y_star):
        """Prediction"""
        s_pred = self.operator_net(u_star, y_star[:, 0], y_star[:, 1])
        return s_pred

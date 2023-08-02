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
# ==============================================================================
"""
User-defined wrapper for training and testing.
"""
import mindspore.common.dtype as mstype
from mindspore import ops, jit_class
from mindflow.pde import FlowWithLoss
from mindflow.loss import RelativeRMSELoss


@jit_class
class BurgersWithLoss(FlowWithLoss):
    """User-defined data-driven problems.

    Args:
        model (Cell): A training or testing model.
        t_out (int): Output time steps. Default: 1.
        loss_fn (Union[str, Cell]): Objective function. Default: "mse".
        pred_factor (float): Factor multiplied to prediction loss. Default: 5.0.
        recons_factor (str): Factor multiplied to reconstruction loss. Default: 0.5.
    """
    def __init__(self, model, t_out, loss_fn, pred_factor=5.0, recons_factor=0.5):
        super(BurgersWithLoss, self).__init__(model, loss_fn)
        self.t_out = t_out
        self.pred_factor = pred_factor
        self.recons_factor = recons_factor
        self.rrmse_loss = RelativeRMSELoss(reduction='mean')
        self.cat = ops.Concat(-1)
        self.cast = ops.Cast()

    def step(self, inputs):
        """Train the model on one batch of train dataset.

        Args:
            inputs (Array): Input data with shape e.g. :math:`[N,H,T]`.
        """
        l_recons, pred = 0, 0
        bs = inputs.shape[0]
        for t in range(self.t_out):
            im, im_re = self.model(inputs)
            im = self.cast(im, mstype.float32)
            l_recons += self.loss_fn(ops.reshape(im_re, (bs, -1)),
                                     ops.reshape(inputs, (bs, -1)))
            if t == 0:
                pred = im[..., -1:]
            else:
                pred = self.cat((pred, im))
            inputs = self.cat((inputs[..., 1:], im[..., -1:]))
        return pred, l_recons

    def get_loss(self, inputs, labels):
        """Calculate the loss, which is used to guide the gradient computing.

        Args:
            inputs (Array): Input data with shape e.g. :math:`[N,H,T]`.
            labels (Array): Label data with shape e.g. :math:`[N,H]` or :math:`[N,H,T]`.
        """
        if labels.ndim == 2:
            labels = labels[:, :, None]
        pred, l_recons = self.step(inputs)
        l_pred = self.loss_fn(pred, labels)
        loss = self.pred_factor * l_pred + self.recons_factor * l_recons
        # Reconstruction loss is accumulated alone t_out. Return its mean here.
        return loss, l_recons / self.t_out, l_pred

    def get_rel_loss(self, inputs, labels):
        """Calculate RelativeRMSELoss.

        Args:
            inputs (Array): Input data with shape e.g. :math:`[N,H,T]`.
            labels (Array): Label data with shape e.g. :math:`[N,H]` or :math:`[N,H,T]`.
        """
        if labels.ndim == 2:
            labels = labels[:, :, None]
        pred, l_recons = self.step(inputs)
        l_pred = self.rrmse_loss(pred, labels)
        # Reconstruction loss is accumulated alone t_out. Return its mean here.
        # RelativeRMSELoss only average the loss along N. Average loss along T here.
        return l_recons / self.t_out, l_pred / self.t_out

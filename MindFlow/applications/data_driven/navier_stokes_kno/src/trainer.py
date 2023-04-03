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
from mindspore import dtype as mstype
from mindspore import jit_class
from mindspore import ops
from mindspore import Tensor
from mindflow.pde import FlowWithLoss
from mindflow.loss import RelativeRMSELoss


@jit_class
class NavierStokesWithLoss(FlowWithLoss):
    """User-defined data-driven problems.

    Args:
        model (Cell): A training or testing model.
        t_out (int): Output time steps. Default: 1.
        loss_fn (Union[str, Cell]): Objective function. Default: "mse".
        data_format (str): Data format.
        pred_factor (float): Factor multiplied to prediction loss. Default: 5.0.
        recons_factor (str): Factor multiplied to reconstruction loss. Default: 0.5.
    """
    def __init__(self, model, t_out, loss_fn,
                 data_format="NHWTC", pred_factor=5.0, recons_factor=0.5):
        super(NavierStokesWithLoss, self).__init__(model, loss_fn)
        self.t_out = t_out
        self.data_format = data_format
        self.pred_factor = pred_factor
        self.recons_factor = recons_factor
        self.rrmse_loss = RelativeRMSELoss(reduction='mean')

    def step(self, xx, t_out=None):
        """Train the model on one batch of train dataset.

        Args:
            xx (Array): Input data with shape e.g. :math:`[N,H,W,T,C]`.
            t_out (int): Number of time steps to forward the model sequentially.
        """
        t_out = t_out or self.t_out
        bs = xx.shape[0]
        l_recons = 0.0
        pred_list = []
        for _ in range(t_out):
            x_flat = self._flatten(xx)
            pred, im_re = self.model(x_flat)
            if self.data_format == "NTCHW":
                t_in = xx.shape[1]
                pred = pred.expand_dims(axis=1)
                if t_in > 1:
                    xx = ops.concat([xx[:, 1:, ...], pred], axis=1)
                else:
                    xx = pred
            elif self.data_format == "NHWTC":
                t_in = xx.shape[-2]
                pred = pred.expand_dims(axis=-2)
                if t_in > 1:
                    xx = ops.concat([xx[..., 1:, :], pred], axis=-2)
                else:
                    xx = pred
            else:
                assert self.data_format == "NHWC"
                xx = pred
            pred_list.append(pred)
            # Calculate reconstruction loss for each time step.
            l_recons += self.loss_fn(ops.reshape(im_re, (bs, -1)),
                                     ops.reshape(xx, (bs, -1)))
        if self.data_format == 'NTCHW':
            pred_list = ops.concat(pred_list, axis=1)
        elif self.data_format == 'NHWTC':
            pred_list = ops.concat(pred_list, axis=-2)
        else:
            assert self.data_format == "NHWC"
            pred_list = pred_list[-1]
        return pred_list, l_recons

    def get_loss(self, inputs, labels):
        """Calculate the loss, which is used to guide the gradient computing.

        Args:
            inputs (Array): Input data with shape e.g. :math:`[N,H,W,T,C]`.
            labels (Array): Label data with shape e.g. :math:`[N,H,W,T,C]`.
        """
        pred, l_recons = self.step(inputs, self.t_out)
        l_pred = self.loss_fn(ops.flatten(pred), ops.flatten(labels))
        loss = self.pred_factor * l_pred + self.recons_factor * l_recons
        # Reconstruction loss is accumulated alone t_out. Return its mean here.
        return loss, l_recons / self.t_out, l_pred

    def test(self, inputs, labels, t_out=10):
        """Evaluate the model on the whole test dataset.

        Predict 1 time step each step, and use the prediction as input for next time step.
        Args:
            inputs (Array): Input data with shape e.g. :math:`[N,T0,H,W,T,C]`.
            labels (Array): Label data with shape e.g. :math:`[N,T0,H,W,T,C]`.
            t_out (int): Number of time steps to predict sequentially.
        """
        l_recons_all, l_pred_all = 0.0, 0.0
        num = inputs.shape[0]  # i.e. number of samples
        t_start = inputs.shape[1] - t_out

        self.model.set_train(False)
        for n in range(num):
            # Predict 1 time step each step, and use the prediction as input for next time step.
            # Only part of the inputs is used here.
            xx = Tensor(inputs[n, t_start, ...], dtype=mstype.float32)
            # Merge two time dimensions, e.g. [19(t),64,64,1(t),1].
            yy_ = Tensor(labels[n, t_start:, ...], dtype=mstype.float32)
            shape = list(yy_.shape)
            if self.data_format == 'NTCHW':
                t1, t2, c, h, w = shape
                yy = ops.reshape(yy_, (t1 * t2, c, h, w))
            elif self.data_format == 'NHWTC':
                t1, h, w, t2, c = shape
                yy_ = ops.transpose(yy_, (1, 2, 0, 3, 4))
                yy = ops.reshape(yy_, (h, w, t1 * t2, c))
            else:
                assert self.data_format == "NHWC"
                yy = yy_[-1, :]

            pred, l_recons = self.step(xx[None, ...], t_out)  # batch_size == 1
            l_pred = self.rrmse_loss(ops.flatten(pred), ops.reshape(yy, (1, -1)))
            # Reconstruction loss is accumulated alone t_out. Return its mean here.
            l_recons_all += (l_recons.asnumpy() / t_out)
            # RelativeRMSELoss only average the loss along N. Average loss along T here.
            l_pred_all += (l_pred.asnumpy() / t_out)

        self.model.set_train()
        return l_recons_all / num, l_pred_all / num

    def _flatten(self, inputs):
        """Flatten T and C dimensions.

        Args:
            inputs (Array): Inputs has to be with data_format NHWTC, NTCHW or NHWC.
        """
        # [bs, t_in, c, x1, x2, ...] -> [bs, t_in*c, x1, x2, ...]
        dim = len(inputs.shape) - 3
        if self.data_format == 'NTCHW':  # NTCHW to NHWTC
            inputs = ops.transpose(inputs, tuple([0] + list(range(3, dim + 3)) + [1, 2]))

        inp_shape = list(inputs.shape)
        inp_shape = inp_shape[:3]
        inp_shape.append(-1)
        inputs = ops.reshape(inputs, tuple(inp_shape))

        if self.data_format == 'NTCHW':  # NHWTC to NTCHW
            inputs = ops.transpose(inputs, tuple([0, dim + 1] + list(range(1, dim + 1))))
        return inputs

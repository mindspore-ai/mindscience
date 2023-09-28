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
"""Allen Cahn"""
import numpy as np
from sympy import diff, symbols, Function

from mindspore import ops, Tensor
from mindspore import dtype as mstype

from mindflow.pde import PDEWithLoss, sympy_to_mindspore
from mindflow.loss import get_loss_metric
from mindflow.cell import MultiScaleFCSequential


class AllenCahn(PDEWithLoss):
    r"""
    Allen Cahn problem based on PDEWithLoss

    Args:
        model (Cell): network for training.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, model, loss_fn="mse"):
        self.x, self.t = symbols("x t")
        self.u = Function("u")(self.x, self.t)
        self.in_vars = [self.x, self.t]
        self.out_vars = [self.u]
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn
        self.pde_nodes = sympy_to_mindspore(
            self.pde(), self.in_vars, self.out_vars)
        model.set_output_transform(self.output_transform)
        super(AllenCahn, self).__init__(model, self.in_vars, self.out_vars)

    def output_transform(self, x, out):
        """
        Define output transforma function in Allen Cahn Equation

        Args:
            x (Tensor): network input
            out (Tensor): network output
        """

        return x[:, 0:1] ** 2 * ops.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * out

    def force_function(self, u):
        """
        Define forcing function in Allen Cahn Equation

        Args:
            u (Tensor)
        """
        return 5 * (u - u ** 3)

    def pde(self):
        """
        Define Allen Cahn equation
        """

        d = 0.001
        loss_1 = (
            self.u.diff(self.t)
            - d * diff(self.u, (self.x, 2))
            - self.force_function(self.u)
        )
        return {"loss_1": loss_1}

    def get_loss(self, pde_data):
        """
        Compute loss of 3 parts: governing equation, initial condition and boundary conditions.

        Args:
            pde_data (Tensor): the input data of governing equations.
            ic_data (Tensor): the input data of initial condition.
            bc_data (Tensor): the input data of boundary condition.
        """
        pde_res = ops.Concat(1)(self.parse_node(
            self.pde_nodes, inputs=pde_data))
        pde_loss = self.loss_fn(
            pde_res, Tensor(np.array([0.0]).astype(np.float32), mstype.float32)
        )

        return pde_loss


class MultiScaleFCSequentialOutputTransform(MultiScaleFCSequential):
    r"""
    The multi-scale fully conneted network. Apply a transform to the network outputs, i.e.,
    outputs = transform(inputs, outputs).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 neurons,
                 residual=True,
                 act="sin",
                 weight_init='normal',
                 weight_norm=False,
                 has_bias=True,
                 bias_init="default",
                 num_scales=4,
                 amp_factor=1.0,
                 scale_factor=2.0,
                 input_scale=None,
                 input_center=None,
                 latent_vector=None,
                 output_transform=None
                 ):
        super(MultiScaleFCSequentialOutputTransform, self).__init__(in_channels, out_channels,
                                                                    layers, neurons, residual,
                                                                    act, weight_init,
                                                                    weight_norm, has_bias,
                                                                    bias_init, num_scales,
                                                                    amp_factor, scale_factor,
                                                                    input_scale, input_center,
                                                                    latent_vector)
        self.output_transform = output_transform

    def set_output_transform(self, output_transform):
        """set output transform function"""
        self.output_transform = output_transform

    def construct(self, x):
        """construct network"""
        x = self.input_scale(x)
        if self.latent_vector is not None:
            batch_size = x.shape[0]
            latent_vectors = self.latent_vector.view(
                self.num_scenarios, 1, self.latent_size)
            latent_vectors = latent_vectors.repeat(batch_size // self.num_scenarios,
                                                   axis=1).view((-1, self.latent_size))
            x = self.concat((x, latent_vectors))
        out = 0
        for i in range(self.num_scales):
            x_s = x * self.scale_coef[i]
            out = out + self.cast(self.cell_list[i](x_s), mstype.float32)
        if self.output_transform is None:
            return out

        return self.output_transform(x, out)

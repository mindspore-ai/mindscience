# Copyright 2023 @ Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""VAE model"""
from mindspore import nn
import mindspore.ops as ops


class VAE(nn.Cell):
    """
    VAE网络结构
    """

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super(VAE, self).__init__()

        assert isinstance(encoder_layer_sizes, list)
        assert isinstance(latent_size, int)
        assert isinstance(decoder_layer_sizes, list)

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size)

    def construct(self, x):
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)
        output = (recon_x, x, means, log_var, z)
        return output

    def reparameterize(self, mu, log_var):
        std = ops.exp(0.5 * log_var)
        eps = ops.StandardNormal()((std.shape[0], std.shape[1]))

        return ops.add(mu, ops.mul(eps, std))

    def inference(self, z):
        recon_x = self.decoder(z)

        return recon_x


class Encoder(nn.Cell):
    """Encoder"""
    def __init__(self, layer_sizes, latent_size):
        super(Encoder, self).__init__()

        self.mlp = nn.SequentialCell()
        for _, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.mlp.append(
                nn.Dense(in_size, out_size))
            self.mlp.append(nn.ReLU())

        self.dense_means = nn.Dense(layer_sizes[-1], latent_size)
        self.dense_log_var = nn.Dense(layer_sizes[-1], latent_size)

    def construct(self, x):
        x = self.mlp(x)
        means = self.dense_means(x)
        log_vars = self.dense_log_var(x)

        return means, log_vars


class Decoder(nn.Cell):
    """Decoder"""
    def __init__(self, layer_sizes, latent_size):

        super(Decoder, self).__init__()

        self.mlp = nn.SequentialCell()
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.mlp.append(
                nn.Dense(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.mlp.append(nn.ReLU())
            else:
                self.mlp.append(nn.Sigmoid())

    def construct(self, z):
        x = self.mlp(z)
        return x


class CvaeLoss(nn.Cell):
    """Cvaeloss"""

    def __init__(self, option, alpha, beta):
        super(CvaeLoss, self).__init__()
        self.option = option
        self.alpha = alpha
        self.beta = beta

    def regularization(self, mu, logvar):
        return -0.5 * ops.ReduceSum()(1 + logvar - ops.Pow()(mu, 2) - ops.Exp()(logvar))

    def guassian_loss(self, recon_x, x):
        weights = x * self.alpha + (1 - x)
        loss_ = x - recon_x
        loss_ = ops.ReduceSum()(weights * loss_ * loss_)
        return loss_

    def bec_loss(self, recon_x, x):
        eps = 1e-8
        loss_ = -ops.ReduceSum()(self.alpha * ops.log(recon_x + eps) * x + ops.log(1 - recon_x + eps) * (1 - x))
        return loss_

    def construct(self, input_, label):
        recon_x, _, mu, logvar, _ = input_
        if self.option == 1:
            loss_ = self.guassian_loss(recon_x, label) + self.regularization(mu, logvar) * self.beta
        else:
            loss_ = self.bec_loss(recon_x, label) + self.regularization(mu, logvar) * self.beta
        return loss_


class CustomTrainOneStepCell(nn.Cell):
    """CustomTrainOneStepCell"""

    def __init__(self, network_, optimizer_):
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network_
        self.network.set_grad()
        self.optimizer_ = optimizer_
        self.weights = self.optimizer_.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss_ = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        self.optimizer_(grads)
        return loss_

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

"""Network architectures for AUQ PINNs"""
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import XavierNormal
from sciai.architecture import MLP, MSE
from sciai.operators.derivatives import grad


def get_all_networks(args, dataset=None):
    encoder, decoder, discriminator = get_networks(args)
    generator_loss = GeneratorLossNet(args, encoder, decoder, discriminator, dataset)
    discriminator_loss = DiscriminatorLossNet(decoder, discriminator)
    return encoder, decoder, discriminator, generator_loss, discriminator_loss


def get_networks(args):
    encoder = EncoderNet(args.layers_q)
    decoder = DecoderNet(args.layers_p)
    discriminator = DiscriminatorNet(args.layers_t)
    if args.load_ckpt:
        ms.load_checkpoint(args.load_ckpt_path[1], encoder, choice_func=_choose_encoder)
        ms.load_checkpoint(args.load_ckpt_path[1], decoder, choice_func=_choose_decoder)
        ms.load_checkpoint(args.load_ckpt_path[0], discriminator)

    return encoder, decoder, discriminator


def _choose_encoder(param_name):
    return param_name.startswith("optimizer.encoder")


def _choose_decoder(param_name):
    return param_name.startswith("network.decoder")


class DiscriminatorNet(nn.Cell):
    def __init__(self, layers):
        super().__init__()
        self.mlp = MLP(layers, weight_init=XavierNormal(), bias_init="zeros", activation="tanh")

    def construct(self, x, y):
        """Network forward pass"""
        t = self.mlp(ops.cat((x, y), axis=1))
        return t


# Encoder: q(z|x,y)
class EncoderNet(nn.Cell):
    def __init__(self, layers):
        super().__init__()
        self.mlp = MLP(layers, weight_init=XavierNormal(), bias_init="zeros", activation="tanh")

    def construct(self, x, y):
        """Network forward pass"""
        z = self.mlp(ops.cat((x, y), axis=1))
        return z


# Decoder: p(y|x,z)
class DecoderNet(nn.Cell):
    def __init__(self, layers):
        super().__init__()
        self.mlp = MLP(layers, weight_init=XavierNormal(), bias_init="zeros", activation="tanh")

    def construct(self, x, z):
        """Network forward pass"""
        y = self.mlp(ops.cat((x, z), axis=1))
        return y


class GeneratorLossNet(nn.Cell):
    """Generator Loss Network"""
    def __init__(self, args, encoder_net, decoder_net, discriminator_net, dataset):
        super().__init__()
        self.lam = args.lam
        self.beta = args.beta
        self.encoder = encoder_net
        self.decoder = decoder_net
        self.decoder_x = grad(self.decoder, output_index=0, input_index=0)
        self.decoder_xx = grad(self.decoder_x, output_index=0, input_index=0)
        self.discriminator = discriminator_net
        self.mse = MSE()
        if dataset:
            self.x_std = dataset.x_std
            self.x_mean = dataset.x_mean
            self.jacobian = dataset.jacobian

    # x_col and x_bound should be normalized before calling it
    def construct(self, x_col, z_col, x_bound, z_bound):
        """Network forward pass"""
        y_bound_pred = self.decoder(x_bound, z_bound)
        z_bound_encoder = self.encoder(x_bound, y_bound_pred)
        t_pred = self.discriminator(x_bound, y_bound_pred)

        kl = ops.reduce_mean(t_pred)
        log_q = - self.mse(z_bound - z_bound_encoder)

        # Physics-Informed residual on the collocation points
        u = self.decoder(x_col, z_col)
        u_x = self.decoder_x(x_col, z_col)
        u_xx = self.decoder_xx(x_col, z_col)
        x_real = x_col * self.x_std + self.x_mean
        y_col_pred = - np.pi ** 2 * ops.sin(np.pi * x_real) \
                     - np.pi * ops.cos(np.pi * x_real) * ops.pow(ops.sin(np.pi * x_real), 2)
        residual = ops.pow(self.jacobian, 2) * u_xx - self.jacobian * ops.pow(u, 2) * u_x - y_col_pred

        loss_f = self.mse(residual)
        loss = kl + (1.0 - self.lam) * log_q + self.beta * loss_f

        return loss, kl, (1.0 - self.lam) * log_q, self.beta * loss_f


class DiscriminatorLossNet(nn.Cell):
    """Discriminator Loss Network"""
    def __init__(self, decoder_net, discriminator_net):
        super().__init__()
        self.decoder = decoder_net
        self.discriminator = discriminator_net

    def construct(self, x, y, z):
        """Network forward pass"""
        y_pred = self.decoder(x, z)

        t_real = self.discriminator(x, y)
        t_fake = self.discriminator(x, y_pred)

        t_real = ops.sigmoid(t_real)
        t_fake = ops.sigmoid(t_fake)

        t_loss = - ops.reduce_mean(ops.log(1 - t_real + 1e-8) + ops.log(t_fake + 1e-8))

        return t_loss

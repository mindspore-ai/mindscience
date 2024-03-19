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
"""Loss"""
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor

from .utils import warp, make_grid


class GANLoss(nn.Cell):
    """ Adversarial loss definition """
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.zero_tensor = None
        self.opt = opt

    def get_target_tensor(self, inputs, target_is_real):
        if target_is_real:
            return ops.ones(inputs.shape, inputs.dtype)
        return ops.zeros(inputs.shape, inputs.dtype)

    def construct(self, inputs, target_is_real):
        target_tensor = self.get_target_tensor(inputs, target_is_real)
        loss = ops.binary_cross_entropy_with_logits(inputs,
                                                    target_tensor,
                                                    weight=ops.ones(inputs.shape, inputs.dtype),
                                                    pos_weight=ops.ones(inputs.shape, inputs.dtype))
        return loss


class WeightDistance(nn.Cell):
    """ Weighted L1 distance """

    def construct(self, true_frame, pred_frame, weights):
        loss = ops.mean(ops.abs(true_frame - pred_frame) * weights)
        return loss


class GenerateLoss(nn.Cell):
    """ Generator loss in generation module """
    def __init__(self, generator, discriminator):
        super(GenerateLoss, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gan_loss = GANLoss()
        self.weighted_distance = WeightDistance()
        self.pool_q = nn.MaxPool2d(kernel_size=4, stride=2)

    def construct(self, inputs, evo_result, noise, real_image, beta=8., gamma=20., k=4):
        """loss function"""
        fake_image = self.generator(inputs, evo_result, noise[..., 0])
        fake_inputs = ops.concat([inputs, fake_image], axis=1)
        pred_fake = self.discriminator(fake_inputs)
        adv = self.gan_loss(pred_fake, True)
        ensemble_image = 0.
        for i in range(k):
            ensemble_image += self.pool_q(self.generator(inputs, evo_result, noise[..., i + 1]))
        ensemble_image = ops.div(ensemble_image, k)
        real_image = self.pool_q(real_image)
        weights = ops.where(real_image > 23., 24., real_image + 1)
        pool = self.weighted_distance(real_image, ensemble_image, weights)
        g_losses = beta * adv + gamma * pool
        return g_losses


class DiscriminatorLoss(nn.Cell):
    """ Discriminator loss in generation module"""
    def __init__(self, generator, discriminator):
        super(DiscriminatorLoss, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gan_loss = GANLoss()

    def construct(self, inputs, evo_result, noise, real_image):
        fake_image = self.generator(inputs, evo_result, noise)
        fake_inputs = ops.concat([inputs, fake_image], axis=1)
        real_inputs = ops.concat([inputs, real_image], axis=1)
        pred_fake = self.discriminator(fake_inputs)
        pred_real = self.discriminator(real_inputs)
        fake_loss = self.gan_loss(pred_fake, False)
        real_loss = self.gan_loss(pred_real, True)
        return fake_loss + real_loss


class MotionLossNet(nn.Cell):
    """ Motion regularization """
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super(MotionLossNet, self).__init__()
        kernel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(
            np.float32)
        kernel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(
            np.float32)
        self.weight_vt1 = self.get_kernels(kernel_v, out_channels)
        self.weight_vt2 = self.get_kernels(kernel_h, out_channels)
        self.conv_2d_v = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, has_bias=False)
        self.conv_2d_v.weight.set_data(Tensor(self.weight_vt1, ms.float32))
        for w in self.conv_2d_v.trainable_params():
            w.requires_grad = False
        self.conv_2d_h = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, has_bias=False)
        self.conv_2d_h.weight.set_data(Tensor(self.weight_vt2, ms.float32))
        for w in self.conv_2d_h.trainable_params():
            w.requires_grad = False

    @staticmethod
    def get_kernels(kernel, repeats):
        kernel = np.expand_dims(kernel, axis=(0, 1))
        kernels = [kernel] * repeats
        kernels = np.concatenate(kernels, axis=0)
        return kernels

    def calc_diff_v(self, image, weights):
        diff_v1 = self.conv_2d_v(image)
        diff_v2 = self.conv_2d_h(image)
        lambda_v = diff_v1 ** 2 + diff_v2 ** 2
        lambda_v = lambda_v * weights
        loss = ops.sum(lambda_v)
        return loss

    def custom_2d_conv_sobel(self, image, weights):
        motion_loss1 = self.calc_diff_v(image, weights)
        motion_loss2 = self.calc_diff_v(image, weights)
        loss = ops.div(motion_loss1 + motion_loss2, image.shape[0] * image.shape[-1] * image.shape[-2])
        return loss

    def construct(self, motion, weights):
        loss1 = self.custom_2d_conv_sobel(motion[:, :1], weights)
        loss2 = self.custom_2d_conv_sobel(motion[:, 1:], weights)
        return loss1 + loss2


class EvolutionLoss(nn.Cell):
    """ Evolution loss definition"""
    def __init__(self, model, config):
        super(EvolutionLoss, self).__init__()
        self.config = config
        self.model = model
        self.loss_fn_accum = WeightDistance()
        self.loss_fn_motion = MotionLossNet(in_channels=1, out_channels=1, kernel_size=3)
        self.t_in = self.config.get('data').get("t_in", 9)
        self.t_out = self.config.get('data').get('t_out', 20)
        sample_tensor = np.zeros((1,
                                  1,
                                  self.config.get('data').get("h_size", 512),
                                  self.config.get('data').get("w_size", 512))).astype(np.float32)
        self.grid = Tensor(make_grid(sample_tensor), ms.float32)
        self.lamb = float(config.get('optimizer-evo').get("motion_lambda", 1e-2))

    def construct(self, inputs):
        """last frame of inputs"""
        intensity, motion = self.model(inputs)
        batch, _, height, width = inputs.shape
        motion_ = motion.reshape(batch, self.t_out, 2, height, width)
        intensity_ = intensity.reshape(batch, self.t_out, 1, height, width)
        last_frame = inputs[:, (self.t_in - 1):self.t_in, :, :]
        grid = self.grid.tile((batch, 1, 1, 1))
        accum = 0
        motion = 0
        for i in range(self.t_out):
            next_frame = inputs[:, self.t_in + i, :, :]
            weights = ops.where(next_frame > 23., 24., next_frame + 1)
            xt_1 = warp(last_frame, motion_[:, i], grid, mode="bilinear", padding_mode="border")
            accum += self.loss_fn_accum(next_frame, xt_1[:, 0], weights)
            last_frame = warp(last_frame, motion_[:, i], grid, mode="nearest", padding_mode="border")
            last_frame = last_frame + intensity_[:, i]
            accum += self.loss_fn_accum(next_frame, last_frame[:, 0], weights)
            last_frame = ops.stop_gradient(last_frame)
            motion += self.loss_fn_motion(motion_[:, i], weights)
        loss = ops.div(accum + self.lamb * motion, self.t_out)
        return loss

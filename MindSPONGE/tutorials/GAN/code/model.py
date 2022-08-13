# Copyright 2021-2022 @ Changping Laboratory &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next GEneration molecular modelling.
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
"""model"""
import numpy as np
import mindspore.numpy as mnp
from mindspore import (nn, ops)
from mindspore.ops.composite import GradOperation
from mindspore.common.initializer import Initializer


def _assignment(arr, num):
    """assignment"""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


class Normal(Initializer):
    """将模型权重从均值为0 标准差为0.02的正态分布中随机初始化"""

    def __init__(self, mean=0.0, sigma=0.02):
        super(Normal, self).__init__()
        self.sigma = sigma
        self.mean = mean

    def _initialize(self, arr):
        np.random.seed(666)
        arr_normal = np.random.normal(self.mean, self.sigma, arr.shape)
        _assignment(arr, arr_normal)


def convt(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="same"):
    """定义转置卷积层"""
    weight_init = Normal(mean=0, sigma=0.02)
    return nn.Conv2dTranspose(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              weight_init=weight_init, has_bias=False, pad_mode=pad_mode)


def bn(num_features):
    """定义BatchNorm2d层"""
    gamma_init = Normal(mean=1, sigma=0.02)
    return nn.BatchNorm2d(num_features=num_features, gamma_init=gamma_init)


class Generator(nn.Cell):
    """DCGAN网络生成器"""

    def __init__(self, input_dim=64):
        super(Generator, self).__init__()

        self.transform_input = nn.SequentialCell()
        self.transform_input.append(nn.Dense(input_dim, 1024))
        self.transform_input.append(nn.BatchNorm1d(1024))
        self.transform_input.append(nn.Dense(1024, 7*7*128))

        self.generator = nn.SequentialCell()
        self.generator.append(convt(128, 64, 4, 2, 0))
        self.generator.append(bn(64))
        self.generator.append(nn.ReLU())

        self.generator.append(convt(64, 1, 4, 2, 0))
        self.generator.append(nn.Tanh())

    def construct(self, z):
        # (B,64) -> (B,7*7*7*128):
        x = self.transform_input(z)

        # (B,128,7,7):
        x = mnp.reshape(x, (-1, 128, 7, 7))

        # (B,1,28,28):
        y = self.generator(x)
        return y


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="pad"):
    """定义卷积层"""
    weight_init = Normal(mean=0, sigma=0.02)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight_init, has_bias=False, pad_mode=pad_mode)


class Discriminator(nn.Cell):
    """
    DCGAN网络判别器
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.SequentialCell()
        self.discriminator.append(conv(1, 64, 4, 2, 1))
        self.discriminator.append(nn.LeakyReLU())
        self.discriminator.append(conv(64, 128, 4, 2, 1))
        self.discriminator.append(nn.LeakyReLU())

        self.shared_mlp = nn.SequentialCell()
        self.shared_mlp.append(
            nn.Dense(128*7*7, 1024, activation=nn.LeakyReLU()))
        self.shared_mlp.append(nn.Dense(1024, 128, activation=nn.LeakyReLU()))

        self.critic_net = nn.Dense(128, 1)
        self.info_net = nn.Dense(128, 10)

    def info_fn(self, x):
        """
        info_fn
        x:(B,1,28,28)
        """

        # (B,128,7,7):
        feat = self.discriminator(x)

        # (B,128*7*7):
        feat = mnp.reshape(feat, (feat.shape[0], -1))
        # (B,128):
        feat = self.shared_mlp(feat)
        # (B,1):
        info = self.info_net(feat)

        return info

    def critic_fn(self, x):
        """
        critic_fn
        x:(B,1,28,28)
        """

        # (B,128,7,7):
        feat = self.discriminator(x)

        # (B,128*7*7):
        feat = mnp.reshape(feat, (feat.shape[0], -1))

        # (B,128):
        feat = self.shared_mlp(feat)
        # (B,1):
        critic = self.critic_net(feat)

        return critic

    def construct(self, x):
        """
        construct
        x:(B,1,28,28)
        """

        # (B,128,7,7):
        feat = self.discriminator(x)
        # (B,128*7*7):
        feat = mnp.reshape(feat, (feat.shape[0], -1))

        # (B,128):
        feat = self.shared_mlp(feat)
        # (B,1):
        critic = self.critic_net(feat)
        # ():
        y = mnp.sum(critic)
        return y


# pylint: disable=invalid-name
class WithLossCellG(nn.Cell):

    """连接生成器和损失"""

    def __init__(self, netD, netG, info_wt=1.0):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.info_wt = info_wt  # Maximize MutualInfo.
        self.xent = nn.SoftmaxCrossEntropyWithLogits(
            sparse=True, reduction="none")

    def gan_fn(self, t):
        y = -ops.Softplus()(-t)
        return y

    def construct(self, latent_code, cat_c):
        """构建生成器损失计算结构"""
        # cat_c: (B,):
        # latent_code (B,Nz+Nc):

        fake_data = self.netG(latent_code)
        critic = self.netD.critic_fn(fake_data)
        info_logits = self.netD.info_fn(fake_data)

        # (B,1):
        _loss_critic = self.gan_fn(critic)
        # (B,):
        loss_critic = mnp.squeeze(_loss_critic, 1)

        # (B,):
        _loss_info = self.xent(info_logits, cat_c)
        loss_info = _loss_info

        # (B,):
        loss = loss_critic + self.info_wt*loss_info
        return loss


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = GradOperation()

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x)


# pylint: disable=invalid-name
class WithLossCellD(nn.Cell):
    """连接判别器和损失"""

    def __init__(self, netD, netG, info_wt=1.0, grad_wt=10.0):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.info_wt = info_wt  # Maximize MutualInfo.
        self.grad_wt = grad_wt  # R1-regularization of GAN
        self.xent = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")

        self.grad_D = GradNetWrtX(self.netD)

    def gan_fn(self, t):
        y = -ops.Softplus()(-t)
        return y

    def construct(self, real_data, latent_code, cat_c):
        """构建判别器损失计算结构"""

        critic_real = self.netD.critic_fn(real_data)

        # (B,1,28,28):
        fake_data = self.netG(latent_code)
        fake_data = ops.stop_gradient(fake_data)

        critic_fake = self.netD.critic_fn(fake_data)
        info_fake = self.netD.info_fn(fake_data)

        # (B,1):
        value_critic1 = self.gan_fn(critic_fake)
        value_critic2 = self.gan_fn(-critic_real)
        # (B,):
        value_critic1 = mnp.squeeze(value_critic1, 1)
        value_critic2 = mnp.squeeze(value_critic2, 1)

        grad_real = self.grad_D(real_data)

        # (B,1,28,28) -> (B,):
        grad_norm = mnp.sum(mnp.square(grad_real), (1, 2, 3))
        value_grad = -self.grad_wt*grad_norm

        # (B,):
        value_info = self.info_wt * (-self.xent(info_fake, cat_c))

        # (B,):
        value = value_critic1 + value_critic2 + value_grad + value_info

        loss = -value
        return loss


# pylint: disable=invalid-name
class GAN(nn.Cell):
    """GAN"""

    def __init__(self, myTrainOneStepCellForD, myTrainOneStepCellForG):
        super(GAN, self).__init__(auto_prefix=True)
        self.myTrainOneStepCellForD = myTrainOneStepCellForD
        self.myTrainOneStepCellForG = myTrainOneStepCellForG

    def pre_process_img(self, x):
        """preprocess images"""

        _min = 0.
        _max = 256.
        y = (2*x - (_max+_min))/(_max-_min)
        return y

    def construct(self, real_data, random_z, cat_c):
        """construct"""

        # cat_c: (B,); real_data:(B,28,28), random_z:(B,Nz)
        cat_c = ops.cast(cat_c, mnp.int32)
        random_c = nn.OneHot(axis=-1, depth=10)(cat_c)

        # (B,Nz+Nc):
        random_c = ops.cast(random_c, mnp.float32)
        latent_code = mnp.concatenate((random_z, random_c), axis=-1)

        # Normalize data:
        real_data = self.pre_process_img(real_data)
        # (B,1,28,28):
        real_data = mnp.expand_dims(real_data, 1)

        output_D = self.myTrainOneStepCellForD(real_data, latent_code, cat_c).view(-1)
        netD_loss = output_D.mean()

        output_G = self.myTrainOneStepCellForG(latent_code, cat_c).view(-1)
        netG_loss = output_G.mean()
        return netD_loss, netG_loss

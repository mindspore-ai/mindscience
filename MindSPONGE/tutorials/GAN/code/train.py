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
"""
GAN tutorial
"""

import numpy as np
import mindspore.dataset as ds
import mindspore.numpy as mnp
from mindspore import (Tensor, context,
                       nn, save_checkpoint, set_seed)
from mnist import MNIST

from model import GAN, Discriminator, Generator, WithLossCellD, WithLossCellG

context.set_context(mode=context.GRAPH_MODE)  # Train
set_seed(666)
np.random.seed(666)

# pylint: disable=invalid-name
batchsize = 64
random_dim = 50
netD = Discriminator()
netG = Generator(input_dim=random_dim)

# 实例化WithLossCell
netG_with_criterion = WithLossCellG(netD, netG, info_wt=1.0)
netD_with_criterion = WithLossCellD(netD, netG, info_wt=1.0, grad_wt=1.0)

# 为生成器和判别器设置优化器
lr_D = 2e-4
lr_G = 1e-3
beta1_D = 0.5
beta1_G = 0.5

optimizerD = nn.Adam(netD.trainable_params(),
                     learning_rate=lr_D, beta1=beta1_D)
optimizerG = nn.Adam(netG.trainable_params(),
                     learning_rate=lr_G, beta1=beta1_G)

# 实例化TrainOneStepCell
myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optimizerD)
myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optimizerG)

# 创建一批隐向量用来观察G
fixed_noise_normal = Tensor(np.random.randn(
    50, random_dim-10), dtype=mnp.float32)
fixed_noise_int = Tensor(np.repeat(np.arange(10), 5), dtype=mnp.int32)

# 实例化GAN网络
infogan = GAN(myTrainOneStepCellForD, myTrainOneStepCellForG)

infogan.set_train(True)  # Setup BatchNorm

# 创建数据迭代器
# 请先下载mnist数据集
mndata = MNIST('../data/')
mndata.gz = True
images, labels = mndata.load_training()

images = np.array(images).reshape((-1, 28, 28)).astype(np.float32)
labels = np.array(labels).astype(np.float32)

dataset = ds.NumpySlicesDataset(
    data=(images, labels),
    column_names=["images", "labels"],
    shuffle=True
)
dataset = dataset.batch(batchsize, drop_remainder=True)
dataset = dataset.repeat(100)

step = 0
G_losses = []
D_losses = []
image_list = []
for _d in dataset:
    step += 1

    # Pre-process Input Data:
    image = Tensor(_d[0], mnp.float32)
    random_z = np.random.randn(batchsize, random_dim-10)
    random_z = Tensor(random_z, mnp.float32)
    random_c = np.random.randint(0, 10, size=(batchsize,))
    random_c = Tensor(random_c, mnp.int32)

    netD_loss, netG_loss = infogan(image, random_z, random_c)

    if step % 50 == 0:
        # 输出训练记录
        print('[%d]\tLoss_D: %.4f\tLoss_G: %.4f' %
              (step, netD_loss.asnumpy(), netG_loss.asnumpy()))
        D_losses.append(netD_loss.asnumpy())
        G_losses.append(netG_loss.asnumpy())

    if step % 1000 == 0 or step == 1:
        fixed_z = nn.OneHot(
            axis=-1, depth=10)(fixed_noise_int).astype(mnp.float32)
        fixed_z = mnp.concatenate((fixed_noise_normal, fixed_z), -1)
        img = netG(fixed_z)
        image_list.append(np.squeeze(img.asnumpy(), 1))

        ckpt_name = "../ckpts/" + f"train_epoch_{step}.ckpt"
        save_checkpoint(infogan, ckpt_name)

        ckpt_name = "../ckpts/" + f"generator_epoch_{step}.ckpt"
        save_checkpoint(netG, ckpt_name)

        img_name = "../results/" + f"images.npy"
        np.save(img_name, image_list)

        np.savetxt('../results/losses.log',
                   np.stack((D_losses, G_losses), axis=1))

    # 我们暂时训练10000步；
    # 为了更好效果，可以加载checkpoint(参考Line 73~75的代码)进行更长时间的训练
    if step == 10000:
        break

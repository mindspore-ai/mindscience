# Copyright 2025 Huawei Technologies Co., Ltd
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
"""autoencoder model for MNIST dataset"""
import time
import os

import matplotlib.pyplot as plt
import numpy as np
from mindspore import Tensor, ops, amp, load_checkpoint, load_param_into_net, nn, jit, save_checkpoint

from dataset import get_dataset, SampleScaler, CKPT_PATH, load_mnist

LATENT_DIM = 16
HIDDEN_DIM = 128
INPUT_DIM = OUTPUT_DIM = 28*28


class Encoder(nn.Cell):
    """encoder"""

    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.layers = nn.SequentialCell(
            nn.Dense(input_size, hidden_size),
            nn.Tanh(),
            nn.Dense(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dense(hidden_size, latent_size)
        )

    def construct(self, x):  # x: bs,input_size
        x = self.layers(x)
        return x


class Decoder(nn.Cell):
    """decoder"""

    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.SequentialCell(
            nn.Dense(latent_size, hidden_size),
            nn.Tanh(),
            nn.Dense(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dense(hidden_size, output_size),
            nn.Sigmoid()
        )

    def construct(self, x):  # x:bs,latent_size
        x = self.layers(x)
        return x


class AE(nn.Cell):
    """AutoEncoder model"""

    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(AE, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, output_size)

    def construct(self, x):  # x: bs,input_size
        feat = self.encoder(x)  # feat: bs,latent_size
        re_x = self.decoder(feat)  # re_x: bs, output_size
        return re_x


def train_ae():
    """train autoencoder on MNIST dataset"""
    criterion = nn.MSELoss()
    model = AE(INPUT_DIM, OUTPUT_DIM, LATENT_DIM, HIDDEN_DIM)

    def forward_fn(data):
        pred = model(data)
        loss = criterion(data, pred)
        return loss

    lr = 0.001
    optimizer = nn.Adam(model.trainable_params(), lr, 0.9, 0.99)
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)

    @jit
    def train_step(data):
        loss, grads = grad_fn(data)
        is_finite = amp.all_finite(grads)
        if is_finite:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    batch_size = 256
    train_dataset = get_dataset(batch_size=batch_size)
    epochs = 200
    for epoch in range(1, epochs+1):
        time_beg = time.time()
        for samples in train_dataset:
            loss = train_step(samples)
        time_end = time.time()
        print(f'epoch: {epoch}  loss: {loss} time: {time_end - time_beg}')
    x = []
    model.set_train(False)
    for samples in train_dataset:
        x.append(samples)
    x = Tensor(np.concatenate(x, axis=0))
    save_checkpoint(model, CKPT_PATH)
    latent = model.encoder(x).numpy()
    np.save('latent.npy', latent)


def generate_image(latent, diffusion_type='ddpm', save_dir='./images'):
    """generate MNIST images from latent"""
    latent = Tensor(latent)
    model = AE(INPUT_DIM, OUTPUT_DIM, LATENT_DIM, HIDDEN_DIM)
    param_dict = load_checkpoint(CKPT_PATH)
    load_param_into_net(model, param_dict)
    recon_img = model.decoder(latent).numpy()
    recon_img = SampleScaler.unscale(recon_img)
    recon_img = recon_img.reshape(-1, OUTPUT_DIM)
    print(recon_img.shape)
    save_dir = os.path.join(save_dir, f'{diffusion_type}')
    os.makedirs(save_dir, exist_ok=True)
    for i in range(recon_img.shape[0]):
        plt.imsave(os.path.join(save_dir, f'{i}.png'), recon_img[i].reshape(
            28, 28), cmap="gray")


def export_latent():
    """export MNIST images latent"""
    images, labels = load_mnist()
    print(images.shape, labels.shape)
    images /= 255.
    images = images.reshape(images.shape[0], -1)
    model = AE(INPUT_DIM, OUTPUT_DIM, LATENT_DIM, HIDDEN_DIM)
    param_dict = load_checkpoint(CKPT_PATH)
    load_param_into_net(model, param_dict)
    latent = model.encoder(Tensor(images)).numpy()
    np.save('latent.npy', latent)
    labels = labels.reshape(labels.shape[0], 1)
    np.save('labels.npy', labels)


if __name__ == '__main__':
    export_latent()

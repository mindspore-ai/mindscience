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
"""Visualization"""
import numpy as np
import mindspore as ms
from mindspore import Tensor
import matplotlib.pyplot as plt


class Visualization:
    """Visualization"""
    def __init__(self, xx, yy, sample_num_train, sample_num_test, plot_n, latent_z_n_channel, merge_n_imgs, loader,
                 n_imgs, u_r10_test, u_r5_test, u_r3_test, u_r1_test, cp_fluc_test, re_c_test, scaling_input_test):
        self.xx = xx
        self.yy = yy
        self.sample_num_train = sample_num_train
        self.sample_num_test = sample_num_test
        self.plot_n = plot_n
        self.latent_z_n_channel = latent_z_n_channel
        self.merge_n_imgs = merge_n_imgs
        self.loader = loader
        self.n_imgs = n_imgs
        self.u_r10_test = u_r10_test
        self.u_r5_test = u_r5_test
        self.u_r3_test = u_r3_test
        self.u_r1_test = u_r1_test
        self.cp_fluc_test = cp_fluc_test
        self.re_c_test = re_c_test
        self.scaling_input_test = scaling_input_test

    def sample_images(self, epoch, gan_model):
        """Plot Images of train and test"""
        idx_train = np.random.randint(0, self.sample_num_train, self.plot_n)
        idx_test = np.random.randint(0, self.sample_num_test, self.plot_n)
        latent_z_input_train = Tensor(np.random.normal(0, 1, (
            self.plot_n, self.latent_z_n_channel, self.merge_n_imgs, self.merge_n_imgs)), dtype=ms.float32)
        latent_z_input_test = Tensor(np.random.normal(0, 1, (
            self.plot_n, self.latent_z_n_channel, self.merge_n_imgs, self.merge_n_imgs)), dtype=ms.float32)
        gen_u_i, gen_u_ii, gen_u_iii, gen_u_iv, _, _, _, _, _, _ = gan_model([
            Tensor(self.loader[idx_train][4]),
            Tensor(self.loader[idx_train][5]),
            Tensor(self.loader[idx_train][6]),
            latent_z_input_train])
        gen_u_i_, gen_u_ii_, gen_u_iii_, gen_u_iv_, _, _, _, _, _, _ = gan_model([
            Tensor(self.cp_fluc_test[idx_test, :, :, :]),
            Tensor(self.re_c_test[idx_test, :]),
            Tensor(self.scaling_input_test[idx_test, :]),
            latent_z_input_test])
        del _
        self.plot_images(epoch,
                         velocity_part=0,
                         gen_u_i=gen_u_i,
                         gen_u_ii=gen_u_ii,
                         gen_u_iii=gen_u_iii,
                         gen_u_iv=gen_u_iv,
                         sel_index=idx_train,
                         is_train=True)
        self.plot_images(epoch,
                         velocity_part=1,
                         gen_u_i=gen_u_i,
                         gen_u_ii=gen_u_ii,
                         gen_u_iii=gen_u_iii,
                         gen_u_iv=gen_u_iv,
                         sel_index=idx_train,
                         is_train=True)
        self.plot_images(epoch,
                         velocity_part=0,
                         gen_u_i=gen_u_i_,
                         gen_u_ii=gen_u_ii_,
                         gen_u_iii=gen_u_iii_,
                         gen_u_iv=gen_u_iv_,
                         sel_index=idx_test,
                         is_test=True)
        self.plot_images(epoch,
                         velocity_part=1,
                         gen_u_i=gen_u_i_,
                         gen_u_ii=gen_u_ii_,
                         gen_u_iii=gen_u_iii_,
                         gen_u_iv=gen_u_iv_,
                         sel_index=idx_test, is_test=True)

    def plot_images(self, epoch, velocity_part, gen_u_i, gen_u_ii, gen_u_iii, gen_u_iv, sel_index, is_train=False,
                    is_test=False):
        """Plot Images"""
        velocity_name = list(['u', 'v'])
        _, ax = plt.subplots(5, 8, figsize=(20, 16))
        if is_train:
            index = 0
            for i in range(0, 5, 1):
                ax[i, 0].contourf(self.xx, self.yy,
                                  gen_u_i[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 0].axis('off')
                ax[i, 1].contourf(self.xx, self.yy, self.loader[sel_index[index]][0][velocity_part, :, :],
                                  cmap='coolwarm')
                ax[i, 1].axis('off')
                ax[i, 2].contourf(self.xx, self.yy,
                                  gen_u_ii[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 2].axis('off')
                ax[i, 3].contourf(self.xx, self.yy, self.loader[sel_index[index]][1][velocity_part, :, :],
                                  cmap='coolwarm')
                ax[i, 3].axis('off')
                ax[i, 4].contourf(self.xx, self.yy,
                                  gen_u_iii[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 4].axis('off')
                ax[i, 5].contourf(self.xx, self.yy, self.loader[sel_index[index]][2][velocity_part, :, :],
                                  cmap='coolwarm')
                ax[i, 5].axis('off')
                ax[i, 6].contourf(self.xx, self.yy,
                                  gen_u_iv[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 6].axis('off')
                ax[i, 7].contourf(self.xx, self.yy, self.loader[sel_index[index]][3][velocity_part, :, :],
                                  cmap='coolwarm')
                ax[i, 7].axis('off')

                index = index + 1
            plt.savefig('training_results/images/imgs_train_' + velocity_name[velocity_part] + '%d.png' % (
                epoch))
            plt.close()
        if is_test:
            index = 0
            for i in range(0, 5, 1):
                ax[i, 0].contourf(self.xx, self.yy,
                                  gen_u_i[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 0].axis('off')
                ax[i, 1].contourf(self.xx, self.yy,
                                  self.u_r10_test[sel_index[index], velocity_part, :, :].reshape(self.n_imgs,
                                                                                                 self.n_imgs),
                                  cmap='coolwarm')
                ax[i, 1].axis('off')
                ax[i, 2].contourf(self.xx, self.yy,
                                  gen_u_ii[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 2].axis('off')
                ax[i, 3].contourf(self.xx, self.yy,
                                  self.u_r5_test[sel_index[index], velocity_part, :, :].reshape(self.n_imgs,
                                                                                                self.n_imgs),
                                  cmap='coolwarm')
                ax[i, 3].axis('off')
                ax[i, 4].contourf(self.xx, self.yy,
                                  gen_u_iii[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 4].axis('off')
                ax[i, 5].contourf(self.xx, self.yy,
                                  self.u_r3_test[sel_index[index], velocity_part, :, :].reshape(self.n_imgs,
                                                                                                self.n_imgs),
                                  cmap='coolwarm')
                ax[i, 5].axis('off')
                ax[i, 6].contourf(self.xx, self.yy,
                                  gen_u_iv[index, velocity_part, :, :].reshape(self.n_imgs, self.n_imgs).asnumpy(),
                                  cmap='coolwarm')
                ax[i, 6].axis('off')
                ax[i, 7].contourf(self.xx, self.yy,
                                  self.u_r1_test[sel_index[index], velocity_part, :, :].reshape(self.n_imgs,
                                                                                                self.n_imgs),
                                  cmap='coolwarm')
                ax[i, 7].axis('off')
                index = index + 1
            plt.savefig('training_results/images/imgs_test_' + velocity_name[velocity_part] + '%d.png' % (
                epoch))
            plt.close()
        plt.cla()
        plt.close('all')

    def __call__(self, epoch, gan_model):
        self.sample_images(epoch, gan_model)

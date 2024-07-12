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
"""train"""
import gc
import argparse
import time

import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
from mindspore import ops, Tensor, context
from mindflow.utils import load_yaml_config
import numpy as np
import matplotlib
from tqdm import tqdm

from src import (init_sub_model, DefineCompoundCritic, DefineCompoundGan, WassersteinLoss, GradLoss, AccesstrainDataset,
                 validation_test_dataset, Visualization)

matplotlib.use('agg')  # to prevent "Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize"
ms.dataset.config.set_prefetch_size(1)


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description="Cascade Net")
    parser.add_argument("--mode", type=str, default="PYNATIVE", choices=["PYNATIVE"],
                        help="This case only supports PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./config/Cascade-Net.yaml")
    result_args = parser.parse_args()
    return result_args


def train(conf):
    """Train"""
    latent_z_n_channel = conf["data"]["latent_z_n_channel"]
    batch_size = conf["data"]["batch_size"]
    merge_n_imgs = conf["summary"]["merge_n_imgs"]
    losslog = []

    loader = AccesstrainDataset(conf["data"]["root_dir"])
    train_dataset = GeneratorDataset(source=loader, column_names=conf["data"]["column_names"])
    train_dataset = train_dataset.shuffle(buffer_size=25)
    (u_r10_validation, u_r5_validation, u_r3_validation, u_r1_validation, cp_fluc_validation, re_c_validation,
     scaling_input_validation, u_r10_test, u_r5_test, u_r3_test, u_r1_test,
     cp_fluc_test, re_c_test, scaling_input_test) \
        = validation_test_dataset(conf["data"]["root_dir"])
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    sample_num_train = loader.__len__()
    sample_num_val = u_r1_validation.shape[0]
    sample_num_test = u_r1_test.shape[0]

    (merge_model, g_model_i, d_model_i, g_model_ii, d_model_ii, g_model_iii, d_model_iii, g_model_iv, d_model_iv) \
        = init_sub_model(conf["data"]["n_channel_p"], conf["data"]["n_channel_u"])
    critic_model = DefineCompoundCritic(conf["data"]["n_channel_p"], conf["data"]["n_channel_u"], batch_size,
                                        d_model_i, d_model_ii, d_model_iii, d_model_iv)
    critic_model.update_parameters_name('critic')
    gan_model = DefineCompoundGan(conf["data"]["n_channel_p"], conf["data"]["n_channel_u"], merge_model,
                                  g_model_i, g_model_ii, g_model_iii, g_model_iv)
    gan_model.update_parameters_name('generator')
    wasserstein_loss = WassersteinLoss()
    mae_loss = nn.MAELoss(reduction='none')
    grad_loss = GradLoss(conf["summary"]["dxy"])

    def d_forward_fn(x):
        g_pred = gan_model([x[4], x[5], x[6], x[7]])
        d_pred = critic_model([x[0], x[1], x[2], x[3], g_pred[4], g_pred[0], g_pred[1], g_pred[2], g_pred[3]])
        loss_wass = wasserstein_loss(ops.stack(d_pred[:8]), ops.Concat(axis=0)(
            [-ops.ones_like(ops.stack(d_pred[:4])), ops.ones_like(ops.stack(d_pred[4:8]))]))
        loss_gradp = ops.stack([d_pred[12](), d_pred[13](), d_pred[14](), d_pred[15]()])
        loss = ops.sum(1 * loss_wass) + ops.sum(conf["critic"]["lambda_GP"] * loss_gradp)
        loss_list = ms.numpy.concatenate((ms.numpy.expand_dims(loss, 0), loss_wass, loss_gradp), axis=0).asnumpy()
        return loss, loss_list

    def g_forward_fn(true, x):
        g_pred = gan_model([x[4], x[5], x[6], x[7]])
        d_pred = critic_model([x[0], x[1], x[2], x[3], g_pred[4], g_pred[0], g_pred[1], g_pred[2], g_pred[3]])
        loss_wass = wasserstein_loss(ops.stack(d_pred[4:8]), -ops.ones_like(ops.stack(d_pred[4:8])))
        loss_mae = mae_loss(ops.stack(g_pred[:4]), ops.stack(true[:4]))
        loss_mae = ops.mean(loss_mae, axis=(1, 2, 3, 4))
        loss_grad = grad_loss(ops.stack(g_pred[5:9]), ops.stack(true[:4]))
        loss = (ops.sum(1 * loss_wass) + ops.sum(conf["generator"]["lambda_L2_u"] * loss_mae) +
                ops.sum(conf["generator"]["lambda_L2_gradu"] * loss_grad))
        loss_list = ms.numpy.concatenate((ms.numpy.expand_dims(loss, 0), loss_wass, loss_mae, loss_grad),
                                         axis=0).asnumpy()
        return loss, loss_list

    d_optimizer = nn.RMSProp(critic_model.trainable_params(), learning_rate=conf["critic"]["critic_model_lr"],
                             epsilon=1e-07)
    d_grad_fn = ms.value_and_grad(d_forward_fn, None, d_optimizer.parameters, has_aux=True)
    g_optimizer = nn.RMSProp(gan_model.trainable_params(), learning_rate=conf["generator"]["gan_model_lr"],
                             epsilon=1e-07)
    g_grad_fn = ms.value_and_grad(g_forward_fn, None, g_optimizer.parameters, has_aux=True)
    d_optimizer.update_parameters_name('optim_d')
    g_optimizer.update_parameters_name('optim_g')

    def train_step(g_real_data, input_true):
        for _ in range(conf["critic"]["n_critic"]):
            (_, d_loss_list), d_grads = d_grad_fn(input_true)
            d_optimizer(d_grads)
            ave_g_loss_train_ncritic.append(d_loss_list)
        ave_d_loss_train.append(np.mean(ave_g_loss_train_ncritic, axis=0))
        (_, g_loss_list), g_grads = g_grad_fn(g_real_data, input_true)
        g_optimizer(g_grads)
        ave_g_loss_train.append(g_loss_list)

    def define_evaluation(u_r10, u_r5, u_r3, u_r1, cp, re, scaling_input, latent_z, idx):
        _, batch_d_loss = d_forward_fn(
            [Tensor(u_r10[idx, :, :, :]), Tensor(u_r5[idx, :, :, :]), Tensor(u_r3[idx, :, :, :]),
             Tensor(u_r1[idx, :, :, :]),
             Tensor(cp[idx, :, :, :]), Tensor(re[idx, :]), Tensor(scaling_input[idx, :]), latent_z])
        _, batch_g_loss = g_forward_fn(
            [Tensor(u_r10[idx, :, :, :]), Tensor(u_r5[idx, :, :, :]), Tensor(u_r3[idx, :, :, :]),
             Tensor(u_r1[idx, :, :, :])],
            [Tensor(u_r10[idx, :, :, :]), Tensor(u_r5[idx, :, :, :]), Tensor(u_r3[idx, :, :, :]),
             Tensor(u_r1[idx, :, :, :]),
             Tensor(cp[idx, :, :, :]), Tensor(re[idx, :]), Tensor(scaling_input[idx, :]), latent_z])
        return batch_d_loss, batch_g_loss

    critic_model.set_train()
    gan_model.set_train()

    xx, yy = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
    sample_images = Visualization(xx, yy, sample_num_train, sample_num_test, conf["summary"]["plot_n"],
                                  latent_z_n_channel, merge_n_imgs, loader, conf["summary"]["n_imgs"],
                                  u_r10_test, u_r5_test, u_r3_test, u_r1_test, cp_fluc_test, re_c_test,
                                  scaling_input_test)

    for epoch in range(conf["summary"]["epochs"]):
        start = time.perf_counter()
        ave_d_loss_train = list()
        ave_g_loss_train = list()
        for _, (u_r10_train, u_r5_train, u_r3_train, u_r1_train, cp_fluc_train, re_c_train, scaling_input_train) in (
                enumerate(tqdm(train_dataset))):
            ave_g_loss_train_ncritic = list()
            latent_z_input_train = ops.normal((batch_size, latent_z_n_channel, merge_n_imgs, merge_n_imgs), 0, 1)
            input_data = [u_r10_train, u_r5_train, u_r3_train, u_r1_train,
                          cp_fluc_train, re_c_train, scaling_input_train, latent_z_input_train]
            train_step([u_r10_train, u_r5_train, u_r3_train, u_r1_train], input_data)
        ave_d_loss_train = np.mean(ave_d_loss_train, axis=0)
        ave_g_loss_train = np.mean(ave_g_loss_train, axis=0)

        idx_ = np.random.randint(0, sample_num_val, batch_size)
        latent_z_input_validation = Tensor(np.random.normal(0, 1, (
            batch_size, latent_z_n_channel, merge_n_imgs, merge_n_imgs)), dtype=ms.float32)
        batch_d_loss_val, batch_g_loss_val = define_evaluation(
            u_r10_validation, u_r5_validation, u_r3_validation, u_r1_validation,
            cp_fluc_validation, re_c_validation, scaling_input_validation, latent_z_input_validation, idx_)

        idx__ = np.random.randint(0, sample_num_test, batch_size)
        latent_z_input_test = Tensor(np.random.normal(0, 1, (
            batch_size, latent_z_n_channel, merge_n_imgs, merge_n_imgs)), dtype=ms.float32)
        batch_d_loss_test, batch_g_loss_test = define_evaluation(
            u_r10_test, u_r5_test, u_r3_test, u_r1_test,
            cp_fluc_test, re_c_test, scaling_input_test, latent_z_input_test, idx__)

        end = time.perf_counter()
        time_consumption = (end - start) // 60
        print("%d:[%d min] [D:%.2f, %.2f, %.2f, %.2f, %.2f] [G:%.2f, %.2f, %.2f, %.2f, %.2f] "
              "[L2_train:%.5f, %.5f, %.5f, %.5f] [L2_val:%.5f, %.5f, %.5f, %.5f] [L2_test:%.5f, %.5f, %.5f, %.5f]" %
              (epoch, time_consumption, ave_d_loss_train[0],
               ave_d_loss_train[1] + ave_d_loss_train[5] + ave_d_loss_train[9],
               ave_d_loss_train[2] + ave_d_loss_train[6] + ave_d_loss_train[10],
               ave_d_loss_train[3] + ave_d_loss_train[7] + ave_d_loss_train[11],
               ave_d_loss_train[4] + ave_d_loss_train[8] + ave_d_loss_train[12],
               ave_g_loss_train[0],
               ave_g_loss_train[1], ave_g_loss_train[2], ave_g_loss_train[3], ave_g_loss_train[4],
               ave_g_loss_train[5], ave_g_loss_train[6], ave_g_loss_train[7], ave_g_loss_train[8],
               batch_g_loss_val[5], batch_g_loss_val[6], batch_g_loss_val[7], batch_g_loss_val[8],
               batch_g_loss_test[5], batch_g_loss_test[6], batch_g_loss_test[7], batch_g_loss_test[8]))

        losslog.append(np.concatenate(  # 60
            ([ave_d_loss_train[i] for i in range(13)],  # 13
             [batch_d_loss_val[i] for i in range(13)],  # 13
             [batch_d_loss_test[i] for i in range(13)],  # 13
             [ave_g_loss_train[i] for i in range(13)],  # 13
             [batch_g_loss_val[i] for i in range(13)],  # 13
             [batch_g_loss_test[i] for i in range(13)])))  # 13

        if epoch % conf["summary"]["sample_interval"] == 0:
            sample_images(epoch, gan_model)
            ms.save_checkpoint(gan_model, "training_results/model/gan_%d" % epoch + ".ckpt")
            ms.save_checkpoint(critic_model, "training_results/model/critic_%d" % epoch + ".ckpt")
            np.savetxt('training_results/loss/loss.txt', losslog, fmt='%.4f')
        del ave_d_loss_train, ave_g_loss_train, batch_d_loss_val, batch_g_loss_val, batch_d_loss_test, batch_g_loss_test
        gc.collect()


if __name__ == '__main__':
    input_args = parse_args()
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=input_args.device_target,
                        device_id=input_args.device_id)
    config = load_yaml_config(input_args.config_file_path)

    train(config)

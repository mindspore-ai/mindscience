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
IDM tutorial
"""

import argparse
import os

import numpy as np
import mindspore.dataset as ds
import mindspore.numpy as mnp
from mindspore import Tensor, context, nn, save_checkpoint, set_seed

from model import (TrainOneStepCell, WithLossCell, beta_schedule, cos_decay_lr,
                   temporal_proximal_sampling)

context.set_context(mode=context.GRAPH_MODE)  # Train
set_seed(666)
np.random.seed(666)

parser = argparse.ArgumentParser(description='IDM')
parser.add_argument('--data_path', required=True, type=str,
                    default=None, help='Data Location.')
parser.add_argument('--resolution', required=True, type=int,
                    default=400, help='Time Resolution (ps).')
parser.add_argument('--max_clusters', required=False, type=int,
                    default=8, help='Maximum Number of Clustering')
parser.add_argument('--latent_dim', required=False, type=int,
                    default=4, help='Dimension of Latent Representation')
args_opt = parser.parse_args()

# 修改数据存储的地址：
dpath = args_opt.data_path  # e.g., dpath = "./data/"
# IDM的时间分辨率：
resolution = args_opt.resolution  # unit: ps; e.g., [4, 50, 200, 400]
# IDM的聚类数目上限：
num_class = args_opt.max_clusters
# IDM的特征空间维度：
latent_dim = args_opt.latent_dim

# pylint: disable=invalid-name
# 设置网络模型的超参数
batchsize = 512
train_steps = 5000
num_samples = max(1, min(resolution, 16))

# 神经网络的大小:
input_dim = 45
hidden_dim = 128

# 根据超参数 实例化WithLossCell
idm_model = WithLossCell(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                         num_class=num_class, temperature=1e-1, reg_recon=1.0)

all_parameters = idm_model.trainable_params()
lr = cos_decay_lr(start_step=0, lr_min=5e-5, lr_max=2e-4,
                  decay_steps=train_steps, warmup_steps=2000)
opt = nn.Adam(params=all_parameters, learning_rate=Tensor(lr), eps=1e-6)

train_net = TrainOneStepCell(
    idm_model, opt, sens=1.0, enable_clip_grad=True, clip_value=5.0)

train_net.set_train(True)  # Setup BatchNorm

# 创建迭代器
_idx_data = np.loadtxt(os.path.join(
    dpath, "selected_data_ids.txt")).astype(np.int32)
# We select 100k FPS samples to augment training set
idx_data = _idx_data[:100000]

local_filename = os.path.join(
    dpath, 'alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
with np.load(local_filename) as fh:
    feat_array = [fh[key] for key in fh.keys()]

feat_array = np.concatenate(feat_array, 0)
feat_array = np.log(feat_array+1e-5)  # transform to log-scale

feat_mean = np.mean(feat_array, 0, keepdims=True)
feat_std = np.std(feat_array, 0, keepdims=True)
feat_array = (feat_array-feat_mean)/feat_std

datasize = feat_array.shape[0]
print("Datasize: ", datasize)

size0 = int(batchsize*0.1)
dataset = ds.NumpySlicesDataset(data=(idx_data,), column_names=["idx",], shuffle=True)
dataset = dataset.batch(size0, drop_remainder=True)
dataset = dataset.repeat(100)

_tau1_full = np.arange(resolution, datasize-resolution)
tau1_full = []
for i in range(5):
    np.random.shuffle(_tau1_full)
    tau1_full.append(_tau1_full)
tau1_full = np.concatenate(tau1_full, 0)

step = 0
losses = []
for _d in dataset:
    step += 1

    # 1. Prepare Data:
    # (B,):
    tau0 = _d[0].asnumpy().astype(np.int32)

    size1 = batchsize-size0
    if size1 > 0:
        _tau1 = tau1_full[(step-1)*size1: step*size1]
        tau0 = np.concatenate((tau0, _tau1), 0)

    tau_list = []
    _t = temporal_proximal_sampling(
        tau0, resolution, datasize, num_samples=num_samples)
    tau_list.append(_t)
    # (B,t):
    tau_list = np.concatenate(tau_list, axis=1)

    # (B,dim):
    feat_0 = feat_array[tau0]
    feat_0 = Tensor(feat_0, mnp.float32)

    # (B,t,dim):
    feat_t = feat_array[tau_list]
    feat_t = Tensor(feat_t, mnp.float32)

    beta, reg_ent = beta_schedule(step)

    beta = Tensor(beta, mnp.float32)
    reg_ent = Tensor(reg_ent, mnp.float32)

    loss, loss_mi, loss_recon, loss_entropy, mut_info = train_net(
        feat_0, feat_t, beta, reg_ent)

    if step % 100 == 0:
        # 输出训练记录
        print('[%d]\tMutual_Info: %.4f\tUncertainty: %.4f' %
              (step, mut_info.asnumpy(), loss_entropy.asnumpy()))
        losses.append((mut_info.asnumpy(), loss_mi.asnumpy(),
                       loss_recon.asnumpy(), loss_entropy.asnumpy()))

    if step % 1000 == 0:
        ckpt_name = "../ckpts/" + f"model_epoch_{step}.ckpt"
        save_checkpoint(idm_model, ckpt_name)

    if step == train_steps:
        break

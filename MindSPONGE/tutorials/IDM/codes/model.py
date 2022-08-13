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
from mindspore import Tensor, nn, ops


# pylint: disable=invalid-name
class IDM_Net(nn.Cell):
    """IDM Net"""

    def __init__(self, input_dim=45, hidden_dim=128, latent_dim=32, num_class=32, temperature=1e-1):  # 2/20/200 ps
        super(IDM_Net, self).__init__()
        self.act_fn = nn.Tanh()  # nn.LeakyReLU()

        self.encoder = nn.SequentialCell()
        self.encoder.append(nn.Dense(input_dim, hidden_dim))
        self.encoder.append(self.act_fn)
        self.encoder.append(nn.BatchNorm1d(hidden_dim))
        self.encoder.append(nn.Dense(hidden_dim, hidden_dim))
        self.encoder.append(self.act_fn)
        self.encoder.append(nn.BatchNorm1d(hidden_dim))
        self.encoder.append(nn.Dense(hidden_dim, latent_dim))
        self.c = Tensor(np.arange(num_class), mnp.int32)
        self.center_embedding = nn.Embedding(
            num_class, hidden_dim, embedding_table='Normal')
        self.center_net = nn.SequentialCell()
        self.center_net.append(nn.Dense(hidden_dim, hidden_dim))
        self.center_net.append(self.act_fn)
        self.center_net.append(nn.BatchNorm1d(hidden_dim))
        self.center_net.append(nn.Dense(hidden_dim, latent_dim))

        self.temperature = temperature
        self.softmax = nn.Softmax(-1)
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.bmatmul_t = ops.BatchMatMul(transpose_b=True)

    def construct(self, x):
        """construct"""

        # x: (B,t/1,dim):
        input_shape = x.shape

        # (B*t/1,dim):
        x_ = mnp.reshape(x, (-1, input_shape[-1]))
        # (B*t/1,c):
        h_ = self.encoder(x_)
        # (B,t/1,cz):
        h = mnp.reshape(h_, input_shape[:-1]+(-1,))

        # (K):
        center = mnp.reshape(self.c, (-1))
        # (K,c):
        center_embed = self.center_embedding(center)
        # (K,cz):
        c_ = self.center_net(center_embed)
        # (B,K,cz):
        c = mnp.tile(mnp.expand_dims(c_, 0), (input_shape[0], 1, 1))

        # (B,t/1,cz)@(B,K,cz).T -> (B,t/1,K)
        att_logits = self.bmatmul_t(h, c)
        att_logits = att_logits/self.temperature / \
            mnp.sqrt(ops.cast(self.latent_dim, mnp.float32))

        # (B,t/1,K):
        att_probs = self.softmax(att_logits)

        return h, att_probs, c_


class WithLossCell(nn.Cell):
    """损失函数与训练方法"""

    def __init__(self, input_dim=45, hidden_dim=128, latent_dim=32, num_class=32, temperature=1e-1, reg_recon=1e-2):
        super(WithLossCell, self).__init__(auto_prefix=True)
        self.act_fn = nn.Tanh()  # nn.LeakyReLU()

        self.decoder = nn.SequentialCell()
        self.decoder.append(nn.Dense(latent_dim, hidden_dim))
        self.decoder.append(self.act_fn)
        self.decoder.append(nn.BatchNorm1d(hidden_dim))
        self.decoder.append(nn.Dense(hidden_dim, hidden_dim))
        self.decoder.append(self.act_fn)
        self.decoder.append(nn.BatchNorm1d(hidden_dim))
        self.decoder.append(nn.Dense(hidden_dim, input_dim))

        self.cluster_net = IDM_Net(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                                   num_class=num_class, temperature=temperature)

        self.bmatmul_t = ops.BatchMatMul(transpose_b=True)
        self.matmul_t = ops.MatMul(transpose_b=True)

        self.eps = 1e-9
        self.reg_recon = reg_recon

    def inference(self, x):
        # x：(B,dim)

        # (B,1,dim):
        x_ = mnp.expand_dims(x, 1)
        # (B,1,c), (B,1,K=Class):
        _, cluster, _ = self.cluster_net(x_)

        return mnp.squeeze(cluster, 1)  # (B,K)

    def construct(self, x0, xt, beta, reg_entropy):
        """
        x0: (B,dim)
        xt: (B,t,dim)
        beta: ()
        wt: (t,)
        reg_entropy: ()
        """
        # (B,1,dim):
        x0_ = mnp.expand_dims(x0, 1)
        # (B,t,dim):
        xt_ = xt

        # (B,1,c), (B,1,K=Class):
        h_0, cluster_0, _ = self.cluster_net(x0_)
        # (B,t,c), (B,t,K=Class):
        _, cluster_t, _ = self.cluster_net(xt_)

        # Reconstruction:
        # (B,c):
        h_0 = mnp.squeeze(h_0, axis=1)
        # (B,dim):
        x_recon = self.decoder(h_0)
        # (B,):
        loss_recon_ = mnp.sqrt(mnp.sum((x_recon-x0)**2, -1) + 1e-5)
        # ():
        loss_recon = mnp.mean(loss_recon_, 0)

        # Compute Mutual Information:
        # (B,1,K) -> (K,1):
        p_margin_0 = mnp.expand_dims(mnp.mean(cluster_0, axis=(0, 1)), axis=-1)
        # (B,t,K) -> (K,1):
        p_margin_t = mnp.expand_dims(mnp.mean(cluster_t, axis=(0, 1)), axis=-1)

        # (K,1)@(K,1).T -> (K,K):
        p_margin_mat = self.matmul_t(p_margin_0, p_margin_t)
        p_margin_mat_clip = mnp.clip(p_margin_mat, self.eps, 1.)

        # -> (B,t,K):
        cluster_0_ = mnp.tile(cluster_0, (1, xt.shape[1], 1))
        cluster_t_ = cluster_t
        # (B,t,K,K):
        p_joint_mat_ = self.bmatmul_t(mnp.expand_dims(
            cluster_0_, -1), mnp.expand_dims(cluster_t_, -1))
        # (K,K):
        p_joint_mat = mnp.mean(p_joint_mat_, (0, 1))
        p_joint_mat_clip = mnp.clip(p_joint_mat, self.eps, 1.)

        # Mutual information:
        # (K,K):
        part1 = p_joint_mat * mnp.log(p_joint_mat_clip)
        # (K,K):
        part2 = beta * p_joint_mat * mnp.log(p_margin_mat_clip)
        # (K,K):
        mutual_info = part1 - part2
        # ():
        mutual_info = mnp.sum(mutual_info, axis=(0, 1))  # to be maximized
        # ():
        loss_mi = - mutual_info

        # Mutual information For Monitor:
        # (K,K):
        part2_ = p_joint_mat * mnp.log(p_margin_mat_clip)
        # (K,K):
        mutual_info_ = part1 - part2_
        # ():
        mutual_info_ = mnp.sum(mutual_info_, axis=(0, 1))  # to be maximized

        # Compute Entropy
        # (B,1,K):
        entropy_ = - cluster_0 * mnp.log(mnp.clip(cluster_0, self.eps, 1.))
        # ():
        entropy = mnp.mean(mnp.sum(entropy_, -1), 0)  # to be minimized
        # ():
        loss_entropy = entropy

        # ():
        loss = loss_mi + self.reg_recon*loss_recon + reg_entropy*loss_entropy
        return loss, loss_mi, loss_recon, loss_entropy, mutual_info_


grad_scale = ops.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


class TrainOneStepCell(nn.Cell):
    """train one step cell"""

    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True, clip_value=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.enable_clip_grad = enable_clip_grad
        self.hyper_map = ops.HyperMap()
        self.clip_value = clip_value

    def construct(self, *inputs):
        """construct"""

        losses = self.network(*inputs)
        loss, loss_mi, loss_recon, loss_entropy, mut_info = losses

        sens = ops.fill(loss.dtype, loss.shape, self.sens)
        sens1 = ops.fill(loss_mi.dtype, loss_mi.shape, 0.)
        sens2 = ops.fill(loss_recon.dtype, loss_recon.shape, 0.)
        sens3 = ops.fill(loss_entropy.dtype, loss_entropy.shape, 0.)
        sens4 = ops.fill(mut_info.dtype, mut_info.shape, 0.)

        grads = self.grad(self.network, self.weights)(
            *inputs, (sens, sens1, sens2, sens3, sens4))

        if self.sens > 1+1e-5:
            grads = self.hyper_map(ops.partial(
                grad_scale, ops.scalar_to_array(self.sens)), grads)

        if self.enable_clip_grad:
            grads = ops.clip_by_global_norm(grads, self.clip_value)

        loss = ops.depend(loss, self.optimizer(grads))

        return loss, loss_mi, loss_recon, loss_entropy, mut_info


def temporal_proximal_sampling(t0, tau, datasize, num_samples=5):
    """
    t0:(B,)
    tau:()
    datasize:()
    """
    bs = t0.shape[0]
    # (B,1):
    tau_min = np.clip(t0-tau, 0, datasize -
                      1).reshape((-1, 1)).astype(np.float32)
    tau_max = np.clip(t0+tau, 0, datasize -
                      1).reshape((-1, 1)).astype(np.float32)

    # (B,num_samples):
    z = np.random.uniform(0., 1., size=(bs, num_samples))
    t = (z*(tau_max-tau_min) + tau_min).astype(np.int32)
    t_pad = np.where(z < 0.5, tau_min, tau_max).astype(np.int32)
    t = np.where(t == t0.reshape((-1, 1)), t_pad, t)
    return t  # (B,num_samples)


def cos_decay_lr(start_step, lr_min, lr_max, decay_steps, warmup_steps, max_steps=1000000):
    """cos decay learning rate"""

    lr_each_step = []
    for i in range(decay_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_min)) / float(warmup_steps)
            lr = float(lr_min) + lr_inc * (i + 1)
        else:
            lr = lr_min + 0.5 * \
                (lr_max-lr_min) * \
                (1 + np.cos((i - warmup_steps) / decay_steps * np.pi))
        lr = max(lr_min, lr)
        lr_each_step.append(lr)
    lr_each_step += (max_steps-decay_steps)*[lr_each_step[-1]]
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step[start_step:]


def beta_schedule(step):
    """beta schedule"""

    beta_bin_list = np.array([500, 1000, 3000], np.int32)
    beta_levels = [1.2, 1.1, 1.0, 0.9]

    beta_id = np.sum(step > beta_bin_list)
    beta = beta_levels[beta_id]

    ent_bin_list = np.array([1000, 3000], np.int32)
    ent_levels = [0., 1e-3, 1e-2]

    ent_id = np.sum(step > ent_bin_list)
    reg_ent = ent_levels[ent_id]

    return beta, reg_ent

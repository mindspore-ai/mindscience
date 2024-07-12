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
"""Compound Main Model"""
from functools import partial

import mindspore.nn as nn
from mindspore import ops

from .ResUnetWithAttentionSpatialChannelGenerator import AttentionResUnet
from .MultiscaleDiscriminator import MultiscaleDiscriminator
from .Loss import GradientPenaltyLoss


class DefineMerge(nn.Cell):
    """Merge Model"""
    def __init__(self, merge_n_imgs, merge_n_channel):
        super(DefineMerge, self).__init__()
        self.merge_n_imgs = merge_n_imgs
        self.merge_n_channel = merge_n_channel
        self.define_merge_sc = nn.SequentialCell(
            nn.Dense(21, self.merge_n_imgs * self.merge_n_imgs * self.merge_n_channel),
            nn.BatchNorm1d(self.merge_n_imgs * self.merge_n_imgs * self.merge_n_channel),
            nn.ReLU()
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, in_u_scaling, in_re):  # dim of in_merge_map is (None, 8, 8, 1)
        in_merge = self.concat((in_u_scaling, in_re))
        in_merge_map = self.define_merge_sc(in_merge)
        in_merge_map = ops.reshape(in_merge_map,
                                   (in_merge_map.shape[0], self.merge_n_channel, self.merge_n_imgs,
                                    self.merge_n_imgs))
        return in_merge_map


def sub_model_block_relu(n_channel_u, in_dim, filter_num_g, filter_num_d, filter_size, g_merge, g_latent_z, d_multi,
                         d_single):
    """Sub Model"""
    sub_g_model = AttentionResUnet(
        input_channel=in_dim,
        output_mask_channel=n_channel_u,
        filter_num=filter_num_g,
        filter_size=filter_size,
        merge=g_merge, latent_z=g_latent_z)
    sub_d_model = MultiscaleDiscriminator(
        input_channel=n_channel_u,
        filter_num=filter_num_d,
        multi_scale=d_multi,
        single_scale=d_single)
    return sub_g_model, sub_d_model


def init_sub_model(n_channel_p, n_channel_u):
    """Sub Model Init"""
    merge_model = DefineMerge(8, 1)
    g_model_i, d_model_i = sub_model_block_relu(
        n_channel_u=n_channel_u,
        in_dim=n_channel_p,
        filter_num_g=8,
        filter_num_d=16,
        filter_size=3,
        g_merge=True,
        g_latent_z=False,
        d_multi=False,
        d_single=True
    )
    g_model_ii, d_model_ii = sub_model_block_relu(
        n_channel_u=n_channel_u,
        in_dim=(n_channel_p + n_channel_u),
        filter_num_g=8,
        filter_num_d=8,
        filter_size=3,
        g_merge=True,
        g_latent_z=False,
        d_multi=True,
        d_single=False
    )
    g_model_iii, d_model_iii = sub_model_block_relu(
        n_channel_u=n_channel_u,
        in_dim=(n_channel_p + n_channel_u + n_channel_u),
        filter_num_g=8,
        filter_num_d=8,
        filter_size=3,
        g_merge=True,
        g_latent_z=False,
        d_multi=True,
        d_single=False
    )
    g_model_iv, d_model_iv = sub_model_block_relu(
        n_channel_u=n_channel_u,
        in_dim=(n_channel_p + n_channel_u + n_channel_u + n_channel_u),
        filter_num_g=8,
        filter_num_d=8,
        filter_size=2,
        g_merge=False,
        g_latent_z=True,
        d_multi=True,
        d_single=False
    )
    return merge_model, g_model_i, d_model_i, g_model_ii, d_model_ii, g_model_iii, d_model_iii, g_model_iv, d_model_iv


class DefineCompoundCritic(nn.Cell):
    """Compound Critic"""
    def __init__(self, n_channel_p, n_channel_u, batch_size, d_model_i, d_model_ii, d_model_iii, d_model_iv):
        super(DefineCompoundCritic, self).__init__()
        self.n_channel_p = n_channel_p
        self.n_channel_u = n_channel_u
        self.batch_size = batch_size

        self.d_model_i = d_model_i
        self.d_model_ii = d_model_ii
        self.d_model_iii = d_model_iii
        self.d_model_iv = d_model_iv

        self.gradient_penalty_loss = GradientPenaltyLoss()
        self.concat = ops.Concat(axis=1)

    def construct(self, input_data):
        """Compound Critic Model"""
        in_u_i = input_data[0]
        in_u_ii = input_data[1]
        in_u_iii = input_data[2]
        in_u_iv = input_data[3]
        in_merge = input_data[4]
        gen_u_i = input_data[5]
        gen_u_ii = input_data[6]
        gen_u_iii = input_data[7]
        gen_u_iv = input_data[8]

        alpha = ops.rand((self.batch_size, 2, 1, 1))

        d_on_real_i = self.d_model_i(in_u_i, in_merge)
        d_on_fake_i = self.d_model_i(gen_u_i, in_merge)
        diff_fv = in_u_i - gen_u_i
        interpolated_img_i = gen_u_i + alpha * diff_fv
        d_on_interp_i = self.d_model_i(interpolated_img_i, in_merge)
        partial_gp_loss_i = partial(self.gradient_penalty_loss, d_model=self.d_model_i,
                                    interpolated_img=interpolated_img_i, in_merge=in_merge)

        d_on_real_ii = self.d_model_ii(in_u_ii, in_merge)
        d_on_fake_ii = self.d_model_ii(gen_u_ii, in_merge)
        diff_fv = in_u_ii - gen_u_ii
        interpolated_img_ii = gen_u_ii + alpha * diff_fv
        d_on_interp_ii = self.d_model_ii(interpolated_img_ii, in_merge)
        partial_gp_loss_ii = partial(self.gradient_penalty_loss, d_model=self.d_model_ii,
                                     interpolated_img=interpolated_img_ii, in_merge=in_merge)

        d_on_real_iii = self.d_model_iii(in_u_iii, in_merge)
        d_on_fake_iii = self.d_model_iii(gen_u_iii, in_merge)
        diff_fv = in_u_iii - gen_u_iii
        interpolated_img_iii = gen_u_iii + alpha * diff_fv
        d_on_interp_iii = self.d_model_iii(interpolated_img_iii, in_merge)
        partial_gp_loss_iii = partial(self.gradient_penalty_loss, d_model=self.d_model_iii,
                                      interpolated_img=interpolated_img_iii, in_merge=in_merge)

        d_on_real_iv = self.d_model_iv(in_u_iv, in_merge)
        d_on_fake_iv = self.d_model_iv(gen_u_iv, in_merge)
        diff_fv = in_u_iv - gen_u_iv
        interpolated_img_iv = gen_u_iv + alpha * diff_fv
        d_on_interp_iv = self.d_model_iv(interpolated_img_iv, in_merge)
        partial_gp_loss_iv = partial(self.gradient_penalty_loss, d_model=self.d_model_iv,
                                     interpolated_img=interpolated_img_iv, in_merge=in_merge)

        return (d_on_real_i, d_on_real_ii, d_on_real_iii, d_on_real_iv,
                d_on_fake_i, d_on_fake_ii, d_on_fake_iii, d_on_fake_iv,
                d_on_interp_i, d_on_interp_ii, d_on_interp_iii, d_on_interp_iv,
                partial_gp_loss_i, partial_gp_loss_ii, partial_gp_loss_iii, partial_gp_loss_iv)


class DefineCompoundGan(nn.Cell):
    """Compound Generator"""
    def __init__(self, n_channel_p, n_channel_v, merge_model, g_model_i, g_model_ii, g_model_iii, g_model_iv):
        super(DefineCompoundGan, self).__init__()
        self.n_channel_p = n_channel_p
        self.n_channel_u = n_channel_v

        self.merge_model = merge_model
        self.g_model_i = g_model_i
        self.g_model_ii = g_model_ii
        self.g_model_iii = g_model_iii
        self.g_model_iv = g_model_iv

        self.concat = ops.Concat(axis=1)

    def construct(self, input_data):
        """Compound Generator Model"""
        in_cp = input_data[0]  # CP [None, 128, 128, 3]
        in_re = input_data[1]  # Conditions Re, [None, 1]
        in_u_scaling = input_data[2]  # Inputs scaling u, [None, 20]
        in_z = input_data[3]  # latent_z inputs, [None, 8, 8, 50]
        in_merge = self.merge_model(in_re, in_u_scaling)  # merge inputs, [None, 8, 8, 1]

        # connect the input and output of the generator
        gen_u_i = self.g_model_i(in_cp, in_merge)
        gen_u_ii = self.g_model_ii(self.concat((in_cp, gen_u_i)), in_merge)
        gen_u_iii = self.g_model_iii(self.concat((in_cp, gen_u_i, gen_u_ii)), in_merge)
        gen_u_iv = self.g_model_iv(self.concat((in_cp, gen_u_i, gen_u_ii, gen_u_iii)), in_merge, in_z)

        return (gen_u_i, gen_u_ii, gen_u_iii, gen_u_iv, in_merge,
                gen_u_i, gen_u_ii, gen_u_iii, gen_u_iv, in_merge)

# Copyright 2022 Huawei Technologies Co., Ltd
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
"""evogen"""
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
import mindspore.nn.probability.distribution as msd

from module.evogen_block import EvoformerIteration, LatentBlock, EvoGenFeatProcess, LatentNormal
from model.fold import MegaFold
import numpy as np
from mindsponge.cell.initializer import lecun_init


class MsaGen(nn.Cell):
    '''MsaGen'''

    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.config = config.model.embeddings_and_evoformer
        self.config_latent = config.model.latent

        self.evoformer_num_block = self.config.evoformer_num_block
        self.msa_act_dim = self.config.msa_channel
        self.pair_act_dim = self.config.pair_channel
        self.num_noise = self.config_latent.num_noise
        self.noise_layers = self.config_latent.noise_layers
        self.latent_dims = self.config_latent.latent_dim_tuple
        self.del_num_bins = self.config.del_num_bins

        evoformer_encoder_blocks = nn.CellList()
        evoformer_decoder_blocks = nn.CellList()
        for i in range(self.evoformer_num_block):
            evoformer_block = EvoformerIteration(config,
                                                 msa_act_dim=self.msa_act_dim,
                                                 pair_act_dim=self.pair_act_dim,
                                                 encoding=True,
                                                 )
            evoformer_encoder_blocks.append(evoformer_block)

            evoformer_block = EvoformerIteration(config,
                                                 msa_act_dim=self.msa_act_dim,
                                                 pair_act_dim=self.pair_act_dim,
                                                 encoding=False,
                                                 )
            evoformer_decoder_blocks.append(evoformer_block)
        self.evoformer_encoder = evoformer_encoder_blocks
        self.evoformer_decoder = evoformer_decoder_blocks

        self.latent_normal = LatentNormal()
        latent_blocks = nn.CellList()
        for i in range(self.num_noise):
            lt_block = LatentBlock(config,
                                   msa_dim=self.msa_act_dim,
                                   latent_dim=self.latent_dims[i],
                                   )
            latent_blocks.append(lt_block)
        self.latent_block = latent_blocks

        self.num_aa_types = config.global_config.num_aa_types
        self.num_msa_types = self.num_aa_types + 1
        self.pair_bins = self.config.num_buckets * 2 + 1
        self.num_del_num_bins = len(self.del_num_bins)
        self.del_num_bins = Tensor(self.del_num_bins, mstype.float32)

        self.preprocess_1d = nn.Dense(self.num_aa_types, self.msa_act_dim, weight_init=lecun_init(self.num_aa_types))
        self.preprocess_msa = nn.Dense(self.num_msa_types + 2, self.msa_act_dim,
                                       weight_init=lecun_init(self.num_msa_types))
        self.left_single = nn.Dense(self.num_aa_types, self.pair_act_dim, weight_init=lecun_init(self.num_aa_types))
        self.right_single = nn.Dense(self.num_aa_types, self.pair_act_dim, weight_init=lecun_init(self.num_aa_types))
        self.pair_activations = nn.Dense(self.pair_bins, self.pair_act_dim, weight_init=lecun_init(self.pair_bins))

        np_mask = np.ones(shape=(self.num_msa_types), dtype=np.float32)
        np_mask[20], np_mask[22] = 0, 0
        self.reconstruct_mask = Tensor(np_mask, mstype.float32)

        self.reconstruct_head = nn.Dense(self.msa_act_dim, self.num_msa_types, weight_init='zeros', has_bias=True)

        self.reconstruct_head_query_new = nn.Dense(self.msa_act_dim, self.num_msa_types, weight_init='zeros',
                                                   has_bias=True)

        self.reconstruct_head_hasdel = nn.Dense(self.msa_act_dim, 1, weight_init='zeros', has_bias=True,
                                                bias_init='ones')
        self.reconstruct_head_delnum = nn.Dense(self.msa_act_dim, self.num_del_num_bins, weight_init='zeros',
                                                has_bias=True)

        self.matmul = P.MatMul(transpose_b=True)
        self.expand_dims = P.ExpandDims()

    def construct(self, q_raw_feat, msa_raw_feat, pair_raw_feat, msa_mask, pair_mask, context_mask, target_mask,
                  res_idx=None, random_feat=None):
        '''construct'''
        mask_tmp = P.Transpose()(msa_mask * context_mask, (1, 0))
        mask_norm = self.matmul(mask_tmp, mask_tmp)
        mask_norm = self.expand_dims(mask_norm, -1)

        msa_activations, pair_activations = self._init_feat(q_raw_feat, msa_raw_feat, pair_raw_feat)
        msa_act_list = [msa_activations]
        pair_act_list = [pair_activations]
        for i in range(self.evoformer_num_block):
            msa_activations, pair_activations = self.evoformer_encoder[i](msa_activations, pair_activations, \
                                                                          msa_mask, pair_mask, context_mask,
                                                                          mask_norm=mask_norm, res_idx=res_idx)
            msa_act_list.append(msa_activations)
            pair_act_list.append(pair_activations)

        msa_recon_act = P.Tile()(self.expand_dims(msa_activations[0], 0), (msa_activations.shape[0], 1, 1))

        kl_all = []
        i_layer = 0
        for i in range(self.num_noise):
            layers = self.noise_layers[i]
            for _ in range(layers):
                msa_recon_act, _ = self.evoformer_decoder[i_layer](msa_recon_act, pair_act_list[-(i_layer + 1)], \
                                                                   msa_mask, pair_mask, context_mask, res_idx=res_idx)
                i_layer += 1

            eps = None
            if random_feat is not None:
                eps = random_feat[i]
                eps = eps[:, :, :self.latent_dims[i]]

            latent_block_result = self.latent_block[i](msa_recon_act, msa_act_list[-(i_layer + 1)], msa_mask,
                                                       context_mask, target_mask, eps)
            msa_recon_act, mu_prior, log_sigma_prior, mu_posterior, log_sigma_posterior = latent_block_result

            mu_posterior[0] = mu_prior[0] * 1.
            log_sigma_posterior[0] = log_sigma_prior[0] * 1.

            kl_per_var = self.latent_normal(mu_posterior, log_sigma_posterior, mu_prior, log_sigma_prior)
            kl_all.append(kl_per_var.sum(axis=-1))

            if i == self.num_noise - 1:
                for j in range(i_layer, self.evoformer_num_block):
                    msa_recon_act, _ = self.evoformer_decoder[j](msa_recon_act, pair_act_list[-(j + 1)], \
                                                                 msa_mask, pair_mask, context_mask, mask_norm=mask_norm,
                                                                 res_idx=res_idx)

        q_act = msa_recon_act[0]
        q_logits = self.reconstruct_head_query_new(q_act)
        q_logits = q_logits.astype(mstype.float32) + 1e9 * P.Reshape()(self.reconstruct_mask - 1., (1, -1))
        q_logits += 1e9 * P.Reshape()(self.reconstruct_mask - 1., (1, -1))

        recon_logits = self.reconstruct_head(msa_recon_act).astype(mstype.float32)
        recon_logits += 1e9 * P.Reshape()(self.reconstruct_mask - 1., (1, 1, -1))

        hasdel_logits = self.reconstruct_head_hasdel(msa_recon_act).astype(mstype.float32)
        delnum_logits = self.reconstruct_head_delnum(msa_recon_act).astype(mstype.float32)

        logits = P.Concat(0)((P.ExpandDims()(q_logits, 0), recon_logits[1:]))

        no_del_prob, mean_delnum = self._compute_del_num(hasdel_logits, delnum_logits)

        return logits, no_del_prob, mean_delnum

    def _init_feat(self, q_raw_feat, msa_raw_feat, pair_raw_feat):
        '''init_feat'''
        q_feat = self.preprocess_1d(q_raw_feat)
        msa_feat = self.preprocess_msa(msa_raw_feat)
        msa_activations = self.expand_dims(q_feat, 0) + msa_feat

        pair_activations = self.pair_activations(pair_raw_feat)
        left_single = self.left_single(q_raw_feat)
        right_single = self.right_single(q_raw_feat)
        pair_activations += self.expand_dims(left_single, 1) + self.expand_dims(right_single, 0)
        return msa_activations, pair_activations

    def _compute_del_num(self, hasdel_logits, delnum_logits):
        '''compute_del_num'''
        hasdel_logits = P.Squeeze(-1)(hasdel_logits.astype(mstype.float32))
        no_del_prob = P.Sigmoid()(hasdel_logits)
        mean_delnum = P.Softmax(-1)(delnum_logits.astype(mstype.float32)) * P.Reshape()(self.del_num_bins, (1, 1, -1))
        mean_delnum = P.ReduceSum()(mean_delnum, -1)
        return no_del_prob, mean_delnum


class MegaEvogen(nn.Cell):
    '''MegaEvogen'''

    def __init__(self, msa_model_config, model_cfg, mixed_precision):
        super().__init__()
        self.msa_vae = MsaGen(msa_model_config)
        self.feat_process = EvoGenFeatProcess(
            config=msa_model_config,
        )
        self.megafold = MegaFold(model_cfg, mixed_precision)

        self.softmax_temperature = msa_model_config.train.softmax_temperature
        self.use_gumbel_trick = msa_model_config.train.use_gumbel_trick
        self.use_dark_knowledge = msa_model_config.train.use_dark_knowledge
        self.uniform = msd.Uniform(1e-5, 1. - 1e-5, dtype=mstype.float32)

        augmented_msa_depth = min(msa_model_config.train.augmented_msa_depth, msa_model_config.train.max_msa_clusters)
        augmented_msa_mask = np.ones([msa_model_config.train.max_msa_clusters])
        augmented_msa_mask[augmented_msa_depth:] *= 0
        self.augmented_msa_mask = Tensor(augmented_msa_mask, mstype.float32)
        self.onehot = nn.OneHot(depth=msa_model_config.global_config.num_aa_types + 1)
        self.concat = P.Concat(-1)
        self.softmax = nn.Softmax()

    def construct(self, target_feat, seq_mask, aatype, residx_atom37_to_atom14, atom37_atom_exists,
                  residue_index, msa_mask, msa_data, query_data, addition_data, random_data,
                  random_mask, fake_template_aatype, fake_template_all_atom_masks,
                  fake_template_all_atom_positions, fake_template_mask,
                  fake_template_pseudo_beta_mask, fake_template_pseudo_beta,
                  fake_extra_msa, fake_extra_has_deletion, fake_extra_deletion_value,
                  fake_extra_msa_mask, prev_pos, prev_msa_first_row, prev_pair):
        '''construct'''
        msa_mask_new = msa_mask[:, 0].astype(mstype.float32)
        context_mask = random_mask[:, 0].astype(mstype.float32)
        target_mask = random_mask[:, 1].astype(mstype.float32)

        context_mask_new = context_mask * msa_mask_new
        target_mask_new = target_mask * context_mask * msa_mask_new

        random_mask_correct = P.Stack(-1)((context_mask_new, target_mask_new))
        msa_input = self.concat((msa_data, P.ExpandDims()(msa_mask, -1)))

        _, feat_tuple = self.feat_process(query_data, msa_input, addition_data,
                                          random_data, random_mask_correct)
        q_raw_feat, msa_raw_feat, pair_raw_feat, msa_mask, pair_mask, context_mask, target_mask, \
        res_idx, random_feat = feat_tuple
        msa_logits, no_del_prob, mean_delnum = self.msa_vae(q_raw_feat, msa_raw_feat,
                                                            pair_raw_feat, msa_mask, pair_mask,
                                                            context_mask, target_mask, res_idx,
                                                            random_feat)

        msa_prob = self.softmax(msa_logits * self.softmax_temperature)

        msa_reduce = P.Argmax(axis=-1)(msa_logits)
        msa = self.onehot(msa_reduce)

        if self.use_gumbel_trick:
            gumbel = self.uniform.sample(msa_logits.shape).astype(msa_logits.dtype)
            msa_reduce = P.Argmax(axis=-1)(msa_logits / self.softmax_temperature + gumbel)
            msa = self.onehot(msa_reduce)

        if self.use_dark_knowledge:
            msa = msa_prob

        pad_zero = P.ZerosLike()(msa)[:, :, :1]
        msa_feat_new_generate = self.concat((msa, pad_zero, pad_zero, msa, pad_zero))

        has_del_prob = 1. - no_del_prob
        del_num_feat = has_del_prob * mean_delnum
        has_del_prob = P.ExpandDims()(has_del_prob, -1)
        del_num_feat = P.ExpandDims()(del_num_feat, -1)
        has_del_prob[0] *= 0.
        del_num_feat[0] *= 0.
        msa_feat_new_reconstruct = self.concat((msa, has_del_prob, del_num_feat, msa, del_num_feat))

        recon_mask = target_mask_new
        recon_mask_new = P.Reshape()(recon_mask, (-1, 1, 1))
        gen_mask_new = P.Reshape()((1. - recon_mask), (-1, 1, 1))
        msa_feat_new = recon_mask_new * msa_feat_new_reconstruct + gen_mask_new * msa_feat_new_generate
        msa_mask_af2 = self.augmented_msa_mask
        msa_mask_new = P.ExpandDims()(msa_mask_af2, 1) * P.ExpandDims()(seq_mask, 0)

        msa_feat_new = msa_feat_new * P.ExpandDims()(msa_mask_new, -1)

        prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = \
            self.megafold(target_feat, msa_feat_new, msa_mask_new, seq_mask, aatype,
                          fake_template_aatype, fake_template_all_atom_masks, fake_template_all_atom_positions,
                          fake_template_mask, fake_template_pseudo_beta_mask, fake_template_pseudo_beta, fake_extra_msa,
                          fake_extra_has_deletion, fake_extra_deletion_value, fake_extra_msa_mask,
                          residx_atom37_to_atom14, atom37_atom_exists, residue_index,
                          prev_pos, prev_msa_first_row, prev_pair)
        result = prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits
        return result

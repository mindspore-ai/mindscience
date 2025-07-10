# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend


import pathlib
import mindspore as ms
from mindspore import ops
from alphafold3.model.params import get_model_af3_params


def np_slice(arr, i, j, dtype=ms.bfloat16):
    if i is not None and j is not None:
        return ms.Parameter(ms.Tensor(arr[i, j], dtype))
    if i is not None and j is None:
        return ms.Parameter(ms.Tensor(arr[i], dtype))
    if i is None and j is not None:
        return ms.Parameter(ms.Tensor(arr[j], dtype))
    return ms.Parameter(ms.Tensor(arr, dtype))



def load_adaptive_layernorm(adaptive_layernorm, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    if not ckpt.get(f'{path}single_cond_layer_norm'):
        adaptive_layernorm.layernorm.layernorm.gamma.set_data(
            np_slice(ckpt[f'{path}layer_norm']['scale'], i, j, dtype=ms.float32))
        adaptive_layernorm.layernorm.layernorm.beta.set_data(
            np_slice(ckpt[f'{path}layer_norm']['offset'], i, j, dtype=ms.float32))
    else:
        adaptive_layernorm.single_cond_layer_norm.layernorm.gamma.set_data(
            np_slice(ckpt[f'{path}single_cond_layer_norm']['scale'], i, j, dtype=ms.float32))
        adaptive_layernorm.single_cond_scale.weight.set_data(
            np_slice(ckpt[f'{path}single_cond_scale']['weights'], i, j, dtype=dtype))
        adaptive_layernorm.single_cond_scale.bias.set_data(
            np_slice(ckpt[f'{path}single_cond_scale']['bias'], i, j, dtype=dtype))
        adaptive_layernorm.single_cond_bias.weight.set_data(
            np_slice(ckpt[f'{path}single_cond_bias']['weights'], i, j, dtype=dtype))


def load_adaptive_zero_init(adaptive_zero_init, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    adaptive_zero_init.cond_linear1.weight.set_data(
        np_slice(ckpt[f'{path}transition2']['weights'], i, j, dtype=dtype))
    if ckpt.get(f'{path}adaptive_zero_cond'):
        adaptive_zero_init.cond_linear2.weight.set_data(
            np_slice(ckpt[f'{path}adaptive_zero_cond']['weights'], i, j, dtype=dtype))
        adaptive_zero_init.cond_linear2.bias.set_data(
            np_slice(ckpt[f'{path}adaptive_zero_cond']['bias'], i, j, dtype=dtype))


def load_transition(transition_block, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    load_adaptive_layernorm(
        transition_block.adaptive_layernorm, f'{path}ffw_', ckpt, i, j, dtype=dtype)
    transition_block.weights.set_data(
        np_slice(ckpt[f'{path}ffw_transition1']['weights'], i, j, dtype=dtype).reshape(
            (transition_block.weights.shape[0], 2, transition_block.num_intermediate)))
    load_adaptive_zero_init(
        transition_block.adaptive_zero_init, f'{path}ffw_', ckpt, i, j, dtype=dtype)


def load_self_attention(self_attention, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    load_adaptive_layernorm(
        self_attention.adaptive_layernorm, path, ckpt, i, j)
    self_attention.q_linear.weight.set_data(
        np_slice(ckpt[f'{path}q_projection']['weights'], i, j, dtype=dtype))
    self_attention.q_linear.bias.set_data(
        np_slice(ckpt[f'{path}q_projection']['bias'], i, j, dtype=dtype))
    self_attention.k_linear.weight.set_data(
        np_slice(ckpt[f'{path}k_projection']['weights'], i, j, dtype=dtype))
    self_attention.v_linear.weight.set_data(
        np_slice(ckpt[f'{path}v_projection']['weights'], i, j, dtype=dtype))
    self_attention.linear.weight.set_data(
        np_slice(ckpt[f'{path}gating_query']['weights'], i, j, dtype=dtype))
    load_adaptive_zero_init(
        self_attention.adaptive_zero_init, path, ckpt, i, j, dtype=dtype)


def load_transformer(transformer, path, ckpt, dtype=ms.bfloat16):
    for i in range(6):
        for j in range(4):
            transformer_path = (path +
                                '/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer')
            load_self_attention(transformer.super_blocks[i].blocks[j].self_attention,
                                transformer_path, ckpt, i, j, dtype=dtype)
            load_transition(transformer.super_blocks[i].blocks[j].transition_block,
                            transformer_path, ckpt, i, j, dtype=dtype)
        if transformer.using_pair_act is True:
            pair_projection_path = f'{path}/__layer_stack_with_per_layer/pair_logits_projection'
            transformer.super_blocks[i].pair_linear.weight.set_data(
                np_slice(ckpt[pair_projection_path]['weights'], i, None, dtype=dtype))
    if transformer.using_pair_act is True:
        pair_norm_path = f'{path}/pair_input_layer_norm'
        transformer.pair_layernorm.layernorm.gamma.set_data(
            np_slice(ckpt[pair_norm_path]['scale'].T, dtype=ms.float32))


def load_transition_block(transition_block, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    transition_block.glu_weight.set_data(
        np_slice(ckpt[f'{path}/transition1']['weights'], i, j, dtype=dtype).reshape(
            (-1, 2, transition_block.num_intermediate)))
    transition_block.out_linear.weight.set_data(
        np_slice(ckpt[f'{path}/transition2']['weights'], i, j, dtype=dtype))
    transition_block.layernorm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/input_layer_norm']['scale'], i, j, dtype=ms.float32))
    transition_block.layernorm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/input_layer_norm']['offset'], i, j, dtype=ms.float32))


def load_grid_self_attention(grid_self_attention, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    grid_self_attention.q_projection.weight.set_data(
        np_slice(ckpt[f'{path}/q_projection']['weights'], i, j, dtype=dtype).transpose(2, 0, 1))
    grid_self_attention.k_projection.weight.set_data(
        np_slice(ckpt[f'{path}/k_projection']['weights'], i, j, dtype=dtype).transpose(2, 0, 1))
    grid_self_attention.v_projection.weight.set_data(
        np_slice(ckpt[f'{path}/v_projection']['weights'], i, j, dtype=dtype))
    grid_self_attention.gating_query.weight.set_data(
        np_slice(ckpt[f'{path}/gating_query']['weights'], i, j, dtype=dtype).T)
    grid_self_attention.output_projection.weight.set_data(
        np_slice(ckpt[f'{path}/output_projection']['weights'], i, j, dtype=dtype))
    grid_self_attention.pair_bias_projection.weight.set_data(
        np_slice(ckpt[f'{path}/pair_bias_projection']['weights'], i, j, dtype=dtype))
    grid_self_attention.act_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/act_norm']['scale'], i, j, dtype=ms.float32))
    grid_self_attention.act_norm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/act_norm']['offset'], i, j, dtype=ms.float32))


def load_outer_product_mean(outer_product_mean, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    outer_product_mean.outer_product_mean.o_biases.set_data(
        np_slice(ckpt[path]['output_b'], i, j, dtype=dtype))
    outer_product_mean.outer_product_mean.linear_output_weight.set_data(
        np_slice(ckpt[path]['output_w'], i, j, dtype=dtype))
    outer_product_mean.outer_product_mean.left_projection_weight.set_data(
        np_slice(ckpt[f'{path}/left_projection']['weights'], i, j, dtype=dtype).T)
    outer_product_mean.outer_product_mean.right_projection_weight.set_data(
        np_slice(ckpt[f'{path}/right_projection']['weights'], i, j, dtype=dtype).T)
    outer_product_mean.outer_product_mean.layer_norm_input_gamma.set_data(
        np_slice(ckpt[f'{path}/layer_norm_input']['scale'], i, j, dtype=ms.float32))
    outer_product_mean.outer_product_mean.layer_norm_input_beta.set_data(
        np_slice(ckpt[f'{path}/layer_norm_input']['offset'], i, j, dtype=ms.float32))


def load_msa_attention(msa_attention, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    msa_attention.actnorm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/act_norm']['scale'], i, j, dtype=ms.float32))
    msa_attention.actnorm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/act_norm']['offset'], i, j, dtype=ms.float32))
    msa_attention.pairnorm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/pair_norm']['scale'], i, j, dtype=ms.float32))
    msa_attention.pairnorm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/pair_norm']['offset'], i, j, dtype=ms.float32))
    msa_attention.pair_logits.weight.set_data(
        np_slice(ckpt[f'{path}/pair_logits']['weights'], i, j, dtype=dtype))
    msa_attention.v_projection.weight.set_data(
        np_slice(ckpt[f'{path}/v_projection']['weights'], i, j, dtype=dtype))
    msa_attention.gating_query.weight.set_data(
        np_slice(ckpt[f'{path}/gating_query']['weights'], i, j, dtype=dtype))
    msa_attention.output_projection.weight.set_data(
        np_slice(ckpt[f'{path}/output_projection']['weights'], i, j, dtype=dtype))


def load_triangle_multiplication(triangle_multiplication, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    triangle_multiplication.triangle_multi.gate.weight.set_data(
        np_slice(ckpt[f'{path}/gate']['weights'], i, j, dtype=dtype).T)
    triangle_multiplication.triangle_multi.projection.weight.set_data(
        np_slice(ckpt[f'{path}/projection']['weights'], i, j, dtype=dtype).T)
    triangle_multiplication.triangle_multi.weight_glu = ops.stack(
        [triangle_multiplication.triangle_multi.gate.weight,
         triangle_multiplication.triangle_multi.projection.weight], axis=1)
    triangle_multiplication.triangle_multi.output_projection.weight.set_data(
        np_slice(ckpt[f'{path}/output_projection']['weights'], i, j, dtype=dtype))
    triangle_multiplication.triangle_multi.gating_linear.weight.set_data(
        np_slice(ckpt[f'{path}/gating_linear']['weights'], i, j, dtype=dtype))
    triangle_multiplication.triangle_multi.left_norm_input.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/left_norm_input']['scale'], i, j, dtype=ms.float32))
    triangle_multiplication.triangle_multi.left_norm_input.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/left_norm_input']['offset'], i, j, dtype=ms.float32))
    triangle_multiplication.triangle_multi.center_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/center_norm']['scale'], i, j, dtype=ms.float32))
    triangle_multiplication.triangle_multi.center_norm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/center_norm']['offset'], i, j, dtype=ms.float32))


def load_pair_former(pair_former, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    load_grid_self_attention(pair_former.grid_self_attention1, f'{path}/pair_attention1',
                             ckpt, i, j, dtype=dtype)
    load_grid_self_attention(pair_former.grid_self_attention2, f'{path}/pair_attention2',
                             ckpt, i, j, dtype=dtype)
    load_triangle_multiplication(pair_former.triangle_multiplication1,
                                 f'{path}/triangle_multiplication_outgoing', ckpt, i, j, dtype=dtype)
    load_triangle_multiplication(pair_former.triangle_multiplication2,
                                 f'{path}/triangle_multiplication_incoming', ckpt, i, j, dtype=dtype)
    load_transition_block(pair_former.transition_block, f'{path}/pair_transition',
                          ckpt, i, j, dtype=dtype)
    if pair_former.with_single:
        pair_former.single_pair_logits_norm.layernorm.gamma.set_data(
            np_slice(ckpt[f'{path}/single_pair_logits_norm']['scale'], i, j, dtype=ms.float32))
        pair_former.single_pair_logits_norm.layernorm.beta.set_data(
            np_slice(ckpt[f'{path}/single_pair_logits_norm']['offset'], i, j, dtype=ms.float32))
        pair_former.single_pair_logits_projection.weight.set_data(
            np_slice(ckpt[f'{path}/single_pair_logits_projection']['weights'], i, j, dtype=dtype))
        load_self_attention(pair_former.single_attention, f'{path}/single_attention_',
                            ckpt, i, j, dtype=dtype)
        load_transition_block(pair_former.single_transition, f'{path}/single_transition',
                              ckpt, i, j, dtype=dtype)


def load_evo_former(evo_former, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    load_outer_product_mean(evo_former.outer_product_mean, f'{path}/outer_product_mean',
                            ckpt, i, j, dtype=dtype)
    load_msa_attention(evo_former.msa_attention, f'{path}/msa_attention1',
                       ckpt, i, j, dtype=dtype)
    load_transition_block(evo_former.msa_transition, f'{path}/msa_transition',
                          ckpt, i, j, dtype=dtype)
    load_triangle_multiplication(evo_former.triangle_multiplication1,
                                 f'{path}/triangle_multiplication_outgoing', ckpt, i, j, dtype=dtype)
    load_triangle_multiplication(evo_former.triangle_multiplication2,
                                 f'{path}/triangle_multiplication_incoming', ckpt, i, j, dtype=dtype)
    load_grid_self_attention(evo_former.pair_attention1, f'{path}/pair_attention1',
                             ckpt, i, j, dtype=dtype)
    load_grid_self_attention(evo_former.pair_attention2, f'{path}/pair_attention2',
                             ckpt, i, j, dtype=dtype)
    load_transition_block(evo_former.transition_block, f'{path}/pair_transition',
                          ckpt, i, j, dtype=dtype)


def load_single_template_embedding(single_template_embedding, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    num_layer = single_template_embedding.config.template_stack.num_layer
    for ii in range(num_layer):
        template_path = f'{path}/__layer_stack_no_per_layer/template_embedding_iteration'
        load_pair_former(single_template_embedding.template_stack[ii], template_path,
                         ckpt, ii, dtype=dtype)
    for jj in range(9):
        template_pair_path = f'{path}/template_pair_embedding_{jj}'
        single_template_embedding.template_pair_embedding[jj].weight.set_data(
            np_slice(ckpt[template_pair_path]['weights'], None, None, dtype=dtype))
    single_template_embedding.output_layer_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/output_layer_norm']['scale'], i, j, dtype=ms.float32))
    single_template_embedding.output_layer_norm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/output_layer_norm']['offset'], i, j, dtype=ms.float32))
    single_template_embedding.query_embedding_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/query_embedding_norm']['scale'], i, j, dtype=ms.float32))
    single_template_embedding.query_embedding_norm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/query_embedding_norm']['offset'], i, j, dtype=ms.float32))


def load_template_embedding(template_embedding, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    template_embedding.output_linear.weight.set_data(
        np_slice(ckpt[f'{path}/output_linear']['weights'], i, j, dtype=dtype))
    load_single_template_embedding(template_embedding.template_embedder,
                                   f'{path}/single_template_embedding', ckpt, i, j, dtype=dtype)


def load_distogram_head(distogram_head, path, ckpt, i=None, j=None, dtype=ms.float32):
    distogram_head.linear.weight.set_data(
        np_slice(ckpt[f'{path}/half_logits']['weights'], i, j, dtype=dtype))


def load_evoformer(evoformer, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    relative_encoding_path = f'{path}/~_relative_encoding/position_activations'
    evoformer.position_activations.weight.set_data(
        np_slice(ckpt[relative_encoding_path]['weights'], i, j, dtype=dtype))
    evoformer.left_single.weight.set_data(
        np_slice(ckpt[f'{path}/left_single']['weights'], i, j, dtype=dtype))
    evoformer.right_single.weight.set_data(
        np_slice(ckpt[f'{path}/right_single']['weights'], i, j, dtype=dtype))
    evoformer.bond_embedding.weight.set_data(
        np_slice(ckpt[f'{path}/bond_embedding']['weights'], i, j, dtype=dtype))
    evoformer.msa_activations.weight.set_data(
        np_slice(ckpt[f'{path}/msa_activations']['weights'], i, j, dtype=dtype))
    evoformer.extra_msa_target_feat.weight.set_data(
        np_slice(ckpt[f'{path}/extra_msa_target_feat']['weights'], i, j, dtype=dtype))
    evoformer.prev_embedding.weight.set_data(
        np_slice(ckpt[f'{path}/prev_embedding']['weights'], i, j, dtype=dtype))
    evoformer.prev_embedding_layer_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/prev_embedding_layer_norm']['scale'], i, j, dtype=ms.float32))
    evoformer.prev_embedding_layer_norm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/prev_embedding_layer_norm']['offset'], i, j, dtype=ms.float32))
    evoformer.single_activations.weight.set_data(
        np_slice(ckpt[f'{path}/single_activations']['weights'], i, j, dtype=dtype))
    evoformer.prev_single_embedding.weight.set_data(
        np_slice(ckpt[f'{path}/prev_single_embedding']['weights'], i, j, dtype=dtype))
    evoformer.prev_single_embedding_layer_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/prev_single_embedding_layer_norm']['scale'], i, j, dtype=ms.float32))
    evoformer.prev_single_embedding_layer_norm.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/prev_single_embedding_layer_norm']['offset'], i, j, dtype=ms.float32))
    load_template_embedding(evoformer.template_module, f'{path}/template_embedding',
                            ckpt, i, j, dtype=dtype)
    for ii in range(evoformer.config.pairformer.num_layer):
        pairformer_path = path+'/__layer_stack_no_per_layer_1/trunk_pairformer'
        load_pair_former(
            evoformer.pairformer_stack[ii], pairformer_path, ckpt, ii, dtype=dtype)
    for jj in range(evoformer.config.msa_stack.num_layer):
        msa_stack_path = path+'/__layer_stack_no_per_layer/msa_stack'
        load_evo_former(
            evoformer.evoformer_stack[jj], msa_stack_path, ckpt, jj, dtype=dtype)


def load_adaptive_layernorm_ms(adaptive_layernorm, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    if not ckpt.get(f'{path}single_cond_layer_norm'):
        adaptive_layernorm.layernorm.layernorm.gamma.set_data(
            np_slice(ckpt[f'{path}layer_norm']['scale'], i, j, dtype=dtype))
        adaptive_layernorm.layernorm.layernorm.beta.set_data(
            np_slice(ckpt[f'{path}layer_norm']['offset'], i, j, dtype=dtype))
    else:
        adaptive_layernorm.single_cond_layer_norm.layernorm.gamma.set_data(
            np_slice(ckpt[f'{path}single_cond_layer_norm']['scale'], i, j, dtype=dtype))
        adaptive_layernorm.single_cond_scale.weight.set_data(
            np_slice(ckpt[f'{path}single_cond_scale']['weights'], i, j, dtype=dtype))
        adaptive_layernorm.single_cond_scale.bias.set_data(
            np_slice(ckpt[f'{path}single_cond_scale']['bias'], i, j, dtype=dtype))
        adaptive_layernorm.single_cond_bias.weight.set_data(
            np_slice(ckpt[f'{path}single_cond_bias']['weights'], i, j, dtype=dtype))


def load_adaptive_zero_init_ms(adaptive_zero_init, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    adaptive_zero_init.cond_linear1.weight.set_data(
        np_slice(ckpt[f'{path}transition2']['weights'], i, j, dtype=dtype))
    if ckpt.get(f'{path}adaptive_zero_cond'):
        adaptive_zero_init.cond_linear2.weight.set_data(
            np_slice(ckpt[f'{path}adaptive_zero_cond']['weights'], i, j, dtype=dtype))
        adaptive_zero_init.cond_linear2.bias.set_data(
            np_slice(ckpt[f'{path}adaptive_zero_cond']['bias'], i, j, dtype=dtype))


def load_transition_ms(transition_block, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    load_adaptive_layernorm_ms(
        transition_block.adaptive_layernorm, f'{path}ffw_', ckpt, i, j, dtype=dtype)
    transition_block.weights.set_data(
        np_slice(ckpt[f'{path}ffw_transition1']['weights'], i, j, dtype=dtype).reshape(
            (transition_block.weights.shape[0], 2, transition_block.num_intermediate)))
    load_adaptive_zero_init_ms(
        transition_block.adaptive_zero_init, f'{path}ffw_', ckpt, i, j, dtype=dtype)


def load_self_attention_ms(self_attention, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    load_adaptive_layernorm_ms(
        self_attention.adaptive_layernorm, path, ckpt, i, j, dtype=dtype)
    self_attention.q_linear.weight.set_data(
        np_slice(ckpt[f'{path}q_projection']['weights'], i, j, dtype=dtype))
    self_attention.q_linear.bias.set_data(
        np_slice(ckpt[f'{path}q_projection']['bias'], i, j, dtype=dtype))
    self_attention.k_linear.weight.set_data(
        np_slice(ckpt[f'{path}k_projection']['weights'], i, j, dtype=dtype))
    self_attention.v_linear.weight.set_data(
        np_slice(ckpt[f'{path}v_projection']['weights'], i, j, dtype=dtype))
    self_attention.linear.weight.set_data(
        np_slice(ckpt[f'{path}gating_query']['weights'], i, j, dtype=dtype))
    load_adaptive_zero_init_ms(
        self_attention.adaptive_zero_init, path, ckpt, i, j, dtype=dtype)


def load_transformer_ms(transformer, path, ckpt, dtype=ms.float16):
    for i in range(6):
        for j in range(4):
            transformer_path = (path +
                                f'/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer')
            load_self_attention_ms(transformer.super_blocks[i].blocks[j].self_attention,
                                   transformer_path, ckpt, i, j, dtype=dtype)
            load_transition_ms(transformer.super_blocks[i].blocks[j].transition_block,
                               transformer_path, ckpt, i, j, dtype=dtype)
        if transformer.using_pair_act:
            pair_projection_path = path + f'/__layer_stack_with_per_layer/pair_logits_projection'
            transformer.super_blocks[i].pair_linear.weight.set_data(
                np_slice(ckpt[pair_projection_path]['weights'], i, None, dtype=dtype))
    if transformer.using_pair_act:
        pair_norm_path = f'{path}/pair_input_layer_norm'
        transformer.pair_layernorm.layernorm.gamma.set_data(
            np_slice(ckpt[pair_norm_path]['scale'].T, None, None, dtype=ms.float32))


def load_cross_attention(cross_attention, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    load_adaptive_layernorm_ms(
        cross_attention.adaptive_layernorm_q, f'{path}q', ckpt, i, j, dtype=dtype)
    load_adaptive_layernorm_ms(
        cross_attention.adaptive_layernorm_k, f'{path}k', ckpt, i, j, dtype=dtype)
    cross_attention.linear_q.weight.set_data(
        np_slice(ckpt[f'{path}q_projection']['weights'], i, j, dtype=dtype))
    cross_attention.linear_q.bias.set_data(
        np_slice(ckpt[f'{path}q_projection']['bias'], i, j, dtype=dtype))
    cross_attention.linear_k.weight.set_data(
        np_slice(ckpt[f'{path}k_projection']['weights'], i, j, dtype=dtype))
    cross_attention.linear_v.weight.set_data(
        np_slice(ckpt[f'{path}v_projection']['weights'], i, j, dtype=dtype))
    cross_attention.gating_query.weight.set_data(
        np_slice(ckpt[f'{path}gating_query']['weights'], i, j, dtype=dtype))
    load_adaptive_zero_init_ms(
        cross_attention.adaptive_zero_init, path, ckpt, i, j, dtype=dtype)


def load_cross_att_transformer_block(cross_att_transformer_block, path, ckpt, i=None, dtype=ms.bfloat16):
    load_cross_attention(
        cross_att_transformer_block.cross_attention, path, ckpt, i, dtype=dtype)
    load_transition_ms(cross_att_transformer_block.transition,
                       path, ckpt, i, dtype=dtype)


def load_cross_attention_transformer(cross_attention_transformer, path, ckpt, last_name, i, j, dtype=ms.bfloat16):
    cross_attention_transformer.pair_input_layer_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/pair_input_layer_norm']['scale'], i, j, dtype=dtype))
    cross_attention_transformer.pair_logits_projection.weight.set_data(
        np_slice(ckpt[f'{path}/pair_logits_projection']['weights'], i, j, dtype=dtype))
    for ii in range(cross_attention_transformer.config.num_blocks):
        block_path = path + f'/__layer_stack_with_per_layer/{last_name}'
        load_cross_att_transformer_block(cross_attention_transformer.block[ii], block_path,
                                         ckpt, ii, dtype=dtype)


def load_per_atom_conditioning(per_atom_conditioning, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    per_atom_conditioning.linear1.weight.set_data(
        np_slice(ckpt[f'{path}_embed_ref_pos']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear2.weight.set_data(
        np_slice(ckpt[f'{path}_embed_ref_mask']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear3.weight.set_data(
        np_slice(ckpt[f'{path}_embed_ref_element']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear4.weight.set_data(
        np_slice(ckpt[f'{path}_embed_ref_charge']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear5.weight.set_data(
        np_slice(ckpt[f'{path}_embed_ref_atom_name']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear_row_act.weight.set_data(
        np_slice(ckpt[f'{path}_single_to_pair_cond_row']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear_col_act.weight.set_data(
        np_slice(ckpt[f'{path}_single_to_pair_cond_col']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear_pair_act1.weight.set_data(
        np_slice(ckpt[f'{path}_embed_pair_offsets']['weights'].T, i, j, dtype=dtype))
    per_atom_conditioning.linear_pair_act2.weight.set_data(
        np_slice(ckpt[f'{path}_embed_pair_distances']['weights'].T, i, j, dtype=dtype))


def load_atom_cross_encoder(atom_cross_att_encoder, path, ckpt, last_name, i=None, j=None, dtype=ms.bfloat16):
    load_per_atom_conditioning(
        atom_cross_att_encoder._per_atom_conditioning, path, ckpt, dtype=dtype)
    if atom_cross_att_encoder.with_cond:
        atom_cross_att_encoder._embed_trunk_single_cond.weight.set_data(
            np_slice(ckpt[f'{path}_embed_trunk_single_cond']['weights'].T, i, j, dtype=dtype))
        atom_cross_att_encoder._lnorm_trunk_single_cond.layernorm.gamma.set_data(
            np_slice(ckpt[f'{path}_lnorm_trunk_single_cond']['scale'], i, j, dtype=ms.float32))
        atom_cross_att_encoder._atom_positions_to_features.weight.set_data(
            np_slice(ckpt[f'{path}_atom_positions_to_features']['weights'].T, i, j, dtype=dtype))
        atom_cross_att_encoder._embed_trunk_pair_cond.weight.set_data(
            np_slice(ckpt[f'{path}_embed_trunk_pair_cond']['weights'].T, i, j, dtype=dtype))
        atom_cross_att_encoder._lnorm_trunk_pair_cond.layernorm.gamma.set_data(
            np_slice(ckpt[f'{path}_lnorm_trunk_pair_cond']['scale'], i, j, dtype=ms.float32))
    atom_cross_att_encoder._single_to_pair_cond_row.weight.set_data(
        np_slice(ckpt[f'{path}_single_to_pair_cond_row_1']['weights'].T, i, j, dtype=dtype))
    atom_cross_att_encoder._single_to_pair_cond_col.weight.set_data(
        np_slice(ckpt[f'{path}_single_to_pair_cond_col_1']['weights'].T, i, j, dtype=dtype))
    if atom_cross_att_encoder.with_cond:
        atom_cross_att_encoder._embed_pair_offsets.weight.set_data(
            np_slice(ckpt[f'{path}_embed_pair_offsets_1']['weights'].T, i, j, dtype=dtype))
        atom_cross_att_encoder._embed_pair_distances.weight.set_data(
            np_slice(ckpt[f'{path}_embed_pair_distances_1']['weights'].T, i, j, dtype=dtype))
    else:
        atom_cross_att_encoder._embed_pair_offsets.weight.set_data(
            np_slice(ckpt[f'{path}_embed_pair_offsets']['weights'].T, i, j, dtype=dtype))
        atom_cross_att_encoder._embed_pair_distances.weight.set_data(
            np_slice(ckpt[f'{path}_embed_pair_distances']['weights'].T, i, j, dtype=dtype))
    atom_cross_att_encoder._embed_pair_offsets_valid.weight.set_data(
        np_slice(ckpt[f'{path}_embed_pair_offsets_valid']['weights'].T, i, j, dtype=dtype))
    atom_cross_att_encoder._pair_mlp_1.weight.set_data(
        np_slice(ckpt[f'{path}_pair_mlp_1']['weights'].T, i, j, dtype=dtype))
    atom_cross_att_encoder._pair_mlp_2.weight.set_data(
        np_slice(ckpt[f'{path}_pair_mlp_2']['weights'].T, i, j, dtype=dtype))
    atom_cross_att_encoder._pair_mlp_3.weight.set_data(
        np_slice(ckpt[f'{path}_pair_mlp_3']['weights'].T, i, j, dtype=dtype))
    atom_cross_att_encoder._project_atom_features_for_aggr.weight.set_data(
        np_slice(ckpt[f'{path}_project_atom_features_for_aggr']['weights'].T, i, j, dtype=dtype))
    load_cross_attention_transformer(atom_cross_att_encoder._atom_transformer_encoder,
                                     f'{path}_atom_transformer_encoder', ckpt,
                                     f"{last_name}_atom_transformer_encoder", i, j, dtype=dtype)


def load_atom_cross_decoder(atom_cross_att_decoder, path, ckpt, i=None, j=None, dtype=ms.bfloat16):
    atom_cross_att_decoder._project_token_features_for_broadcast.weight.set_data(
        np_slice(ckpt[f'{path}_project_token_features_for_broadcast']['weights'].T, i, j, dtype=dtype))
    atom_cross_att_decoder._atom_features_layer_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}_atom_features_layer_norm']['scale'], i, j, dtype=ms.float32))
    atom_cross_att_decoder._atom_features_to_position_update.weight.set_data(
        np_slice(ckpt[f'{path}_atom_features_to_position_update']['weights'].T, i, j, dtype=dtype))
    load_cross_attention_transformer(atom_cross_att_decoder._atom_transformer_decoder,
                                     f'{path}_atom_transformer_decoder', ckpt,
                                     last_name='diffusion_atom_transformer_decoder', i=i, j=j, dtype=dtype)


def load_diffusion_head(diffusion_head, path, ckpt, i=None, j=None, dtype=ms.float32):
    diffusion_head.pair_cond_initial_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/pair_cond_initial_norm']['scale'], i, j, dtype=ms.float32))
    diffusion_head.pair_cond_initial_projection.weight.set_data(
        np_slice(ckpt[f'{path}/pair_cond_initial_projection']['weights'].T, i, j, dtype=ms.float32))
    load_transition_ms(diffusion_head.transition_block1,
                       f'{path}/pair_transition_0', ckpt, dtype=dtype)
    load_transition_ms(diffusion_head.transition_block2,
                       f'{path}/pair_transition_1', ckpt, dtype=dtype)
    diffusion_head.single_cond_initial_norm.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/single_cond_initial_norm']['scale'], i, j, dtype=ms.float32))
    diffusion_head.single_cond_initial_projection.weight.set_data(
        np_slice(ckpt[f'{path}/single_cond_initial_projection']['weights'].T, i, j, dtype=dtype))
    diffusion_head.layer_norm_noise.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/noise_embedding_initial_norm']['scale'], i, j, dtype=ms.float32))
    diffusion_head.linear_noise.weight.set_data(
        np_slice(ckpt[f'{path}/noise_embedding_initial_projection']['weights'].T, i, j, dtype=dtype))
    load_transition_ms(diffusion_head.single_transition1,
                       f'{path}/single_transition_0', ckpt, dtype=dtype)
    load_transition_ms(diffusion_head.single_transition2,
                       f'{path}/single_transition_1', ckpt, dtype=dtype)
    diffusion_head.layer_norm_act.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/single_cond_embedding_norm']['scale'], i, j, dtype=ms.float32))
    diffusion_head.linear_act.weight.set_data(
        np_slice(ckpt[f'{path}/single_cond_embedding_projection']['weights'].T, i, j, dtype=dtype))
    diffusion_head.layer_norm_out.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/output_norm']['scale'], i, j, dtype=ms.float32))
    load_atom_cross_encoder(diffusion_head.atom_cross_att_encoder, f'{path}/diffusion', ckpt,
                            last_name="diffusion", dtype=dtype)
    load_transformer_ms(diffusion_head.transformer, path +
                        '/transformer', ckpt, dtype=dtype)
    load_atom_cross_decoder(
        diffusion_head.atom_cross_att_decoder, f'{path}/diffusion', ckpt, dtype=dtype)


def load_confidence_head(confidence_head, path, ckpt, i=None, j=None, dtype=ms.float32):
    confidence_head.left_target_feat_project.weight.set_data(
        np_slice(ckpt[f'{path}/~_embed_features/left_target_feat_project']['weights'].T, i, j, dtype=dtype))
    confidence_head.right_target_feat_project.weight.set_data(
        np_slice(ckpt[f'{path}/~_embed_features/right_target_feat_project']['weights'].T, i, j, dtype=dtype))
    confidence_head.distogram_feat_project.weight.set_data(
        np_slice(ckpt[f'{path}/~_embed_features/distogram_feat_project']['weights'].T, i, j, dtype=dtype))
    for ii in range(confidence_head.config.pairformer.num_layer):
        confidence_pairformer_path = path + \
            f'/__layer_stack_no_per_layer/confidence_pairformer'
        load_pair_former(confidence_head.pairformer_block[ii], confidence_pairformer_path,
                         ckpt, ii, dtype=dtype)
    confidence_head.left_half_distance_logits.weight.set_data(
        np_slice(ckpt[f'{path}/left_half_distance_logits']['weights'].T, i, j, dtype=ms.float32))
    confidence_head.logits_ln.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/logits_ln']['scale'], i, j, dtype=ms.float32))
    confidence_head.logits_ln.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/logits_ln']['offset'], i, j, dtype=ms.float32))
    confidence_head.pae_logits.weight.set_data(
        np_slice(ckpt[f'{path}/pae_logits']['weights'].T, i, j, dtype=ms.float32))
    confidence_head.pae_logits_ln.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/pae_logits_ln']['scale'], i, j, dtype=ms.float32))
    confidence_head.pae_logits_ln.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/pae_logits_ln']['offset'], i, j, dtype=ms.float32))
    confidence_head.plddt_logits.weight.set_data(
        np_slice(ckpt[f'{path}/plddt_logits']['weights'], i, j, dtype=ms.float32))
    confidence_head.plddt_logits_ln.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/plddt_logits_ln']['scale'], i, j, dtype=ms.float32))
    confidence_head.plddt_logits_ln.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/plddt_logits_ln']['offset'], i, j, dtype=ms.float32))
    confidence_head.experimentally_resolved_logits.weight.set_data(
        np_slice(ckpt[f'{path}/experimentally_resolved_logits']['weights'], i, j, dtype=ms.float32))
    confidence_head.experimentally_resolved_ln.layernorm.gamma.set_data(
        np_slice(ckpt[f'{path}/experimentally_resolved_ln']['scale'], i, j, dtype=ms.float32))
    confidence_head.experimentally_resolved_ln.layernorm.beta.set_data(
        np_slice(ckpt[f'{path}/experimentally_resolved_ln']['offset'], i, j, dtype=ms.float32))


def load_diffuser(diffuser, ckpt_dir, dtype=ms.bfloat16):
    path = 'diffuser'
    ckpt = get_model_af3_params(pathlib.Path(ckpt_dir))
    load_evoformer(diffuser.embedding_module, path +
                   '/evoformer', ckpt, dtype=dtype)
    load_distogram_head(diffuser.distogram_head, path +
                        '/distogram_head', ckpt, dtype=ms.float32)
    load_atom_cross_encoder(diffuser.create_target_feat_embedding.atom_cross_att_encoder,
                            f'{path}/evoformer_conditioning', ckpt,
                            last_name='evoformer_conditioning', dtype=ms.float32)
    load_diffusion_head(diffuser.diffusion_module, path +
                        '/~/diffusion_head', ckpt, dtype=ms.float32)
    load_confidence_head(diffuser.confidence_head, path +
                         '/confidence_head', ckpt, dtype=ms.float32)

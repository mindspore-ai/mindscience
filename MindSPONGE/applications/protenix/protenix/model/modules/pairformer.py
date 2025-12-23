# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 ByteDance and/or its affiliates.
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

"""pairformer"""

from dataclasses import dataclass
import mindspore as ms
from mindspore import nn, ops, mint
from protenix.model.modules import base_modules as bm
from protenix.model.modules.primitives import TransitionBlock
from protenix.model.modules import transformer
from protenix.model.modules.triangle_multiplication import TriangleMultiplication
from protenix.model.modules.triangular_attention import GridSelfAttention
from protenix.openfold_local.model.outer_product_mean import OuterProductMean
from protenix.model.modules.atom_cross_attention import AtomCrossAttEncoderConfig
from protenix.model.modules.module_utils.common import residue_names
from protenix.model import base_config
from protenix.model.modules import featurization
from mindscience.e3nn.utils import Ncon


def chunk_forward(module, m, z=None, chunk_size=128):
    """chunk forward"""
    def fixed_length_chunk(m, chunk_length, dim=0):
        dim_size = m.shape[dim]
        chunk_num = (dim_size + chunk_length - 1) // chunk_length
        chunks = []

        for i in range(chunk_num):
            start = i * chunk_length
            end = min(start + chunk_length, dim_size)
            chunk = m.narrow(dim, start, end - start)
            chunks.append(chunk)

        return chunks

    m_chunks = fixed_length_chunk(m, chunk_size, dim=0)
    if z is not None:
        processed_chunks = [module(chunk, z) for chunk in m_chunks]
    else:
        processed_chunks = [module(chunk) for chunk in m_chunks]
    m = mint.cat(processed_chunks, dim=0)
    return m


class MSAAttention(nn.Cell):
    """
    Multi-Head Self-Attention (MSA) attention mechanism for processing sequence and pair data.

    Args:
        config (Config): Configuration object containing parameters for the attention mechanism.
        global_config (GlobalConfig): Global configuration object.
        act_shape (tuple): Shape of the activation tensor.
        pair_shape (tuple): Shape of the pair tensor.

    Inputs:
        - **act** (Tensor) - Input activation tensor.
        - **mask** (Tensor) - Mask tensor to prevent attention weights from focusing on invalid positions.
        - **pair_act** (Tensor) - Pair activation tensor.

    Outputs:
        - **output** (Tensor) - Output tensor after processing through the attention mechanism.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        num_head: int = 8

    def __init__(self, config, global_config, act_shape, pair_shape, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.actnorm = bm.LayerNorm(act_shape, dtype=ms.float32)
        self.pairnorm = bm.LayerNorm(pair_shape, dtype=ms.float32)
        num_channel = act_shape[-1]
        value_dim = num_channel // self.config.num_head
        self.pair_logits = bm.CustomDense(pair_shape[-1], self.config.num_head, use_bias=False,
                                          weight_init='ones', ndim=3, dtype=dtype)  # None, change to ones for test
        self.v_projection = bm.CustomDense(num_channel, (self.config.num_head, value_dim),
                                           use_bias=False, ndim=len(act_shape), dtype=dtype)
        ncon_list1 = [-3, -2, 1]
        ncon_list2 = [-1, 1, -3, -4]
        self.ncon = Ncon([ncon_list1, ncon_list2])
        self.gating_query = bm.CustomDense(
            num_channel, self.config.num_head * value_dim, weight_init='zeros', use_bias=False, ndim=3, dtype=dtype)
        self.output_projection = bm.CustomDense(self.config.num_head * value_dim, num_channel,
                                                weight_init=self.global_config.final_init, use_bias=False,
                                                ndim=3, dtype=dtype)

    def construct(self, act, pair_act):
        """construct"""
        act = self.actnorm(act)
        pair_act = self.pairnorm(pair_act)
        logits = self.pair_logits(pair_act).transpose([2, 0, 1])
        # logits = logits + 1e9 * (mint.max(mask, dim=0)[0] - 1.0)
        weights = mint.softmax(logits, dim=-1)
        v = self.v_projection(act)
        v_avg = self.ncon([weights, v])
        v_avg = v_avg.reshape(v_avg.shape[:-2]+(-1,))
        gate_value = self.gating_query(act)
        v_avg = v_avg * \
            mint.sigmoid(gate_value.astype(ms.float32)
                         ).astype(gate_value.dtype)
        out = self.output_projection(v_avg)
        return out


class PairFormerIteration(nn.Cell):
    """
    Single Iteration of PairFormer, which processes pairwise and single activations in a single iteration.

    Args:
        config (PairFormerIteration.Config): Configuration for the PairFormerIteration module.
        global_config: Global configuration for the model.
        normalized_shape (tuple): Shape of the input tensor for normalization.
        single_shape (tuple | None): Shape of the single activation tensor. Default: ``None``.
        with_single (bool): Whether to include single activation processing. Default: ``False``.

    Inputs:
        - **act** (Tensor) - Pairwise activations tensor.
        - **pair_mask** (Tensor) - Padding mask for pairwise activations.
        - **single_act** (Tensor | None) - Single activations tensor, optional.
        - **seq_mask** (Tensor | None) - Sequence mask, optional.

    Outputs:
        - **act** (Tensor) - Processed pairwise activations tensor.
        - **single_act** (Tensor) - Processed single activations tensor (if `with_single` is True).
    """
    @dataclass
    class Config(base_config.BaseConfig):
        """Config for PairFormerIteration."""
        num_layer: int = 1
        pair_attention: GridSelfAttention.Config = base_config.autocreate()
        pair_transition: TransitionBlock.Config = base_config.autocreate()
        single_attention: "Union[transformer.SelfAttentionConfig, None]" = base_config.autocreate()  # None
        single_transition: "Union[TransitionBlock.Config, None]" = base_config.autocreate()  # None
        triangle_multiplication_incoming: TriangleMultiplication.Config = (
            base_config.autocreate(equation='kjc,kic->ijc')
        )
        triangle_multiplication_outgoing: TriangleMultiplication.Config = (
            base_config.autocreate(equation='ikc,jkc->ijc')
        )
        shard_transition_blocks: bool = True

    def __init__(self, config, global_config, normalized_shape, pair_shape,
                 single_shape=None, with_single=False, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.with_single = with_single
        num_channel = normalized_shape[-1]
        self.triangle_multiplication1 = TriangleMultiplication(
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            num_channel,
            normalized_shape,
            dtype=dtype
        )
        self.triangle_multiplication2 = TriangleMultiplication(
            self.config.triangle_multiplication_incoming,
            self.global_config,
            num_channel,
            normalized_shape,
            dtype=dtype
        )
        self.grid_self_attention1 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            False,
            normalized_shape,
            dtype=dtype
        )
        self.grid_self_attention2 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            True,
            normalized_shape,
            dtype=dtype
        )
        self.grid_self_attention1.recompute()
        self.grid_self_attention2.recompute()
        self.transition_block = TransitionBlock(
            self.config.pair_transition, self.global_config, normalized_shape, dtype=dtype
        )
        if self.with_single:
            if self.config.single_attention is None:
                raise ValueError("single_attention must not be None")
            self.single_attention = transformer.SelfAttention(
                self.config.single_attention, self.global_config, single_shape[-1],
                pair_shape, with_single_cond=False, dtype=dtype
            )
            self.single_transition = TransitionBlock(
                self.config.single_transition,
                self.global_config,
                single_shape,
                2,
                dtype=dtype
            )

    def construct(self, act, pair_mask, single_act=None):
        """construct"""
        act = act + self.triangle_multiplication1(act, pair_mask)
        act = act + self.triangle_multiplication2(act, pair_mask)
        act = act + self.grid_self_attention1(act, pair_mask)
        act = act + self.grid_self_attention2(act, pair_mask)
        act = act + self.transition_block(act)
        if self.with_single:
            single_act = single_act + self.single_attention(
                single_act, None, act
            )
            single_act = single_act + \
                self.single_transition(single_act)
            return act, single_act
        return act


class EvoformerIteration(nn.Cell):
    """
    EvoformerIteration is a single iteration of the Evoformer main stack, which processes
    activations and masks through a series of attention and transformation layers to
    update the MSA (Multiple Sequence Alignment) and pair representations.

    Args:
        config (EvoformerIteration.Config): Configuration for the EvoformerIteration.
        global_config (base_config.BaseConfig): Global configuration for the model.
        act_shape (tuple): Shape of the activation tensor.
        pair_shape (tuple): Shape of the pair tensor.

    Inputs:
        - **activations** (dict): A dictionary containing the MSA and pair activations.
        - **masks** (dict): A dictionary containing the MSA and pair masks.

    Outputs:
        - **activations** (dict): A dictionary containing the updated MSA and pair activations.
    """
    @dataclass
    class Config(base_config.BaseConfig):
        """Configuration for EvoformerIteration."""

        num_layer: int = 4
        msa_attention: MSAAttention.Config = base_config.autocreate()
        outer_product_mean: OuterProductMean.Config = base_config.autocreate()
        msa_transition: TransitionBlock.Config = base_config.autocreate()
        pair_attention: GridSelfAttention.Config = base_config.autocreate()
        pair_transition: TransitionBlock.Config = base_config.autocreate()
        triangle_multiplication_incoming: TriangleMultiplication.Config = (
            base_config.autocreate(equation='kjc,kic->ijc')
        )
        triangle_multiplication_outgoing: TriangleMultiplication.Config = (
            base_config.autocreate(equation='ikc,jkc->ijc')
        )
        shard_transition_blocks: bool = False

    def __init__(self, config, global_config, act_shape, pair_shape, last_block=False, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.last_block = last_block
        num_channel = pair_shape[-1]
        self.outer_product_mean = OuterProductMean(
            config=self.config.outer_product_mean,
            global_config=self.global_config,
            num_output_channel=num_channel,
            in_channel=act_shape[-1],
            dtype=dtype
        )
        if not self.last_block:
            self.msa_attention = MSAAttention(self.config.msa_attention,
                                              self.global_config, act_shape, pair_shape, dtype=dtype)
            self.msa_transition = TransitionBlock(
                self.config.msa_transition, self.global_config, act_shape, dtype=dtype
            )
        self.triangle_multiplication1 = TriangleMultiplication(
            self.config.triangle_multiplication_outgoing,
            self.global_config,
            num_channel,
            pair_shape,
            dtype=dtype
        )
        self.triangle_multiplication2 = TriangleMultiplication(
            self.config.triangle_multiplication_incoming,
            self.global_config,
            num_channel,
            pair_shape,
            dtype=dtype
        )
        self.pair_attention1 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            False,
            pair_shape,
            dtype=dtype
        )
        self.pair_attention2 = GridSelfAttention(
            self.config.pair_attention,
            self.global_config,
            True,
            pair_shape,
            dtype=dtype
        )
        self.pair_attention1.recompute()
        self.pair_attention2.recompute()
        self.transition_block = TransitionBlock(
            self.config.msa_transition, self.global_config, pair_shape, dtype=dtype
        )

    def construct(self, msa_activations, pair_activations, msa_mask, pair_mask):
        """construct"""
        msa_act, pair_act = msa_activations, pair_activations
        pair_act = pair_act + self.outer_product_mean(msa_act, msa_mask)
        chunk_size = 64
        if not self.last_block:
            if chunk_size is None:
                msa_act = msa_act + self.msa_attention(msa_act, pair_act)
                msa_act = msa_act + self.msa_transition(msa_act)
            else:
                msa_act = msa_act + \
                    chunk_forward(self.msa_attention, msa_act,
                                  pair_act, chunk_size=chunk_size)
                msa_act = msa_act + \
                    chunk_forward(self.msa_transition, msa_act, chunk_size=chunk_size)
        pair_act = pair_act + \
            self.triangle_multiplication1(pair_act, pair_mask)
        pair_act = pair_act + \
            self.triangle_multiplication2(pair_act, pair_mask)
        pair_act = pair_act + \
            self.pair_attention1(pair_act, pair_mask)
        # pair_act = pair_act.transpose(-2, -3)
        pair_act = pair_act + \
            self.pair_attention2(pair_act, pair_mask)
        # pair_act = pair_act.transpose(-2, -3)
        pair_act = pair_act + self.transition_block(pair_act)
        return msa_act, pair_act


class Evoformer(nn.Cell):
    """
    Evoformer class for generating 'single' and 'pair' embeddings in protein structure prediction.

    Args:
        config (Evoformer.Config): Configuration object defining the parameters for the Evoformer module.
        global_config (base_config.BaseConfig): Global configuration object containing general settings.
        feat_shape (tuple): Shape of the feature tensor.
        act_shape (tuple): Shape of the activation tensor.
        pair_shape (tuple): Shape of the pair tensor.
        single_shape (tuple): Shape of the single tensor.

    Inputs:
        - **batch** (dict): Dictionary containing batch data including token features, MSA,
            and other relevant information.
        - **prev** (dict): Dictionary containing previous embeddings for 'single' and 'pair' activations.
        - **target_feat** (Tensor): Target feature tensor used for generating embeddings.
        - **key** (int): Random key for reproducibility.

    Outputs:
        - **output** (dict): Dictionary containing the generated embeddings:
            - **single** (Tensor): Single residue embeddings.
            - **pair** (Tensor): Pairwise residue embeddings.
            - **target_feat** (Tensor): Target feature tensor.

    Notes:
        - The class processes input data through multiple modules including position encoding, 
            bond embedding, MSA processing, and Pairformer iterations.
        - The `construct` method iteratively processes the input data to generate rich 
            embeddings for downstream tasks in protein structure prediction.
    """
    @dataclass
    # pytype: disable=invalid-function-definition
    class PairformerConfig(PairFormerIteration.Config):
        block_remat: bool = False
        remat_block_size: int = 8

    @dataclass
    class Config(base_config.BaseConfig):
        """Configuration for Evoformer."""

        max_relative_chain: int = 2
        msa_channel: int = 64
        seq_channel: int = 384
        max_relative_idx: int = 32
        num_msa: int = 128  # 1024
        pair_channel: int = 128
        pairformer: 'Evoformer.PairformerConfig' = base_config.autocreate(
            single_transition=base_config.autocreate(),
            single_attention=base_config.autocreate(),
            num_layer=48,
        )
        per_atom_conditioning: AtomCrossAttEncoderConfig = (
            base_config.autocreate(
                per_token_channels=384,
                per_atom_channels=128,
                atom_transformer=base_config.autocreate(
                    num_intermediate_factor=2,
                    num_blocks=3,
                ),
                per_atom_pair_channels=16,
            )
        )
        msa_stack: EvoformerIteration.Config = base_config.autocreate()

    def __init__(self, config, global_config, feat_shape, act_shape, pair_shape, single_shape, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.config = config
        self.global_config = global_config
        in_channel = feat_shape[-1]
        position_activations_in = 4 * self.config.max_relative_idx + \
            4 + 2 * self.config.max_relative_chain + 2 + 1
        self.position_activations = bm.CustomDense(
            position_activations_in, self.config.pair_channel, ndim=3, dtype=dtype)
        self.left_single = bm.CustomDense(
            self.config.seq_channel, self.config.pair_channel, ndim=2, dtype=dtype)
        self.right_single = bm.CustomDense(
            self.config.seq_channel, self.config.pair_channel, ndim=2, dtype=dtype)
        self.bond_embedding = bm.CustomDense(
            1, self.config.pair_channel, ndim=3, dtype=dtype)
        self.msa_activations = bm.CustomDense(
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 3, self.config.msa_channel, ndim=3, dtype=dtype)
        self.extra_msa_target_feat = bm.CustomDense(
            in_channel, self.config.msa_channel, ndim=2, dtype=dtype)
        evofromer_act_shape = (self.config.num_msa,
                               act_shape[1], self.config.msa_channel)
        self.evoformer_stack = nn.CellList(
            [
                EvoformerIteration(
                    self.config.msa_stack, self.global_config, evofromer_act_shape, pair_shape,
                    last_block=(i == self.config.msa_stack.num_layer-1), dtype=dtype
                ) for i in range(self.config.msa_stack.num_layer)
            ]
        )
        for evo in self.evoformer_stack:
            evo.recompute()
        self.prev_embedding = bm.CustomDense(
            pair_shape[-1], pair_shape[-1], ndim=3, dtype=dtype)
        self.prev_embedding_layer_norm = bm.LayerNorm(
            pair_shape, dtype=ms.float32)
        self.prev_single_embedding = bm.CustomDense(
            self.config.seq_channel, self.config.seq_channel, ndim=2, dtype=dtype)
        self.prev_single_embedding_layer_norm = bm.LayerNorm(act_shape[:-1] +
                                                             (self.config.seq_channel,), dtype=ms.float32)
        self.pairformer_stack = nn.CellList(
            [
                PairFormerIteration(
                    self.config.pairformer, self.global_config, act_shape, pair_shape,
                    single_shape, with_single=True, dtype=dtype
                ) for _ in range(self.config.pairformer.num_layer)
            ]
        )

        for pair in self.pairformer_stack:
            pair.recompute()

        self.single_activations = bm.LinearAMP(
            in_channel, self.config.seq_channel, has_bias=False, dtype=dtype)

    def _relative_encoding(self, batch, pair_activations):
        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            self.config.max_relative_idx,
            self.config.max_relative_chain,
        )
        rel_feat = rel_feat.astype(pair_activations.dtype)
        pair_activations = pair_activations + \
            self.position_activations(rel_feat)
        return pair_activations

    def _seq_pair_embedding(self, target_feat):
        left_single = self.left_single(target_feat)[:, None]
        right_single = self.right_single(target_feat)[None]
        pair_activations = left_single + right_single
        pair_mask = None
        return pair_activations, pair_mask

    def _embed_bonds(self, batch, pair_activations):
        """Embeds bond features and merges into pair activations."""
        # Construct contact matrix.
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = ops.zeros((num_tokens, num_tokens))

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).astype(
                gather_idxs_polymer_ligand.dtype
            )[:, None]
        )
        # If valid mask then it will be all 1's, so idxs should be unchanged.
        gather_idxs_polymer_ligand = (
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        )
        tokens_to_ligand_ligand_bonds = (
            batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
        )
        gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
            dim=1
        ).astype(gather_idxs_ligand_ligand.dtype)[:, None]
        gather_idxs_ligand_ligand = (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )
        gather_idxs = ops.concat(
            [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
        )
        contact_matrix[gather_idxs[:, 0], gather_idxs[:, 1]] = 1.0
        contact_matrix[0, 0] = 0.0

        bonds_act = self.bond_embedding(
            contact_matrix[:, :, None].astype(pair_activations.dtype)
        )
        return pair_activations + bonds_act

    def _embed_process_msa(self, msa_batch, pair_activations, pair_mask, key, target_feat):
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype
        msa_batch = featurization.truncate_msa_batch(
            msa_batch, self.config.num_msa)
        msa_feat = featurization.create_msa_feat(msa_batch).astype(dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations = msa_activations + \
            self.extra_msa_target_feat(target_feat)[None]
        msa_mask = None

        # Evoformer MSA stack.
        for i in range(self.config.msa_stack.num_layer):
            msa_activations, pair_activations = self.evoformer_stack[i](
                msa_activations, pair_activations, msa_mask, pair_mask)
        return pair_activations, key

    def construct(self, batch, prev_pair, prev_single, init_pair, init_single, target_feat, key, idx=0):
        """construct"""
        dtype = self.dtype
        if idx == 0:
            single_activations_init = self.single_activations(target_feat)
            pair_activations_init, _ = self._seq_pair_embedding(
                single_activations_init
            )
            pair_activations_init = self._relative_encoding(
                batch, pair_activations_init)
            pair_activations_init = pair_activations_init + \
                self.bond_embedding(
                    batch.token_bonds.unsqueeze(-1).astype(dtype))
        else:
            pair_activations_init, single_activations_init = init_pair, init_single

        init = (pair_activations_init, single_activations_init)

        pair_activations = prev_pair.astype(dtype)
        single_activations = prev_single.astype(dtype)

        pair_activations = pair_activations_init + self.prev_embedding(
            self.prev_embedding_layer_norm(
                prev_pair
            ).astype(pair_activations.dtype)
        )
        pair_activations, key = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=None,
            key=key,
            target_feat=target_feat,
        )

        single_activations = single_activations_init + self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(
                prev_single.astype(single_activations.dtype)
            )
        )
        for i in range(self.config.pairformer.num_layer):
            pair_activations, single_activations = self.pairformer_stack[i](
                pair_activations, None, single_act=single_activations,
            )
        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }
        return output, init

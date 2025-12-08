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

"""
Model for AlphaFold3
"""
from dataclasses import dataclass
import random
import concurrent
import functools
from absl import logging
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from alphafold3.constants import residue_names
from alphafold3.model import base_config
from alphafold3.model import confidences
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import base_model
from alphafold3.model.components import base_modules as bm
from alphafold3.model.diffusion import atom_cross_attention
from alphafold3.model.diffusion import confidence_head
from alphafold3.model.diffusion import diffusion_head
from alphafold3.model.diffusion import distogram_head
from alphafold3.model.diffusion import featurization
from alphafold3.model.diffusion import modules
from alphafold3.model.diffusion import template_modules
from alphafold3.structure import mmcif


def get_predicted_structure(result, batch):
    """Creates the predicted structure and ion preditions.

    Args:
        result: model output in a model specific layout
        batch: model input batch

    Returns:
        Predicted structure.
    """
    model_output_coords = result['diffusion_samples']['atom_positions']

    # Rearrange model output coordinates to the flat output layout.
    model_output_to_flat = atom_layout.compute_gather_idxs(
        source_layout=batch.convert_model_output.token_atoms_layout,
        target_layout=batch.convert_model_output.flat_output_layout,
    )
    pred_flat_atom_coords = atom_layout.convert(
        gather_info=model_output_to_flat,
        arr=model_output_coords.asnumpy(),
        layout_axes=(-3, -2),
    )

    predicted_lddt = result.get('predicted_lddt')

    if predicted_lddt is not None:
        pred_flat_b_factors = atom_layout.convert(
            gather_info=model_output_to_flat,
            arr=predicted_lddt.asnumpy(),
            layout_axes=(-2, -1),
        )
    else:
        # Handle models which don't have predicted_lddt outputs.
        pred_flat_b_factors = np.zeros(pred_flat_atom_coords.shape[:-1])

    (missing_atoms_indices,) = np.nonzero(
        model_output_to_flat.gather_mask == 0)
    if missing_atoms_indices.shape[0] > 0:
        missing_atoms_flat_layout = batch.convert_model_output.flat_output_layout[
            missing_atoms_indices
        ]
        missing_atoms_uids = list(
            zip(
                missing_atoms_flat_layout.chain_id,
                missing_atoms_flat_layout.res_id,
                missing_atoms_flat_layout.res_name,
                missing_atoms_flat_layout.atom_name,
            )
        )
        logging.warning(
            'Target %s: warning: %s atoms were not predicted by the '
            'model, setting their coordinates to (0, 0, 0). '
            'Missing atoms: %s',
            batch.convert_model_output.empty_output_struc.name,
            missing_atoms_indices.shape[0],
            missing_atoms_uids,
        )

    # Put them into a structure
    pred_struc = batch.convert_model_output.empty_output_struc
    pred_struc = pred_struc.copy_and_update_atoms(
        atom_x=pred_flat_atom_coords[..., 0],
        atom_y=pred_flat_atom_coords[..., 1],
        atom_z=pred_flat_atom_coords[..., 2],
        atom_b_factor=pred_flat_b_factors,
        # Always 1.0.
        atom_occupancy=np.ones(pred_flat_atom_coords.shape[:-1]),
    )
    # Set manually/differently when adding metadata.
    pred_struc = pred_struc.copy_and_update_globals(release_date=None)
    return pred_struc


class CreateTargetFeatEmbedding(nn.Cell):
    """
    A class that creates target feature embeddings by combining raw features with cross-attention encoded features.

    Args:
        config (Config): Configuration object containing parameters for the target feature embedding.
        global_config (GlobalConfig): Global configuration object.

    Inputs:
        - **batch** (dict) - Dictionary containing batch features.

    Outputs:
        - **target_feat** (Tensor) - Tensor of target feature embeddings.
    """

    def __init__(self, config, global_config, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.dtype = dtype
        self.atom_cross_att_encoder = atom_cross_attention.AtomCrossAttEncoder(
            self.config.per_atom_conditioning, self.global_config, '', with_cond=False, dtype=dtype
        )

    def construct(self, batch):
        """Create target feature embedding."""
        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
            dtype=ms.float32
        ).astype(self.dtype)
        enc = self.atom_cross_att_encoder(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )
        target_feat = ops.concat(
            [target_feat, enc.token_act.astype(self.dtype)], axis=-1)
        return target_feat


def _compute_ptm(result, num_tokens, asym_id, pae_single_mask, interface):
    """Computes the pTM metrics from PAE."""
    return np.stack(
        [
            confidences.predicted_tm_score(
                tm_adjusted_pae=tm_adjusted_pae[:num_tokens, :num_tokens].asnumpy(
                ),
                asym_id=asym_id.asnumpy(),
                pair_mask=pae_single_mask[:num_tokens, :num_tokens],
                interface=interface,
            )
            for tm_adjusted_pae in result['tmscore_adjusted_pae_global']
        ],
        axis=0,
    )


def _compute_chain_pair_iptm(
        num_tokens,
        asym_ids,
        mask,
        tm_adjusted_pae):
    """Computes the chain pair ipTM metrics from PAE."""
    return np.stack(
        [
            confidences.chain_pairwise_predicted_tm_scores(
                tm_adjusted_pae=sample_tm_adjusted_pae[:num_tokens],
                asym_id=asym_ids[:num_tokens],
                pair_mask=mask[:num_tokens, :num_tokens],
            )
            for sample_tm_adjusted_pae in tm_adjusted_pae
        ],
        axis=0,
    )


class Diffuser(nn.Cell):
    """
    Diffuser class for processing and generating diffusion samples, confidence scores, and distanceograms.

    Args:
        config (Diffuser.Config): Configuration object containing parameters for the diffuser.
        in_channel (int): Number of input channels.
        feat_shape (tuple): Shape of the feature tensor.
        act_shape (tuple): Shape of the activation tensor.
        pair_shape (tuple): Shape of the pair tensor.
        single_shape (tuple): Shape of the single tensor.
        atom_shape (tuple): Shape of the atom tensor.
        out_channel (int): Number of output channels.
        num_templates (int): Number of templates.

    Inputs:
        - **batch** (dict): Dictionary containing batch data.
        - **key** (int): Random key generator.

    Outputs:
        - **result** (dict): Dictionary containing diffusion samples, distanceogram, and confidence outputs.
    """
    @dataclass
    class HeadsConfig(base_config.BaseConfig):
        diffusion: diffusion_head.DiffusionHead.Config = base_config.autocreate()
        confidence: confidence_head.ConfidenceHead.Config = base_config.autocreate()
        distogram: distogram_head.DistogramHead.Config = base_config.autocreate()

    @dataclass
    class Config(base_config.BaseConfig):
        evoformer: 'Evoformer.Config' = base_config.autocreate()
        global_config: model_config.GlobalConfig = base_config.autocreate()
        heads: 'Diffuser.HeadsConfig' = base_config.autocreate()
        num_recycles: int = 10
        return_embeddings: bool = False

    def __init__(self, config, in_channel, feat_shape, act_shape, pair_shape, single_shape, atom_shape,
                 out_channel, num_templates, dtype=ms.float32, name="model"):
        super().__init__(auto_prefix=True)
        self.config = config
        self.global_config = config.global_config
        self.dtype = dtype
        self.diffusion_module = diffusion_head.DiffusionHead(
            self.config.heads.diffusion, self.global_config, pair_shape, dtype=ms.float32
        )
        self.embedding_module = Evoformer(self.config.evoformer, self.global_config,
                                          feat_shape, act_shape, pair_shape, single_shape, num_templates, dtype=dtype)
        self.create_target_feat_embedding = CreateTargetFeatEmbedding(
            self.embedding_module.config, self.global_config, dtype=ms.float32)
        self.confidence_head = confidence_head.ConfidenceHead(
            self.config.heads.confidence, self.global_config,
            pair_shape, single_shape, atom_shape, feat_shape[-1], out_channel, dtype=dtype
        )
        self.distogram_head = distogram_head.DistogramHead(
            self.config.heads.distogram, self.global_config, pair_shape[-1], dtype=ms.float32
        )

    def _sample_diffusion(self, batch, embeddings, sample_config, key, init_positions=None):
        """Sample diffusion."""
        denoising_step = functools.partial(
            self.diffusion_module,
            batch=batch,
            embeddings=embeddings,
            use_conditioning=True,
        )
        sample = diffusion_head.sample(
            denoising_step=denoising_step,
            batch=batch,
            key=key+1,
            config=sample_config,
            init_positions=init_positions,
        )
        return sample

    def construct(self, batch, key):
        """Construct diffusion model."""
        if key is None:
            # generate a random number
            key = int(np.random.randint(100))
        # batch = feat_batch.Batch.from_data_dict(batch)
        target_feat = self.create_target_feat_embedding(
            batch)

        def recycle_body(prev, key):
            key, subkey = random.randint(0, 1e6), key
            embeddings = self.embedding_module(
                batch=batch,
                prev=prev,
                target_feat=target_feat,
                key=subkey,
            )
            embeddings['pair'] = embeddings['pair']
            embeddings['single'] = embeddings['single']
            return embeddings, key

        num_res = batch.num_res
        embeddings = {
            'pair': ops.zeros(
                [num_res, num_res, self.config.evoformer.pair_channel],
                dtype=ms.float32,
            ),
            'single': ops.zeros(
                [num_res, self.config.evoformer.seq_channel], dtype=ms.float32
            ),
            'target_feat': target_feat,
        }
        num_iter = self.config.num_recycles + 1
        for _ in range(num_iter):
            embeddings, _ = recycle_body(embeddings, key)

        samples = self._sample_diffusion(
            batch,
            embeddings,
            sample_config=self.config.heads.diffusion.eval,
            key=key
        )
        confidence_output = []
        for i in range(samples['atom_positions'].shape[0]):
            confidence_output.append(self.confidence_head(
                dense_atom_positions=samples['atom_positions'][i],
                embeddings=embeddings,
                seq_mask=batch.token_features.mask,
                token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch.token_features.asym_id,
            ))
        for key in confidence_output[0].keys():
            confidence_output[0][key] = ops.stack(
                [value[key] for value in confidence_output])
        confidence_output = confidence_output[0]
        distogram = self.distogram_head(batch, embeddings)
        output = {
            'diffusion_samples': samples,
            'distogram': distogram,
            **confidence_output,
        }
        if self.config.return_embeddings:
            output['single_embeddings'] = embeddings['single']
            output['pair_embeddings'] = embeddings['pair']
        return output

    @classmethod
    def get_inference_result(cls, batch, result, target_name,):
        """Get the predicted structure, scalars, and arrays for inference.

        This function also computes any inference-time quantities, which are not a
        part of the forward-pass, e.g. additional confidence scores. Note that this
        function is not serialized, so it should be slim if possible.

        Args:
        batch: data batch used for model inference, incl. TPU invalid types.
        result: output dict from the model's forward pass.
        target_name: target name to be saved within structure.

        Yields:
        inference_result: dataclass object that contains a predicted structure,
        important inference-time scalars and arrays, as well as a slightly trimmed
        dictionary of raw model result from the forward pass (for debugging).
        """
        del target_name
        # Retrieve structure and construct a predicted structure.
        pred_structure = get_predicted_structure(result=result, batch=batch)
        num_tokens = batch.token_features.seq_length.item()
        pae_single_mask = np.tile(
            batch.frames.mask[:, None],
            [1, batch.frames.mask.shape[0]],
        )
        ptm = _compute_ptm(
            result=result,
            num_tokens=num_tokens,
            asym_id=batch.token_features.asym_id[:num_tokens],
            pae_single_mask=pae_single_mask,
            interface=False,
        )
        iptm = _compute_ptm(
            result=result,
            num_tokens=num_tokens,
            asym_id=batch.token_features.asym_id[:num_tokens],
            pae_single_mask=pae_single_mask,
            interface=True,
        )
        ptm_iptm_average = 0.8 * iptm + 0.2 * ptm

        asym_ids = batch.token_features.asym_id[:num_tokens].asnumpy()
        chain_ids = [mmcif.int_id_to_str_id(asym_id) for asym_id in asym_ids]
        res_ids = batch.token_features.residue_index[:num_tokens]

        if len(np.unique(asym_ids)) > 1:
            # There is more than one chain, hence interface pTM (i.e. ipTM) defined,
            # so use it.
            ranking_confidence = ptm_iptm_average
        else:
            # There is only one chain, hence ipTM=NaN, so use just pTM.
            ranking_confidence = ptm

        contact_probs = result['distogram']['contact_probs'].astype(ms.float32)
        # Compute PAE related summaries.
        _, chain_pair_pae_min, _ = confidences.chain_pair_pae(
            num_tokens=num_tokens,
            asym_ids=batch.token_features.asym_id.asnumpy(),
            full_pae=result['full_pae'].asnumpy(),
            mask=pae_single_mask,
        )
        chain_pair_pde_mean, chain_pair_pde_min = confidences.chain_pair_pde(
            num_tokens=num_tokens,
            asym_ids=batch.token_features.asym_id.asnumpy(),
            full_pde=result['full_pde'].asnumpy(),
        )
        intra_chain_single_pde, cross_chain_single_pde, _ = confidences.pde_single(
            num_tokens,
            batch.token_features.asym_id.asnumpy(),
            result['full_pde'].asnumpy(),
            contact_probs.asnumpy(),
        )
        pae_metrics = confidences.pae_metrics(
            num_tokens=num_tokens,
            asym_ids=batch.token_features.asym_id.asnumpy(),
            full_pae=result['full_pae'].asnumpy(),
            mask=pae_single_mask,
            contact_probs=contact_probs.asnumpy(),
            tm_adjusted_pae=result['tmscore_adjusted_pae_interface'].asnumpy(),
        )
        ranking_confidence_pae = confidences.rank_metric(
            result['full_pae'].asnumpy(),
            contact_probs.asnumpy() * batch.frames.mask[:, None].astype(float),
        )
        chain_pair_iptm = _compute_chain_pair_iptm(
            num_tokens=num_tokens,
            asym_ids=batch.token_features.asym_id.asnumpy(),
            mask=pae_single_mask,
            tm_adjusted_pae=result['tmscore_adjusted_pae_interface'].asnumpy(),
        )
        # iptm_ichain is a vector of per-chain ptm values. iptm_ichain[0],
        # for example, is just the zeroth diagonal entry of the chain pair iptm
        # matrix:
        # [[x, , ],
        #  [ , , ],
        #  [ , , ]]]
        iptm_ichain = chain_pair_iptm.diagonal(axis1=-2, axis2=-1)
        # iptm_xchain is a vector of cross-chain interactions for each chain.
        # iptm_xchain[0], for example, is an average of chain 0's interactions with
        # other chains:
        # [[ ,x,x],
        #  [x, , ],
        #  [x, , ]]]
        iptm_xchain = confidences.get_iptm_xchain(chain_pair_iptm)

        predicted_distance_errors = result['average_pde']

        # Computing solvent accessible area with dssp can be slow for large
        # structures with lots of chains, so we parallelize the call.
        pred_structures = pred_structure.unstack()
        num_workers = len(pred_structures)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
        ) as executor:
            has_clash = list(executor.map(
                confidences.has_clash, pred_structures))
            fraction_disordered = list(
                executor.map(confidences.fraction_disordered, pred_structures)
            )
        for idx, pred_structure in enumerate(pred_structures):
            ranking_score = confidences.get_ranking_score(
                ptm=ptm[idx],
                iptm=iptm[idx],
                fraction_disordered_=fraction_disordered[idx],
                has_clash_=has_clash[idx],
            )
            print(f"####### result {idx} ######")
            print(f"####### ranking_score {ranking_score} ######")
            print(f"####### predicted_tm_score {ptm[idx]} ######")
            print(f"####### interface_predicted_tm_score {iptm[idx]} ######")
            yield base_model.InferenceResult(
                predicted_structure=pred_structure,
                numerical_data={
                    'full_pde': result['full_pde'][idx, :num_tokens, :num_tokens],
                    'full_pae': result['full_pae'][idx, :num_tokens, :num_tokens],
                    'contact_probs': contact_probs[:num_tokens, :num_tokens],
                },
                metadata={
                    'predicted_distance_error': predicted_distance_errors[idx],
                    'ranking_score': ranking_score,
                    'fraction_disordered': fraction_disordered[idx],
                    'has_clash': has_clash[idx],
                    'predicted_tm_score': ptm[idx],
                    'interface_predicted_tm_score': iptm[idx],
                    'chain_pair_pde_mean': chain_pair_pde_mean[idx],
                    'chain_pair_pde_min': chain_pair_pde_min[idx],
                    'chain_pair_pae_min': chain_pair_pae_min[idx],
                    'ptm': ptm[idx],
                    'iptm': iptm[idx],
                    'ptm_iptm_average': ptm_iptm_average[idx],
                    'intra_chain_single_pde': intra_chain_single_pde[idx],
                    'cross_chain_single_pde': cross_chain_single_pde[idx],
                    'pae_ichain': pae_metrics['pae_ichain'][idx],
                    'pae_xchain': pae_metrics['pae_xchain'][idx],
                    'ranking_confidence': ranking_confidence[idx],
                    'ranking_confidence_pae': ranking_confidence_pae[idx],
                    'chain_pair_iptm': chain_pair_iptm[idx],
                    'iptm_ichain': iptm_ichain[idx],
                    'iptm_xchain': iptm_xchain[idx],
                    'token_chain_ids': chain_ids,
                    'token_res_ids': res_ids,
                },
            )


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
        num_templates (int): Number of templates used in the model.

    Inputs:
        - **batch** (dict): Dictionary containing batch data including token features, MSA, and other
          relevant information.
        - **prev** (dict): Dictionary containing previous embeddings for 'single' and 'pair' activations.
        - **target_feat** (Tensor): Target feature tensor used for generating embeddings.
        - **key** (int): Random key for reproducibility.

    Outputs:
        - **output** (dict): Dictionary containing the generated embeddings:
            - **single** (Tensor): Single residue embeddings.
            - **pair** (Tensor): Pairwise residue embeddings.
            - **target_feat** (Tensor): Target feature tensor.

    Notes:
        - The class processes input data through multiple modules including position encoding, bond embedding,
          template embedding, MSA processing, and Pairformer iterations.
        - The `construct` method iteratively processes the input data to generate rich embeddings for
          downstream tasks in protein structure prediction.
    """
    @dataclass
    # pytype: disable=invalid-function-definition
    class PairformerConfig(modules.PairFormerIteration.Config):
        block_remat: bool = False
        remat_block_size: int = 8

    @dataclass
    class Config(base_config.BaseConfig):
        """Configuration for Evoformer."""

        max_relative_chain: int = 2
        msa_channel: int = 64
        seq_channel: int = 384
        max_relative_idx: int = 32
        num_msa: int = 1024
        pair_channel: int = 128
        pairformer: 'Evoformer.PairformerConfig' = base_config.autocreate(
            single_transition=base_config.autocreate(),
            single_attention=base_config.autocreate(),
            num_layer=48,
        )
        per_atom_conditioning: atom_cross_attention.AtomCrossAttEncoderConfig = (
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
        template: template_modules.TemplateEmbedding.Config = (
            base_config.autocreate()
        )
        msa_stack: modules.EvoformerIteration.Config = base_config.autocreate()

    def __init__(self, config, global_config, feat_shape, act_shape, pair_shape, single_shape,
                 num_templates, dtype=ms.float32):
        super().__init__()
        self.config = config
        self.global_config = global_config
        in_channel = feat_shape[-1]
        position_activations_in = 4 * self.config.max_relative_idx + \
            4 + 2 * self.config.max_relative_chain + 2 + 1
        self.position_activations = bm.CustomDense(
            position_activations_in, self.config.pair_channel, ndim=3, dtype=dtype)
        self.left_single = bm.CustomDense(
            in_channel, self.config.pair_channel, ndim=2, dtype=dtype)
        self.right_single = bm.CustomDense(
            in_channel, self.config.pair_channel, ndim=2, dtype=dtype)
        self.bond_embedding = bm.CustomDense(
            1, self.config.pair_channel, ndim=3, dtype=dtype)
        self.template_module = template_modules.TemplateEmbedding(
            self.config.template, self.global_config, num_templates, act_shape, dtype=dtype
        )
        self.msa_activations = bm.CustomDense(
            residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP + 3, self.config.msa_channel, ndim=3, dtype=dtype)
        self.extra_msa_target_feat = bm.CustomDense(
            in_channel, self.config.msa_channel, ndim=2, dtype=dtype)
        evofromer_act_shape = (self.config.num_msa,
                               act_shape[1], self.config.msa_channel)
        self.evoformer_stack = nn.CellList(
            [
                modules.EvoformerIteration(
                    self.config.msa_stack, self.global_config, evofromer_act_shape, pair_shape, dtype=dtype
                ) for _ in range(self.config.msa_stack.num_layer)
            ]
        )
        self.prev_embedding = bm.CustomDense(
            pair_shape[-1], pair_shape[-1], ndim=3, dtype=dtype)
        self.prev_embedding_layer_norm = bm.LayerNorm(
            pair_shape, dtype=ms.float32)
        self.single_activations = bm.CustomDense(
            in_channel, self.config.seq_channel, ndim=2, dtype=dtype)
        self.prev_single_embedding = bm.CustomDense(
            self.config.seq_channel, self.config.seq_channel, ndim=2, dtype=dtype)
        self.prev_single_embedding_layer_norm = bm.LayerNorm(act_shape[:-1] +
                                                             (self.config.seq_channel,), dtype=ms.float32)
        self.pairformer_stack = nn.CellList(
            [
                modules.PairFormerIteration(
                    self.config.pairformer, self.global_config, pair_shape, single_shape, with_single=True, dtype=dtype
                ) for _ in range(self.config.pairformer.num_layer)
            ]
        )

    def _relative_encoding(self, batch, pair_activations):
        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            self.config.max_relative_idx,
            self.config.max_relative_chain,
        )
        rel_feat = rel_feat.astype(pair_activations.dtype)
        pair_activations += self.position_activations(rel_feat)
        return pair_activations

    def _seq_pair_embedding(self, token_features, target_feat):
        left_single = self.left_single(target_feat)[:, None]
        right_single = self.right_single(target_feat)[None]
        dtype = left_single.dtype
        pair_activations = left_single + right_single
        mask = token_features.mask
        pair_mask = (mask[:, None] * mask[None, :]).astype(dtype)
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

    def _embed_template_pair(self, batch, pair_activations, pair_mask, key):
        """Embeds Templates and merges into pair activations."""
        dtype = pair_activations.dtype
        key, subkey = key + 1, key

        templates = batch.templates
        asym_id = batch.token_features.asym_id
        # Construct a mask such that only intra-chain template features are
        # computed, since all templates are for each chain individually.
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).astype(dtype)
        template_fn = functools.partial(self.template_module, key=subkey)
        template_act = template_fn(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask,
        )
        return pair_activations + template_act, key

    def _embed_process_msa(self, msa_batch, pair_activations, pair_mask, key, target_feat):
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype
        msa_batch = featurization.truncate_msa_batch(
            msa_batch, self.config.num_msa)
        msa_feat = featurization.create_msa_feat(msa_batch).astype(dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat)[None]
        msa_mask = msa_batch.mask.astype(dtype)
        # Evoformer MSA stack.
        evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
        mask = {'msa': msa_mask, 'pair': pair_mask}
        for i in range(self.config.msa_stack.num_layer):
            evoformer_input = self.evoformer_stack[i](evoformer_input, mask)

        return evoformer_input['pair'], key

    def construct(self, batch, prev, target_feat, key):
        """evoformer"""
        dtype = (ms.bfloat16 if self.global_config.bfloat16 ==
                 'all' else ms.float32)
        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )
        pair_activations += self.prev_embedding(
            self.prev_embedding_layer_norm(
                prev['pair']
            ).astype(pair_activations.dtype)
        )
        pair_activations = self._relative_encoding(batch, pair_activations)
        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )
        pair_activations, key = self._embed_template_pair(
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            key=key,
        )
        pair_activations, key = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            key=key,
            target_feat=target_feat,
        )
        del key  # Unused after this point.
        single_activations = self.single_activations(target_feat)

        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(
                prev['single'].astype(single_activations.dtype)
            )
        )
        for i in range(self.config.pairformer.num_layer):
            pair_activations, single_activations = self.pairformer_stack[i](
                pair_activations, pair_mask, single_act=single_activations,
                seq_mask=batch.token_features.mask.astype(dtype)
            )
        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output

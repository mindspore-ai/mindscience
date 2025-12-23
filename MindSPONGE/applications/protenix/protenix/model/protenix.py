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

"""protenix model"""

import random
from dataclasses import dataclass
import numpy as np
import mindspore as ms
from mindspore import nn, ops, mint, _no_grad
from protenix.model import base_config
from protenix.model import model_config
from protenix.model.modules.diffusion import DiffusionHead
from protenix.model.modules.pairformer import Evoformer
from protenix.model.modules.target_feat_embedding import CreateTargetFeatEmbedding
from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.modules.head import DistogramHead
from protenix.model.generator import sample_diffusion, sample_diffusion_training, TrainingNoiseSampler
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.model import sample_confidence

SIGMA_DATA = 16.0


class NoiseSchedule(nn.Cell):
    def __init__(self):
        super().__init__(self)

    def construct(self, t, smin=0.0004, smax=160.0, p=7):
        out = SIGMA_DATA * (smax ** (1 / p) + t *
                            (smin ** (1 / p) - smax ** (1 / p))) ** p
        return out


class Protenix(nn.Cell):
    """
    protenix class for processing and generating diffusion samples, confidence scores, and distanceograms.

    Args:
        config (Protenix.Config): Configuration object containing parameters for the Protenix.
        in_channel (int): Number of input channels.
        feat_shape (tuple): Shape of the feature tensor.
        act_shape (tuple): Shape of the activation tensor.
        pair_shape (tuple): Shape of the pair tensor.
        single_shape (tuple): Shape of the single tensor.
        max_atoms_per_token (int): max_atoms_per_token.
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
        diffusion: DiffusionHead.Config = base_config.autocreate()
        confidence: ConfidenceHead.Config = base_config.autocreate()
        distogram: DistogramHead.Config = base_config.autocreate()

    @dataclass
    class Config(base_config.BaseConfig):
        evoformer: 'Evoformer.Config' = base_config.autocreate()
        global_config: model_config.GlobalConfig = base_config.autocreate()
        heads: 'Protenix.HeadsConfig' = base_config.autocreate()
        num_recycles: int = 10
        return_embeddings: bool = False
        train_num_recycles: int = 0

    def __init__(self, config, protenix_config, feat_shape, act_shape, pair_shape, single_shape,
                 out_channel, num_templates, loss=None, is_train=False, dtype=ms.float32, recompute=True):  # pylint: disable=unused-argument
        super().__init__(auto_prefix=True)
        self.config = config
        self.protenix_config = protenix_config
        self.global_config = config.global_config
        self.dtype = dtype
        self.is_train = is_train
        max_atoms_per_token = self.protenix_config.max_atoms_per_token
        if self.is_train:
            self.diffusion_module = DiffusionHead(
                self.config.heads.diffusion, self.global_config, pair_shape, ndim=3, dtype=ms.float32
            )
            self.symmetric_permutation = SymmetricPermutation(
                self.protenix_config, error_dir=None
            )
        else:
            self.diffusion_module = DiffusionHead(
                self.config.heads.diffusion, self.global_config, pair_shape, ndim=2, dtype=ms.float32
            )
        self.embedding_module = Evoformer(self.config.evoformer, self.global_config,
                                          feat_shape, act_shape, pair_shape, single_shape, dtype=dtype)
        self.create_target_feat_embedding = CreateTargetFeatEmbedding(
            self.embedding_module.config, self.global_config, dtype=ms.float32)
        self.confidence_head = ConfidenceHead(
            self.config.heads.confidence, self.global_config, act_shape,
            pair_shape, single_shape, max_atoms_per_token, feat_shape[-1], out_channel, dtype=ms.float32
        )
        self.distogram_head = DistogramHead(
            self.config.heads.distogram, self.global_config, pair_shape[-1], dtype=ms.float32
        )
        self.train_noise_sampler = TrainingNoiseSampler()
        self.loss = loss
        self.noise_schedule = NoiseSchedule()
        if recompute and is_train:
            self.diffusion_module.recompute()
            self.symmetric_permutation.recompute()
            self.embedding_module.recompute()
            self.confidence_head.recompute()
            self.create_target_feat_embedding.recompute()
            self.noise_schedule.recompute()

    def _sample_diffusion(self, batch, embeddings,):
        noise_levels = self.noise_schedule(mint.linspace(
            0, 1, self.diffusion_module.config.eval.steps + 1))
        sample = sample_diffusion(
            self.diffusion_module, batch, embeddings, noise_levels,
            n_sample=self.protenix_config.n_sample)
        return sample, noise_levels

    def _sample_diffusion_training(self, batch, embeddings, label_dict=None):
        x_gt_augment, x_denoised, sigma, mask = sample_diffusion_training(
            self.train_noise_sampler,
            self.diffusion_module,
            label_dict,
            batch,
            embeddings,
            n_sample=self.protenix_config.diffusion_batch_size)
        return x_gt_augment, x_denoised, sigma, mask

    def training_block(self, batch, key):
        """training block"""
        label_dict = batch.label_dict
        label_full_dict = batch.label_full_dict
        pred_dict = {}
        if key is None:
            # generate a random number
            key = int(np.random.randint(100))
        target_feat = self.create_target_feat_embedding(
            batch).astype(self.dtype)

        def recycle_body(prev, key, idx, init=None):
            key, subkey = random.randint(0, 1e6), key
            embeddings, init = self.embedding_module(
                batch=batch,
                prev_pair=prev['pair'],
                prev_single=prev['single'],
                init_pair=init[0],
                init_single=init[1],
                target_feat=target_feat,
                key=subkey,
                idx=idx,
            )

            embeddings['pair'] = embeddings['pair'].astype(ms.float32)
            embeddings['single'] = embeddings['single'].astype(ms.float32)
            return embeddings, init

        num_res = batch.num_tokens
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
        self.config.train_num_recycles = 10
        num_iter = int(np.random.randint(
            1, self.config.train_num_recycles + 1))
        init = (None, None)
        for i in range(num_iter):
            if i == num_iter - 1:
                embeddings, init = recycle_body(embeddings, key, i, init)
            else:
                with _no_grad():
                    embeddings, init = recycle_body(embeddings, key, i, init)

        with _no_grad():
            noise_levels = self.noise_schedule(mint.linspace(
                0, 1, self.diffusion_module.config.train_mini.steps + 1))
            coordinate_mini = sample_diffusion(self.diffusion_module,
                                               batch,
                                               embeddings,
                                               noise_levels,
                                               n_sample=1)
            pred_dict["coordinates_mini"] = coordinate_mini

            label_dict, _ = (
                self.symmetric_permutation('permute_label_to_match_mini_rollout',
                                           coordinate_mini['atom_positions'][0],
                                           batch,
                                           label_dict,
                                           label_full_dict,
                                           )
            )

        confidence_output = self.confidence_head(
            dense_atom_positions=pred_dict["coordinates_mini"]['atom_positions'][0],
            embeddings_single=embeddings['single'],
            embeddings_pair=embeddings['pair'],
            embeddings_target=embeddings['target_feat'],
            seq_mask=None,
            distogram_rep_atom_mask=batch.distogram_rep_atom_mask,
            batch=batch
        )

        pred_dict.update(
            {
                "plddt": confidence_output['predicted_lddt'],
                "pae": confidence_output['predicted_pae'],
                "pde": confidence_output['predicted_pde'],
                "resolved": confidence_output['predicted_experimentally_resolved'],
            }
        )
        _, x_denoised, x_noise_level, mask = self._sample_diffusion_training(
            batch,
            embeddings,
            label_dict=label_dict,
        )

        breaks, contact_probs, distogram_logits = self.distogram_head(
            embeddings['pair']
        )
        distogram = {
            'bin_edges': breaks,
            'contact_probs': contact_probs,
            'logits': distogram_logits,
        }

        pred_dict.update(
            {
                "distogram": distogram,
                # [..., n_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
                "coordinate_mask": mask,
            }
        )
        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        pred_dict, *_ = (
            self.symmetric_permutation.permute_diffusion_sample_to_match_label(
                batch, pred_dict, label_dict, stage="train"
            )
        )

        cum_loss = self.get_loss(batch, pred_dict, self.loss, label_dict)
        pred_dict = ms.Tensor([0])
        return cum_loss, pred_dict

    def inference(self, batch, key):
        """inference block"""
        if key is None:
            # generate a random number
            key = int(np.random.randint(100))
        target_feat = self.create_target_feat_embedding(
            batch).astype(self.dtype)

        def recycle_body(prev, key, idx=0, init=None):
            key, subkey = random.randint(0, 1e6), key
            embeddings, init = self.embedding_module(
                batch=batch,
                prev_pair=prev['pair'],
                prev_single=prev['single'],
                init_pair=init[0],
                init_single=init[1],
                target_feat=target_feat,
                key=subkey,
                idx=idx,
            )
            embeddings['pair'] = embeddings['pair'].astype(ms.float32)
            embeddings['single'] = embeddings['single'].astype(ms.float32)
            return embeddings, init

        num_res = batch.num_tokens  # batch.num_res
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
        init = (None, None)
        for i in range(num_iter):
            embeddings, init = recycle_body(embeddings, key, i, init)

        samples, noise_levels = self._sample_diffusion(
            batch,
            embeddings,
        )
        pred_dict = {}
        pred_dict["coordinate"] = samples['atom_positions'][0]
        confidence_output = (self.confidence_head(
            dense_atom_positions=samples['atom_positions'][0],
            embeddings_single=embeddings['single'],
            embeddings_pair=embeddings['pair'],
            embeddings_target=embeddings['target_feat'],
            seq_mask=None,
            distogram_rep_atom_mask=batch.distogram_rep_atom_mask,
            batch=batch
        ))

        pred_dict["plddt"] = confidence_output['predicted_lddt']
        pred_dict["pae"] = confidence_output['predicted_pae']
        pred_dict["pde"] = confidence_output['predicted_pde']
        pred_dict["resolved"] = confidence_output['predicted_experimentally_resolved']
        breaks, contact_probs, distogram_logits = self.distogram_head(
            embeddings['pair'])
        distogram = {
            'bin_edges': breaks,
            'contact_probs': contact_probs,
            'logits': distogram_logits,
        }
        pred_dict["contact_probs"] = distogram['contact_probs']

        ########### protenix post processing #######
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = sample_confidence.compute_full_data_and_summary(
            configs=self.protenix_config,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=batch.token_features.asym_id,
            token_has_frame=batch.has_frame,
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=batch.atom_to_token_idx,
            atom_is_polymer=1 - batch.is_ligand,
            num_recycle=10,
            interested_atom_mask=None,
            return_full_data=True,
            mol_id=batch.mol_id,
            elements_one_hot=batch.ref_structure.element
        )
        ############################################

        output = {
            'pred_dict': pred_dict,
            'diffusion_samples': samples,
            'noise_level': noise_levels,
            'distogram': distogram,
            **confidence_output,
        }
        if self.config.return_embeddings:
            output['single_embeddings'] = embeddings['single']
            output['pair_embeddings'] = embeddings['pair']

        return output

    def construct(self, data, key):
        if self.training:
            cum_loss, output = self.training_block(data, key)
        else:
            output = self.inference(data, key)
            cum_loss = ms.Tensor(0)
        return cum_loss, output

    def get_loss(self, data, logits, loss_fn, label_dict):
        feat_dict = data
        coord, coord_mini = get_predicted_structure(logits, True)

        logits['coordinate'] = coord
        logits['coordinate_mini'] = coord_mini

        cum_loss, _ = loss_fn(feat_dict, logits, label_dict)
        return cum_loss


def get_predicted_structure(result, with_mini=False):
    """Creates the predicted structure and ion preditions.

    Args:
        result: model output in a model specific layout

    Returns:
        Predicted structure.
    """
    model_output_coords = result['coordinate']
    pred_flat_atom_coords = model_output_coords
    if with_mini:
        model_output_coords_mini = result['coordinates_mini']['atom_positions']
        pred_flat_atom_coords_mini = model_output_coords_mini
        return pred_flat_atom_coords, pred_flat_atom_coords_mini

    return pred_flat_atom_coords

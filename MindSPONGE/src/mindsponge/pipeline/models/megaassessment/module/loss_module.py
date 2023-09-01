# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""loss module"""

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindsponge.metrics.structure_violations import local_distance_difference_test
from .metrics import BalancedMSE, BinaryFocal, MultiClassFocal
# pylint: disable=relative-beyond-top-level
from ...megafold.module.loss_module import LossNet


class LossNetAssessment(nn.Cell):
    """loss net"""

    def __init__(self, config):
        super(LossNetAssessment, self).__init__()
        self.orign_loss = LossNet(config, train_fold=False)
        self.num_bins = config.model.heads.distogram.num_bins
        self.cutoff = 15.0
        self.within_cutoff_clip = 0.3
        self.beyond_cutoff_clip = 3.0
        self.beyond_cutoff_weight = 0.2
        self.regressor_idx = 1
        self.regressor_weight = 2.
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.allreduce = P.AllReduce()
            self.device_num = D.get_group_size()

        self.reg_loss_distogram = RegressionLosses(first_break=2., last_break=22.,
                                                   num_bins=config.model.heads.distogram.num_bins, bin_shift=True,
                                                   charbonnier_eps=0.1, reducer_flag=self.reducer_flag)
        self.reg_loss_lddt = RegressionLosses(first_break=0., last_break=1.,
                                              num_bins=config.model.heads.predicted_lddt.num_bins, bin_shift=False,
                                              charbonnier_eps=1e-5, reducer_flag=self.reducer_flag)

        self.binary_focal_loss = BinaryFocal(alpha=0.25, gamma=1., feed_in=False, not_focal=False)
        self.softmax_focal_loss_lddt = MultiClassFocal(num_class=config.model.heads.predicted_lddt.num_bins,
                                                       gamma=1., e=0.1, neighbors=2, not_focal=False,
                                                       reducer_flag=self.reducer_flag)
        self.softmax_focal_loss_distogram = MultiClassFocal(num_class=config.model.heads.distogram.num_bins,
                                                            gamma=1., e=0.1, neighbors=2, not_focal=False,
                                                            reducer_flag=self.reducer_flag)
        self.cameo_focal_loss = BinaryFocal(alpha=0.2, gamma=0.5, feed_in=True, not_focal=False)
        self.distogram_one_hot = nn.OneHot(depth=self.num_bins, axis=-1)
        self.breaks = np.linspace(2.0, 22.0, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]
        self.centers = ms.Tensor(self.breaks + 0.5 * self.width, ms.float32)

    def distogram_loss(self, logits, bin_edges, pseudo_beta, pseudo_beta_mask):
        """Log loss of a distogram."""
        positions = pseudo_beta
        mask = pseudo_beta_mask

        sq_breaks = mnp.square(bin_edges)
        dist_t = mnp.square(mnp.expand_dims(positions, axis=-2) - mnp.expand_dims(positions, axis=-3))
        dist2 = P.ReduceSum(True)(dist_t.astype(ms.float32), -1)
        aa = (dist2 > sq_breaks).astype(ms.float32)

        square_mask = mnp.expand_dims(mask, axis=-2) * mnp.expand_dims(mask, axis=-1)
        probs = nn.Softmax(-1)(logits)
        dmat_pred = mnp.sum(probs * mnp.reshape(self.centers, (1, 1, -1)), -1)
        dist2 = dist2[..., 0]
        dmat_true = mnp.sqrt(1e-6 + dist2)

        within_cutoff_mask = F.cast(dmat_true < self.cutoff, mnp.float32)
        within_cutoff_mask *= (1. - mnp.eye(within_cutoff_mask.shape[1]))
        beyond_cutoff_mask = F.cast(dmat_true > self.cutoff, mnp.float32)
        beyond_cutoff_mask *= self.beyond_cutoff_weight

        true_bins = P.ReduceSum()(aa, -1)
        true_bins = true_bins.astype(ms.int32)

        nres, nres, nbins = logits.shape
        logits = mnp.reshape(logits, (-1, nbins))
        labels = self.distogram_one_hot(true_bins)
        labels = mnp.reshape(labels, (-1, nbins))

        error = self.softmax_focal_loss_distogram(logits, labels)
        error = mnp.reshape(error, (nres, nres))
        focal_error = within_cutoff_mask * error + beyond_cutoff_mask * error

        focal_loss = (P.ReduceSum()(focal_error * square_mask, (-2, -1)) /
                      (1e-6 + P.ReduceSum()(square_mask.astype(ms.float32), (-2, -1))))

        error_tuple = self.reg_loss_distogram(dmat_pred, dmat_true)
        regression_error = error_tuple[1]

        regression_error_clip_within = mnp.clip(regression_error, self.within_cutoff_clip,
                                                20.) - self.within_cutoff_clip
        regression_error_clip_beyond = mnp.clip(regression_error, self.beyond_cutoff_clip,
                                                20.) - self.beyond_cutoff_clip

        regression_error = regression_error_clip_within * within_cutoff_mask + regression_error_clip_beyond \
                           * beyond_cutoff_mask

        square_mask_off_diagonal = square_mask * (1 - mnp.eye(square_mask.shape[1]))

        regression_loss = (P.ReduceSum()(regression_error * square_mask_off_diagonal, (-2, -1)) /
                           (1e-6 + P.ReduceSum()(square_mask_off_diagonal.astype(ms.float32), (-2, -1))))

        loss = focal_loss + self.regressor_weight * regression_loss

        dist_loss = loss, focal_loss, regression_loss, dmat_true

        return dist_loss

    def construct(self, distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask,
                  atom37_atom_exists, all_atom_mask, true_msa, bert_mask,
                  final_atom14_positions, residue_index, aatype, residx_atom14_to_atom37, lower_bound, upper_bound,
                  seq_mask, atomtype_radius, final_affines, angles_sin_cos,
                  um_angles_sin_cos, backbone_affine_tensor, backbone_affine_mask, atom14_gt_positions,
                  atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_atom_exists,
                  atom14_alt_gt_exists, final_atom_positions, all_atom_positions, predicted_lddt_logits, traj,
                  rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_alt_gt_frames,
                  pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask,
                  decoy_pseudo_beta, decoy_pseudo_beta_mask, decoy_predicted_lddt_logits, plddt_dist, pred_mask2d):
        """construct"""
        _, l_fape_side, l_fape_backbone, l_anglenorm, _, _, predict_lddt_loss = self.orign_loss(
            distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask, None,
            atom37_atom_exists, all_atom_mask, true_msa, None, bert_mask,
            final_atom14_positions, residue_index, aatype, residx_atom14_to_atom37, lower_bound, upper_bound,
            seq_mask, atomtype_radius, final_affines, None, None, angles_sin_cos,
            um_angles_sin_cos, backbone_affine_tensor, backbone_affine_mask, atom14_gt_positions,
            atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists, atom14_atom_exists,
            atom14_alt_gt_exists, final_atom_positions, all_atom_positions, predicted_lddt_logits, traj,
            rigidgroups_gt_frames, rigidgroups_gt_exists, rigidgroups_alt_gt_frames,
            pred_frames, pred_positions, sin_cos_true_chi, torsion_angle_mask, 1.0, 1.0)

        fold_loss = l_fape_side + l_fape_backbone + l_anglenorm + predict_lddt_loss

        lddt_cb = local_distance_difference_test(
            predicted_points=decoy_pseudo_beta[None, ...],
            true_points=pseudo_beta[None, ...],
            true_points_mask=decoy_pseudo_beta_mask[None, ..., None].astype(mnp.float32),
            cutoff=15.,
            per_residue=True)[0]
        lddt_cb = F.stop_gradient(lddt_cb)

        distogram_loss, distogram_focal_loss, distogram_regression_loss, dmat_true = self.distogram_loss(
            distogram_logits, bin_edges, pseudo_beta, pseudo_beta_mask)

        mask1d = decoy_pseudo_beta_mask
        mask2d = mnp.expand_dims(mask1d, 1) * mnp.expand_dims(mask1d, 0)
        error_tuple = self.reg_loss_lddt(plddt_dist, lddt_cb)
        plddt2_error = error_tuple[self.regressor_idx]

        plddt2_regression_loss = mnp.sum(plddt2_error * mask1d) / (mnp.sum(mask1d) + 1e-8)
        plddt2_loss = self.regressor_weight * plddt2_regression_loss

        true_mask2d = P.Cast()(dmat_true < self.cutoff, ms.float32)
        mask_error = self.binary_focal_loss(mnp.reshape(pred_mask2d, (-1,)), mnp.reshape(true_mask2d, (-1,)))
        mask_error = mnp.reshape(mask_error, true_mask2d.shape)
        mask_loss = mnp.sum(mask_error * mask2d) / (mnp.sum(mask2d) + 1e-6)
        confidence_pred = mnp.sum(plddt_dist * mask1d) / (mnp.sum(mask1d) + 1e-6)
        confidence_gt = mnp.sum(lddt_cb * mask1d) / (mnp.sum(mask1d) + 1e-6)
        confidence_loss = nn.MSELoss()(confidence_pred, confidence_gt)
        confidence_loss = mnp.sqrt(confidence_loss + 1e-5)

        cameo_label = F.cast(lddt_cb < 0.6, mnp.float32)
        cameo_scale = decoy_predicted_lddt_logits[:, 0]
        cameo_shift = decoy_predicted_lddt_logits[:, 1]
        cameo_scale = 5. * P.Tanh()(cameo_scale / 5.)
        decoy_cameo_logit = -F.exp(cameo_scale) * (plddt_dist + cameo_shift - 0.6)
        cameo_error = self.cameo_focal_loss(decoy_cameo_logit, cameo_label)
        cameo_loss = mnp.sum(cameo_error * mask1d) / (mnp.sum(mask1d) + 1e-6)

        score_loss = distogram_loss + plddt2_loss + mask_loss + 2.0 * confidence_loss + 2.0 * cameo_loss

        loss = 0.5 * fold_loss + score_loss

        seq_len = F.cast(P.ReduceSum()(pseudo_beta_mask), mnp.float32)
        loss_weight = mnp.power(seq_len, 0.5)
        if self.reducer_flag:
            loss_weight_sum = self.allreduce(loss_weight) / self.device_num
            loss_weight = loss_weight / loss_weight_sum
        loss_weight *= 64.

        loss = loss * loss_weight

        loss_all = loss, l_fape_side, l_fape_backbone, l_anglenorm, predict_lddt_loss, \
                   distogram_focal_loss, distogram_regression_loss, plddt2_regression_loss, mask_loss, \
                   confidence_loss, cameo_loss

        return loss_all


class RegressionLosses(nn.Cell):
    """Return various regressor losses"""

    def __init__(self, first_break, last_break, num_bins, bin_shift=True, beta=0.99, charbonnier_eps=1e-5,
                 reducer_flag=False):
        super(RegressionLosses, self).__init__()

        self.beta = beta
        self.charbonnier_eps = charbonnier_eps

        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins
        self.breaks = np.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]

        bin_width = 2
        start_n = 1
        stop = self.num_bins * 2
        centers = np.divide(np.arange(start=start_n, stop=stop, step=bin_width), self.num_bins * 2.0)
        self.centers = ms.Tensor(centers / (self.last_break - self.first_break) + self.first_break, ms.float32)

        if bin_shift:
            centers = np.linspace(self.first_break, self.last_break, self.num_bins)
            self.centers = ms.Tensor(centers + 0.5 * self.width, ms.float32)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bmse = BalancedMSE(first_break, last_break, num_bins, beta, reducer_flag)

    def construct(self, prediction, target):
        """construct"""
        target = mnp.clip(target, self.centers[0], self.centers[-1])

        mse = self.mse(prediction, target)
        mae = self.mae(prediction, target)
        bmse = self.bmse(prediction, target)
        return [mse, mae, bmse]

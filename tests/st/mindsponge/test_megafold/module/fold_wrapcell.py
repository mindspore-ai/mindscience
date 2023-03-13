# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""wrap cell"""

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.context import ParallelMode
from mindspore.nn import DistributedGradReducer
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_device_num
from mindspore.parallel._utils import (_get_gradients_mean, _get_parallel_mode)
from module.loss_module import LossNet, LossNetAssessment

GRADIENT_CLIP_TYPE = 1

clip_grad = ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """_clip_grad"""
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt),
                                     ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    """tensor_grad_scale"""
    return grad * ops.Reciprocal()(scale)


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""
    def __init__(self, network, optimizer, sens=1.0, enable_clip_grad=True, use_global_norm=True,
                 gradient_clip_value=1.0, train_fold=True, train_assessment=False):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.enable_clip_grad = enable_clip_grad
        self.hyper_map = ops.HyperMap()
        self.use_global_norm = use_global_norm
        self.gradient_clip_value = gradient_clip_value
        self.train_fold = train_fold
        self.train_assessment = train_assessment

        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        """construct"""
        if self.train_backward:
            loss_all = self.network(*inputs)
            grads = None
            if self.train_fold:
                loss, l_fape_side, l_fape_backbone, l_anglenorm, distogram_loss, masked_loss, \
                predict_lddt_loss = loss_all
                sens = F.fill(loss.dtype, loss.shape, self.sens)
                sens1 = F.fill(l_fape_side.dtype, l_fape_side.shape, 0.0)
                sens2 = F.fill(l_fape_backbone.dtype, l_fape_backbone.shape, 0.0)
                sens3 = F.fill(l_anglenorm.dtype, l_anglenorm.shape, 0.0)
                sens4 = F.fill(distogram_loss.dtype, distogram_loss.shape, 0.0)
                sens5 = F.fill(masked_loss.dtype, masked_loss.shape, 0.0)
                sens6 = F.fill(predict_lddt_loss.dtype, predict_lddt_loss.shape, 0.0)
                grads = self.grad(self.network, self.weights)(*inputs, (sens, sens1, sens2, sens3, sens4, sens5, sens6))
            elif self.train_assessment:
                loss, l_fape_side, l_fape_backbone, l_anglenorm, predict_lddt_loss, \
                distogram_focal_loss, distogram_regression_loss, plddt_dist_loss, mask_loss, \
                confidence_loss, cameo_loss = loss_all
                sens = F.fill(loss.dtype, loss.shape, self.sens)
                sens1 = F.fill(l_fape_side.dtype, l_fape_side.shape, 0.0)
                sens2 = F.fill(l_fape_backbone.dtype, l_fape_backbone.shape, 0.0)
                sens3 = F.fill(l_anglenorm.dtype, l_anglenorm.shape, 0.0)
                sens4 = F.fill(predict_lddt_loss.dtype, predict_lddt_loss.shape, 0.0)
                sens5 = F.fill(distogram_focal_loss.dtype, distogram_focal_loss.shape, 0.0)
                sens6 = F.fill(distogram_regression_loss.dtype, distogram_regression_loss.shape, 0.0)
                sens7 = F.fill(plddt_dist_loss.dtype, plddt_dist_loss.shape, 0.0)
                sens8 = F.fill(mask_loss.dtype, mask_loss.shape, 0.0)
                sens9 = F.fill(confidence_loss.dtype, confidence_loss.shape, 0.0)
                sens10 = F.fill(cameo_loss.dtype, cameo_loss.shape, 0.0)
                grads = self.grad(self.network, self.weights)(*inputs, (sens, sens1, sens2, sens3, sens4,
                                                                        sens5, sens6, sens7, sens8, sens9, sens10))
            grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_tensor(self.sens)), grads)
            grads = self.grad_reducer(grads)
            if self.enable_clip_grad:
                if self.use_global_norm:
                    grads = C.clip_by_global_norm(grads, self.gradient_clip_value)
                else:
                    grads = self.hyper_map(ops.partial(clip_grad, GRADIENT_CLIP_TYPE, self.gradient_clip_value), grads)

            loss_all = F.depend(loss_all, self.optimizer(grads))
            return loss_all

        out = self.network(*inputs)
        return out


class WithLossCell(nn.Cell):
    """WithLossCell"""
    def __init__(self, backbone, config):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.loss_net = LossNet(config).to_float(mstype.float32)

    def construct(self, target_feat, msa_feat, msa_mask, seq_mask, aatype,
                  template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion,
                  extra_deletion_value, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, residue_index,
                  prev_pos, prev_msa_first_row, prev_pair, pseudo_beta_gt, pseudo_beta_mask_gt,
                  all_atom_mask_gt, true_msa, bert_mask,
                  residx_atom14_to_atom37, restype_atom14_bond_lower_bound, restype_atom14_bond_upper_bound,
                  atomtype_radius, backbone_affine_tensor, backbone_affine_mask,
                  atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists,
                  atom14_atom_exists, atom14_alt_gt_exists, all_atom_positions, rigidgroups_gt_frames,
                  rigidgroups_gt_exists, rigidgroups_alt_gt_frames, torsion_angles_sin_cos_gt, use_clamped_fape,
                  filter_by_solution, chi_mask):
        """construct"""
        if self.train_backward:
            dist_logits, bin_edges, experimentally_logits, masked_logits, aligned_error_logits, aligned_error_breaks, \
            atom14_pred_positions, final_affines, angles_sin_cos_new, predicted_lddt_logits, structure_traj, \
            sidechain_frames, sidechain_atom_pos, um_angles_sin_cos_new, final_atom_positions = \
                self._backbone(target_feat, msa_feat, msa_mask, seq_mask, aatype, template_aatype,
                               template_all_atom_masks, template_all_atom_positions, template_mask,
                               template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion,
                               extra_deletion_value, extra_msa_mask, residx_atom37_to_atom14, atom37_atom_exists,
                               residue_index, prev_pos, prev_msa_first_row, prev_pair)
            out = self.loss_net(dist_logits, bin_edges, pseudo_beta_gt, pseudo_beta_mask_gt,
                                experimentally_logits, atom37_atom_exists, all_atom_mask_gt, true_msa,
                                masked_logits, bert_mask, atom14_pred_positions, residue_index, aatype,
                                residx_atom14_to_atom37, restype_atom14_bond_lower_bound,
                                restype_atom14_bond_upper_bound, seq_mask, atomtype_radius, final_affines,
                                aligned_error_breaks, aligned_error_logits, angles_sin_cos_new,
                                um_angles_sin_cos_new, backbone_affine_tensor, backbone_affine_mask,
                                atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                                atom14_gt_exists, atom14_atom_exists, atom14_alt_gt_exists,
                                final_atom_positions, all_atom_positions, predicted_lddt_logits,
                                structure_traj, rigidgroups_gt_frames, rigidgroups_gt_exists,
                                rigidgroups_alt_gt_frames,
                                sidechain_frames, sidechain_atom_pos, torsion_angles_sin_cos_gt,
                                chi_mask, use_clamped_fape, filter_by_solution)
        else:
            out = self._backbone(target_feat, msa_feat, msa_mask, seq_mask, aatype, template_aatype,
                                 template_all_atom_masks, template_all_atom_positions, template_mask,
                                 template_pseudo_beta_mask, template_pseudo_beta, extra_msa,
                                 extra_has_deletion, extra_deletion_value, extra_msa_mask, residx_atom37_to_atom14,
                                 atom37_atom_exists, residue_index, prev_pos, prev_msa_first_row, prev_pair)
        return out


class WithLossCellAssessment(nn.Cell):
    """WithLossCell"""
    def __init__(self, backbone, config):
        super(WithLossCellAssessment, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.loss_net = LossNetAssessment(config).to_float(mstype.float32)

    def construct(self, target_feat, msa_feat, msa_mask, seq_mask, aatype,
                  template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion,
                  extra_deletion_value, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, residue_index,
                  prev_pos, prev_msa_first_row, prev_pair, pseudo_beta_gt, pseudo_beta_mask_gt,
                  all_atom_mask_gt, true_msa, bert_mask,
                  residx_atom14_to_atom37, restype_atom14_bond_lower_bound, restype_atom14_bond_upper_bound,
                  atomtype_radius, backbone_affine_tensor, backbone_affine_mask,
                  atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous, atom14_gt_exists,
                  atom14_atom_exists, atom14_alt_gt_exists, all_atom_positions, rigidgroups_gt_frames,
                  rigidgroups_gt_exists, rigidgroups_alt_gt_frames, torsion_angles_sin_cos_gt, chi_mask,
                  final_atom_positions_recycle, final_atom_mask_recycle):
        """construct"""
        dist_logits, bin_edges, atom14_pred_positions, final_affines, angles_sin_cos_new, \
        predicted_lddt_logits, structure_traj, sidechain_frames, sidechain_atom_pos, \
        um_angles_sin_cos_new, final_atom_positions, decoy_pseudo_beta, \
        decoy_pseudo_beta_mask, decoy_logits, plddt_dist, pred_mask2d = \
            self._backbone(target_feat, msa_feat, msa_mask, seq_mask, aatype, template_aatype,
                           template_all_atom_masks, template_all_atom_positions, template_mask,
                           template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion,
                           extra_deletion_value, extra_msa_mask, residx_atom37_to_atom14, atom37_atom_exists,
                           residue_index, prev_pos, prev_msa_first_row, prev_pair, final_atom_positions_recycle,
                           final_atom_mask_recycle)
        out = self.loss_net(dist_logits, bin_edges, pseudo_beta_gt, pseudo_beta_mask_gt,
                            atom37_atom_exists, all_atom_mask_gt, true_msa,
                            bert_mask, atom14_pred_positions, residue_index, aatype,
                            residx_atom14_to_atom37, restype_atom14_bond_lower_bound,
                            restype_atom14_bond_upper_bound, seq_mask, atomtype_radius, final_affines,
                            angles_sin_cos_new, um_angles_sin_cos_new, backbone_affine_tensor, backbone_affine_mask,
                            atom14_gt_positions, atom14_alt_gt_positions, atom14_atom_is_ambiguous,
                            atom14_gt_exists, atom14_atom_exists, atom14_alt_gt_exists,
                            final_atom_positions, all_atom_positions, predicted_lddt_logits,
                            structure_traj, rigidgroups_gt_frames, rigidgroups_gt_exists,
                            rigidgroups_alt_gt_frames,
                            sidechain_frames, sidechain_atom_pos, torsion_angles_sin_cos_gt,
                            chi_mask, decoy_pseudo_beta, decoy_pseudo_beta_mask, decoy_logits,
                            plddt_dist, pred_mask2d)
        return out

# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
Loss Function
"""

import mindspore as ms
from mindspore import nn
from mindspore import ops
import numpy as np
import ot

from .. import core
from ..core import Registry as R
from .. import util


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))


def g_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = ops.exp(-((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2).sum(axis=2) / float(sigma))  # (m, n)
    return - sigma * ops.log(1e-3 + e.sum(axis=1))


def body_intersection_loss(lig_coord, rec_coord, sigma, surface_ct):
    loss = ops.mean(util.clip(surface_ct - g_fn(rec_coord, lig_coord, sigma), xmin=0)) + \
        ops.mean(util.clip(surface_ct - g_fn(lig_coord, rec_coord, sigma), xmin=0))
    return loss


def calc_sq_dist_mat(x1, x2):
    '''Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    '''
    n_1, _ = x1.shape
    n_2, _ = x2.shape
    x1 = x1.view(n_1, 1, -1)
    x2 = x2.view(1, n_2, -1)
    squared_dist = (x1 - x2) ** 2
    cost_mat = squared_dist.sum(axis=2)
    return cost_mat


def calc_ot_emd(cost_mat):
    """_summary_

    Args:
        cost_mat (_type_): _description_

    Returns:
        _type_: _description_
    """
    cost_mat_detach = cost_mat.asnumpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach)
    ot_mat_attached = ms.Tensor.from_numpy(ot_mat)
    ot_dist = (ot_mat_attached * cost_mat).sum()
    return ot_dist


def revised_intersection_loss(lig_coords, rec_coords, alpha=0.2, beta=8, aggression=0):
    """_summary_

    Args:
        lig_coords (_type_): _description_
        rec_coords (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.2.
        beta (int, optional): _description_. Defaults to 8.
        aggression (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    distances = calc_sq_dist_mat(lig_coords, rec_coords)
    if aggression > 0:
        aggression_term = ops.clip(-ops.log(ops.sqrt(distances) / aggression+0.01), min=1)
    else:
        aggression_term = 1
    distance_losses = aggression_term * \
        ops.exp(-alpha*distances * ops.clip(distances*4-beta, min=1))
    return distance_losses.sum()


@R.register('loss.BindingLoss')
class BindingLoss(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """
    def __init__(self,
                 ot_loss_weight=1,
                 intersection_loss_weight=0,
                 intersection_sigma=0,
                 geom_reg_loss_weight=1,
                 loss_rescale=True,
                 intersection_surface_ct=0,
                 key_point_alignmen_loss_weight=0,
                 revised_intersection_loss_weight=0,
                 centroid_loss_weight=0,
                 kabsch_rmsd_weight=0,
                 translated_lig_kpt_ot_loss=False,
                 revised_intersection_alpha=0.1,
                 revised_intersection_beta=8,
                 aggression=0) -> None:
        super().__init__()
        self.ot_loss_weight = ot_loss_weight
        self.intersection_loss_weight = intersection_loss_weight
        self.intersection_sigma = intersection_sigma
        self.revised_intersection_loss_weight = revised_intersection_loss_weight
        self.intersection_surface_ct = intersection_surface_ct
        self.key_point_alignmen_loss_weight = key_point_alignmen_loss_weight
        self.centroid_loss_weight = centroid_loss_weight
        self.translated_lig_kpt_ot_loss = translated_lig_kpt_ot_loss
        self.kabsch_rmsd_weight = kabsch_rmsd_weight
        self.revised_intersection_alpha = revised_intersection_alpha
        self.revised_intersection_beta = revised_intersection_beta
        self.aggression = aggression
        self.loss_rescale = loss_rescale
        self.geom_reg_loss_weight = geom_reg_loss_weight
        self.mse_loss = nn.MSELoss()

    def construct(self,
                  ligs,
                  recs,
                  ligs_coords_,
                  pockets,
                  new_pockets,
                  ligs_keypts,
                  recs_keypts,
                  rotations,
                  translations,
                  geom_reg_loss):
        """_summary_

        Args:
            ligs (_type_): _description_
            recs (_type_): _description_
            ligs_coords_ (_type_): _description_
            pockets (_type_): _description_
            new_pockets (_type_): _description_
            ligs_keypts (_type_): _description_
            recs_keypts (_type_): _description_
            rotations (_type_): _description_
            translations (_type_): _description_
            geom_reg_loss (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Compute MSE loss for each protein individually, then average over the minibatch.
        ligs = ligs.unpack()
        recs = recs.unpack()
        ligs_coords_loss = 0
        recs_coords_loss = 0
        ot_loss = 0
        intersection_loss = 0
        intersection_loss_revised = 0
        keypts_loss = 0
        centroid_loss = 0
        kabsch_rmsd_loss = 0

        for i, lig_coords_ in enumerate(ligs_coords_):
            # Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            ligs_coords_loss += self.mse_loss(lig_coords_, ligs[i].coord)

            if self.ot_loss_weight > 0:
                # Compute the OT loss for the binding pocket:
                # (N, 3), N = num pocket nodes
                ligand_pocket = new_pockets[i]
                # (N, 3), N = num pocket nodes
                receptor_pocket = pockets[i]

                # (N, K) cost matrix
                if self.translated_lig_kpt_ot_loss:
                    lig_keypt = ligs_keypts[i] @ rotations[i].T + translations[i]
                    cost_mat_ligand = calc_sq_dist_mat(
                        receptor_pocket, lig_keypt)
                else:
                    cost_mat_ligand = calc_sq_dist_mat(
                        ligand_pocket, ligs_keypts[i])
                cost_mat_receptor = calc_sq_dist_mat(
                    receptor_pocket, recs_keypts[i])

                ot_dist = calc_ot_emd(cost_mat_ligand + cost_mat_receptor)
                ot_loss += ot_dist

            if self.key_point_alignmen_loss_weight > 0:
                keypts_loss += self.mse_loss(ligs_keypts[i] @ rotations[i].T + translations[i], recs_keypts[i])

            if self.intersection_loss_weight > 0:
                intersection_loss += body_intersection_loss(
                    lig_coords_, recs[i].coord, self.intersection_sigma, self.intersection_surface_ct
                )

            if self.revised_intersection_loss_weight > 0:
                intersection_loss_revised += revised_intersection_loss(
                    lig_coords_,
                    recs[i].coord,
                    alpha=self.revised_intersection_alpha,
                    beta=self.revised_intersection_beta,
                    aggression=self.aggression
                )

            if self.kabsch_rmsd_weight > 0:
                lig_coords_pred = ligs_coords_[i]
                lig_coords = ligs[i].coord
                lig_coords_pred_mean = lig_coords_pred.mean(axis=0, keep_dims=True)  # (1,3)
                lig_coords_mean = lig_coords.mean(axis=0, keep_dims=True)  # (1,3)

                mat_a = (lig_coords_pred - lig_coords_pred_mean).T @ (lig_coords - lig_coords_mean)

                _, ut, vt = ops.svd(mat_a)

                corr_mat = ops.diag(ms.Tensor([1, 1, np.sign(mat_a.det().asnumpy())]))
                rotation = (ut @ corr_mat) @ vt
                translation = lig_coords_pred_mean - (rotation @ lig_coords_mean.T).T  # (1,3)
                kabsch_rmsd_loss += self.mse_loss((rotation @ lig_coords.T).T + translation, lig_coords_pred)

            centroid_loss += self.mse_loss(lig_coords_.mean(axis=0), ligs[i].coord.mean(axis=0))

        if self.loss_rescale:
            ligs_coords_loss /= float(len(ligs_coords_))
            ot_loss /= float(len(ligs_coords_))
            intersection_loss /= float(len(ligs_coords_))
            keypts_loss /= float(len(ligs_coords_))
            centroid_loss /= float(len(ligs_coords_))
            kabsch_rmsd_loss /= float(len(ligs_coords_))
            intersection_loss_revised /= float(len(ligs_coords_))
            geom_reg_loss /= float(len(ligs_coords_))

        loss = ligs_coords_loss + self.ot_loss_weight * ot_loss + \
            self.intersection_loss_weight * intersection_loss + \
            keypts_loss * self.key_point_alignmen_loss_weight + \
            centroid_loss * self.centroid_loss_weight + \
            kabsch_rmsd_loss * self.kabsch_rmsd_weight + \
            intersection_loss_revised * self.revised_intersection_loss_weight + \
            geom_reg_loss*self.geom_reg_loss_weight
        return loss, {'ligs_coords_loss': ligs_coords_loss, 'recs_coords_loss': recs_coords_loss, 'ot_loss': ot_loss,
                      'intersection_loss': intersection_loss, 'keypts_loss': keypts_loss,
                      'centroid_loss:': centroid_loss, 'kabsch_rmsd_loss': kabsch_rmsd_loss,
                      'intersection_loss_revised': intersection_loss_revised, 'geom_reg_loss': geom_reg_loss}


class TorsionLoss(core.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def construct(self, angles_pred, angles, masks, **kwargs):
        return self.mse_loss(angles_pred*masks, angles*masks, **kwargs)

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
# ============================================================================
"""design loss"""
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
import mindsponge.common.residue_constants as residue_constants
from mindsponge.common.utils import pseudo_beta_fn


class SVD(nn.Cell):
    """SVD"""

    def __init__(self):
        super(SVD, self).__init__()
        self.matmul_a = P.MatMul(transpose_a=True)

    def qr_split(self, ori_matrix):
        """QR Split"""
        shapes = ori_matrix.shape[0]
        quadrature_matrix = [[]]
        res = ori_matrix
        for i in range(0, shapes - 1):
            batch = res
            if i != 0:
                batch = batch[i:, i:]
            x = batch[:, 0]
            m = mnp.norm(x)
            y = [0 for j in range(0, shapes - i)]
            y[0] = m
            w = x - y
            w = w / mnp.norm(w)
            h = mnp.eye(shapes - i) - 2 * P.MatMul()(w.reshape(shapes - i, 1), w.reshape(1, shapes - i))
            if i == 0:
                quadrature_matrix = h
                res = P.MatMul()(h, res)
            else:
                dim = mnp.concatenate((mnp.eye(i), mnp.zeros((i, shapes - i))), axis=1)
                h = mnp.concatenate((mnp.zeros((shapes - i, i)), h), axis=1)
                h = mnp.concatenate((dim, h), axis=0)
                quadrature_matrix = P.MatMul()(h, quadrature_matrix)
                res = P.MatMul()(h, res)
        quadrature_matrix = quadrature_matrix.T
        return [quadrature_matrix, res]

    def qr_egis(self, ori_matrix):
        """QR egis"""
        qr = []
        shapes = ori_matrix.shape[0]
        quadrature_matrix = mnp.eye(shapes)
        for i in range(0, 100):
            qr = self.qr_split(ori_matrix)
            quadrature_matrix = P.MatMul()(quadrature_matrix, qr[0])
            ori_matrix = P.MatMul()(qr[1], qr[0])

        ak = P.MatMul()(qr[0], qr[1])
        e = P.Ones()((3, 1), mstype.float32)
        for i in range(0, shapes):
            e[i] = ak[i][i]
        return e, quadrature_matrix

    def rebuild_matrix(self, u, sigma, v):
        """rebuild matrix"""
        a = P.MatMul()(u, sigma)
        a = P.MatMul()(a, np.transpose(v))
        return a

    def sort_eigenvalue(self, eigenvalues, eigenvectors):
        """sort_eigenvalue"""
        _, index = P.Sort(axis=0)(-1 * eigenvalues)
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]
        return eigenvalues, eigenvectors

    def svd(self, matrixa, numofleft=None):
        """Singular value decomposition of a matrix"""
        matrixat_matrixa = self.matmul_a(matrixa, matrixa)
        lambda_v, x_v = self.qr_egis(matrixat_matrixa)
        lambda_v, x_v = self.sort_eigenvalue(lambda_v, x_v)
        sigmas = lambda_v

        sigmas_new = mnp.where(sigmas > 0, sigmas, 0)
        sigmas = P.Sqrt()(sigmas_new)

        sigmas = mnp.concatenate(sigmas)
        sigmasmatrix = mnp.diag(sigmas[:, 0])
        if numofleft is None:
            rankofsigmasmatrix = 3
        else:
            rankofsigmasmatrix = numofleft
        sigmasmatrix = sigmasmatrix[0:rankofsigmasmatrix, :]

        x_u = mnp.zeros((matrixa.shape[0], rankofsigmasmatrix))
        for i in range(rankofsigmasmatrix):
            x_u[:, i] = (P.MatMul()(matrixa, x_v[:, i]) / sigmas[i])[:, 0]

        x_v = mnp.squeeze(x_v[:, 0:numofleft])
        sigmasmatrix = sigmasmatrix[0:rankofsigmasmatrix, 0:rankofsigmasmatrix]
        return x_u, mnp.diag(sigmasmatrix), x_v


class LossNet(nn.Cell):
    "loss net"

    def __init__(self, design_cfg, protocol):
        super(LossNet, self).__init__()
        self.mul = P.Mul()
        self.expand_dims = P.ExpandDims()
        self.batch_matmul = P.BatchMatMul()
        self.matmul_a = P.MatMul(transpose_a=True)
        self.matmul = P.MatMul()
        self.svd = SVD()
        if protocol == 'fixbb':
            loss_weights = design_cfg.fixbb
        elif protocol == 'hallucination':
            loss_weights = design_cfg.hallu
        self.con_weights = loss_weights.con
        self.plddt_weights = loss_weights.plddt
        self.rmsd_weights = loss_weights.rmsd
        self.seq_weights = loss_weights.seq
        self.dgram_weights = loss_weights.dgram
        self.fape_weights = loss_weights.fape
        self.pae_weights = loss_weights.pae
        self.exp_weights = loss_weights.exp
        self.rg_weights = loss_weights.rg

    def get_fape_loss(self, true_all_atom_positions, true_all_atom_mask, final_atom_positions, clamp=10.0,
                      return_mtx=False):
        "fape loss"

        def robust_norm(x, axis=-1, keepdims=False, eps=1e-8):
            return P.Sqrt()(P.Square()(x).sum(axis=axis, keepdims=keepdims) + eps)

        def get_r(n, ca, cinput):
            (v1, v2) = (cinput - ca, n - ca)
            e1 = v1 / robust_norm(v1, axis=-1, keepdims=True)
            c1 = self.mul(e1, v2).sum(axis=1)
            c = self.expand_dims(c1, 1)
            e2 = v2 - c * e1
            e2 = e2 / robust_norm(e2, axis=-1, keepdims=True)
            e3 = mnp.cross(e1, e2, axis=-1)
            e1 = self.expand_dims(e1, 2)
            e2 = self.expand_dims(e2, 2)
            e3 = self.expand_dims(e3, 2)
            return mnp.concatenate([e1, e2, e3], axis=-1)

        def get_ij(r, t):
            t = self.expand_dims(t, 0) - self.expand_dims(t, 1)
            return self.batch_matmul(t, r)

        def loss_fn(t, p, m):
            fape = robust_norm(t - p)
            fape = mnp.clip(fape, 0, clamp) / 10.0
            return fape, (fape * m).sum((-1, -2)) / (m.sum((-1, -2)) + 1e-8)

        true = true_all_atom_positions
        pred = final_atom_positions

        n, ca, cinput = (residue_constants.atom_order[k] for k in ["N", "CA", "C"])

        true_mask = true_all_atom_mask
        weights = true_mask[:, n] * true_mask[:, ca] * true_mask[:, cinput]

        true = get_ij(get_r(true[:, n], true[:, ca], true[:, cinput]), true[:, ca])
        pred = get_ij(get_r(pred[:, n], pred[:, ca], pred[:, cinput]), pred[:, ca])

        return self._get_pw_loss(true, pred, loss_fn, weights=weights, return_mtx=return_mtx)

    def get_rmsd_loss(self, true_all_atom_positions, true_all_atom_mask, true_final_atom_positions):
        """rmsd loss"""
        true = true_all_atom_positions[:, 1]
        pred = true_final_atom_positions[:, 1]
        weights = true_all_atom_mask[:, 1]
        return self._get_rmsd_loss(true, pred, weights=weights)

    def get_dgram_loss(self, batch_aatype, batch_all_atom, batch_all_atom_mask, dist_logits, aatype=None,
                       return_mtx=False):
        """dgram_loss"""

        if aatype is None:
            aatype = batch_aatype

        pred = dist_logits
        x, weights = pseudo_beta_fn(aatype=aatype,
                                    all_atom_positions=batch_all_atom,
                                    all_atom_masks=batch_all_atom_mask)
        #
        dm = mnp.square(x[:, None] - x[None, :]).sum(-1, keepdims=True).astype(ms.float32)
        bin_edges = mnp.linspace(2.3125, 21.6875, pred.shape[-1] - 1)
        hot_value = (dm > mnp.square(bin_edges)).astype(ms.float32)
        hot_value = hot_value.sum(-1).astype(ms.int32)
        one_hot = nn.OneHot(depth=pred.shape[-1])
        true_label = one_hot(hot_value).astype(ms.float32)

        def loss_fn(t, p, m):
            cce = -(t * ms.ops.log_softmax(p)).sum(-1)
            return cce, (cce * m).sum((-1, -2)) / (m.sum((-1, -2)) + 1e-8)

        return self._get_pw_loss(true_label, pred, loss_fn, weights=weights, return_mtx=return_mtx)

    def get_seq_ent_loss(self, inputs):
        """seq_ent loss"""
        softmax = ms.nn.Softmax()
        x = inputs / mnp.array(1.)
        ent = -(softmax(x) * ms.ops.log_softmax(x)).sum(-1)
        mask = mnp.ones(ent.shape[-1])

        ent = (ent * mask).sum() / (mask.sum() + 1e-8)
        return ent.mean()

    def mask_loss(self, x, mask=None, mask_grad=False):
        """mask_loss"""
        if mask is None:
            result = x.mean()
        else:
            x_masked = (x * mask).sum() / (1e-8 + mask.sum())
            if mask_grad:
                result = ms.ops.stop_gradient(x.mean() - x_masked) + x_masked
            else:
                result = x_masked
        return result

    def get_exp_res_loss(self, outputs, mask_1d=None):
        """exp_res loss"""

        sigmoid = ms.nn.Sigmoid()
        p = sigmoid(outputs)
        p = 1 - p[..., residue_constants.atom_order["CA"]]
        return self.mask_loss(p, mask_1d)

    def get_plddt_loss(self, outputs, mask_1d=None):
        """plddt loss"""
        softmax = ms.nn.Softmax()
        p = softmax(outputs)
        op = ops.ReverseV2(axis=[-1])
        p = (p * op(mnp.arange(p.shape[-1]))).mean(-1)

        return self.mask_loss(p, mask_1d)

    def get_pae_loss(self, outputs, mask_1d=None, mask_1b=None, mask_2d=None):
        """pae loss"""
        # aligned error logits
        softmax = ms.nn.Softmax()
        p = softmax(outputs)
        p = (p * mnp.arange(p.shape[-1])).mean(-1)
        p = (p + p.T) / 2
        leng = p.shape[0]
        if mask_1d is None:
            mask_1d = mnp.ones(leng)
        if mask_1b is None:
            mask_1b = mnp.ones(leng)
        if mask_2d is None:
            mask_2d = mnp.ones((leng, leng))
        mask_2d = mask_2d * mask_1d[:, None] * mask_1b[None, :]
        return self.mask_loss(p, mask_2d)

    def get_con_loss(self, residue_index, loss_dgram_logits, loss_dgram_bin,
                     mask_1d=None, mask_1b=None, mask_2d=None):
        """con loss"""

        # get top k
        def min_k(x, k=1, mask=None):
            sort = ops.Sort()
            y = sort(x if mask is None else mnp.where(mask, x, Tensor(65504, dtype=ms.float32)))[0].astype(ms.float32)
            nan_mask = mnp.where(y != Tensor(65504, dtype=ms.float32), False, True)
            k_mask = mnp.logical_and(mnp.arange(y.shape[-1]) < k, nan_mask == Tensor(False)).astype(ms.float32)
            return mnp.where(k_mask, y, Tensor(0)).sum(-1) / (k_mask.sum(-1) + 1e-8)

        def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=True):
            """dgram to contacts"""
            if cutoff is None:
                cutoff = dgram_bins[-1]
            softmax = ms.nn.Softmax()
            bins = dgram_bins < cutoff
            px = softmax(dgram)
            px_ = softmax(dgram - 1e7 * (1 - bins))
            # binary/cateogorical cross-entropy
            con_loss_cat_ent = -(px_ * ms.ops.log_softmax(dgram)).sum(-1)
            con_loss_bin_ent = -mnp.log((bins * px + 1e-8).sum(-1))
            return mnp.where(binary, con_loss_bin_ent, con_loss_cat_ent)

        idx = residue_index.flatten()
        offset = idx[:, None] - idx[None, :]
        # # # define distogram
        dgram = loss_dgram_logits
        dgram_bins = mnp.append(Tensor(0), loss_dgram_bin)
        p = _get_con_loss(dgram, dgram_bins, cutoff=mnp.array(14.), binary=mnp.array(False))

        m = mnp.abs(offset) >= mnp.array(9)

        if mask_1d is None:
            mask_1d = mnp.ones(m.shape[0], dtype=bool)
        if mask_1b is None:
            mask_1b = mnp.ones(m.shape[0], dtype=bool)
        #
        if mask_2d is None:
            m = mnp.logical_and(m, mnp.array(mask_1b))
        else:
            m = mnp.logical_and(m, mnp.array(mask_2d))

        p = min_k(p, mnp.array(2), m)

        return min_k(p, mnp.array(mnp.inf), mask_1d)

    def rg_loss(self, final_atom_positions):
        positions = final_atom_positions
        ca = positions[:, residue_constants.atom_order["CA"]]
        center = ca.mean(0)
        rg = mnp.sqrt(mnp.square(ca - center).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        rg = ms.nn.ELU()(rg - rg_th)
        return rg

    def construct(self, true_aatype, true_all_atom_positions, true_all_atom_mask, true_final_atom_positions,
                  ori_seq_len, dist_logits, bin_edges, experimentally_logits, predicted_lddt_logits,
                  aligned_error_logits, residue_index, seq_logits):
        """construct"""
        mask_1d = mnp.ones((ori_seq_len,))
        mask_2d = (mask_1d[:, None] == mask_1d[None, :])
        masks = {"mask_1d": mask_1d,
                 "mask_2d": mask_2d}
        fape_loss = self.get_fape_loss(true_all_atom_positions[:ori_seq_len, :, :], true_all_atom_mask[:ori_seq_len, :],
                                       true_final_atom_positions[:ori_seq_len, :, :])
        dgram_cce = self.get_dgram_loss(true_aatype[:ori_seq_len], true_all_atom_positions[:ori_seq_len, :, :],
                                        true_all_atom_mask[:ori_seq_len, :], dist_logits[:ori_seq_len, :ori_seq_len, :])
        exp_res = self.get_exp_res_loss(experimentally_logits[:ori_seq_len, :], mask_1d=mask_1d)
        plddt = self.get_plddt_loss(predicted_lddt_logits[:ori_seq_len, :], mask_1d=mask_1d)
        pae = self.get_pae_loss(aligned_error_logits[:ori_seq_len, :ori_seq_len, :], **masks)
        con = self.get_con_loss(residue_index[:ori_seq_len], dist_logits[:ori_seq_len, :ori_seq_len, :], bin_edges,
                                **masks)
        rg_loss = self.rg_loss(true_final_atom_positions)
        seq_loss = self.get_seq_ent_loss(seq_logits[:, :ori_seq_len, :])
        rmsd_loss = fape_loss
        if self.rmsd_weights:
            rmsd_loss = self.get_rmsd_loss(true_all_atom_positions[:ori_seq_len, :, :],
                                           true_all_atom_mask[:ori_seq_len, :],
                                           true_final_atom_positions[:ori_seq_len, :, :])

        loss_all = con * self.con_weights + exp_res * self.exp_weights + self.plddt_weights * plddt + \
                   self.seq_weights * seq_loss + self.pae_weights * pae + fape_loss * self.fape_weights + \
                   self.dgram_weights * dgram_cce + rmsd_loss * self.rmsd_weights + rg_loss * self.rg_weights
        return loss_all

    def _get_rmsd_loss(self, true, pred, weights=None):
        """
        get rmsd + alignment function
        align based on the first L positions, computed weighted rmsd using all
        positions (if include_l=True) or remaining positions (if include_l=False).
        """
        # normalize weights
        length = true.shape[-2]
        if weights is None:
            weights = (mnp.ones(length) / length)[..., None]
        else:
            weights = (weights / (weights.sum(-1, keepdims=True) + 1e-8))[..., None]

        (t_fixbb, p_fixbb, w_fixbb) = (true, pred, weights)

        (t_mu, p_mu) = ((x * w_fixbb).sum(-2, keepdims=True) / w_fixbb.sum((-1, -2)) for x in (t_fixbb, p_fixbb))
        aln = self._np_kabsch((p_fixbb - p_mu) * w_fixbb, t_fixbb - t_mu)

        align_value = P.MatMul()(pred - p_mu, aln) + t_mu
        msd_scalar = (weights * mnp.square(align_value - true)).sum((-1, -2))
        rmsd = P.Sqrt()(msd_scalar + 1e-8)

        return rmsd

    def _np_kabsch(self, a, b):
        """get alignment matrix for two sets of coordinates"""
        ab = self.matmul_a(a, b)

        u, _, vh = self.svd.svd(ab)
        flip = self._det(self.matmul(u, vh)) < 0
        u_ = mnp.where(flip, -u[..., -1].T, u[..., -1].T).T
        u[..., -1] = u_
        return self.matmul(u, vh)

    def _det(self, matrix):
        """det"""
        # matrix dim=3
        result = matrix[0, 0] * matrix[1, 1] * matrix[2, 2] + matrix[0, 1] * matrix[1, 2] * matrix[2, 0] + \
                 matrix[0, 2] * matrix[1, 0] * matrix[2, 1] - matrix[0, 2] * matrix[1, 1] * matrix[2, 0] - \
                 matrix[0, 1] * matrix[1, 0] * matrix[2, 2] - matrix[0, 0] * matrix[1, 2] * matrix[2, 1]
        return result

    def _get_pw_loss(self, true, pred, loss_fn, weights=None, return_mtx=False):
        """get pw loss"""

        expand_dims = ops.ExpandDims()
        fs = {"t": true, "p": pred, "m": expand_dims(weights, 1) * expand_dims(weights, 0)}

        mtx, loss = loss_fn(**fs)
        return mtx if return_mtx else loss

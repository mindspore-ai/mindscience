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
Common metrics
"""
from typing import List

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.train import Metric

from . import sascorer
from .. import util
from ..util import functional
from ..core import Registry as R
from ..util import scatter_add, scatter_mean, scatter_max


@R.register('metric.accuracy')
class Accuracy(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self, eval_type='binary'):
        super().__init__()
        self._type = eval_type
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    def update(self, y_pred, y):
        """update"""
        class_num = y_pred.shape[1] if y_pred.ndim > 1 else 1
        if self._class_num == 0:
            self._class_num = class_num
        elif class_num != self._class_num:
            raise ValueError(f"For 'Accuracy.update', class number not match, last input predicted data "
                             f"contain {self._class_num} classes, but current predicted data contain "
                             f"{y_pred.shape[1]} classes, please check your predicted value(inputs[0]).")

        if self._type == 'classification':
            indices = y_pred.argmax(axis=1)
            result = (np.equal(indices, y) * 1).reshape(-1)
        elif self._type == 'multilabel':
            dimension_index = y_pred.ndim - 1
            y_pred = y_pred.swapaxes(1, dimension_index).reshape(-1, self._class_num)
            y = y.swapaxes(1, dimension_index).reshape(-1, self._class_num)
            result = np.equal(y_pred, y).all(axis=1) * 1
        elif self._type == 'binary':
            indices = (y_pred > 0.5).astype(ms.float32)
            result = (np.equal(indices, y) * 1).reshape(-1)
        self._correct_num += result.sum()
        self._total_num += result.shape[0]

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            np.float64, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._total_num == 0:
            raise RuntimeError("The 'Accuracy' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        return self._correct_num / self._total_num


@R.register('auc')
class AUC(Metric):
    """Area under receiver operating characteristic curve (ROC).
    Args:
        Metric (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self._target = []

    def update(self, pred, target):
        """
        Parameters:
            pred (Tensor): predictions of shape :math:`(n,)`
            target (Tensor): binary targets of shape :math:`(n,)`
        """
        order = pred.argsort(descending=True)
        target = target[order]
        self._target = np.hstack([self._target, target])

    def eval(self):
        hit = self._target.cumsum(0)
        all_val = (self._target == 0).sum() * (self._target == 1).sum()
        auroc = hit[self._target == 0].sum() / (all_val + 1e-10)
        return auroc

    def clear(self):
        self._target = []


@R.register('auprc')
def auprc(pred, target):
    """
    Area under precision-recall curve (PRC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True)
    target = target[order]
    precision = np.cumsum(target, 0) / np.arange(1, len(target) + 1)
    auprc_val = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc_val


@R.register('r2')
def r2(pred, target):
    """
    :math:`R^2` regression score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): targets of shape :math:`(n,)`
    """
    total = np.var(target, unbiased=False)
    residual = ops.mse_loss(pred, target)
    return 1 - residual / total


@R.register('logp')
def logp(pred):
    """
    Logarithm of partition coefficient between octanol and water for a compound.

    Parameters:
        pred (MoleculeBatch): molecules to evaluate
    """
    logps = []
    for mol in pred:
        mol = mol.to_molecule()
        try:
            with util.no_rdkit_log():
                mol.UpdatePropertyCache()
                score = Descriptors.MolLogP(mol)
        except Chem.AtomValenceException:
            score = 0
        logps.append(score)

    return np.array(logps)


@R.register('plogp')
def penalized_logp(pred):
    """
    Logarithm of partition coefficient, penalized by cycle length and synthetic accessibility.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    # statistics from ZINC250k
    logp_mean = 2.4570953396190123
    logp_std = 1.434324401111988
    sa_mean = 3.0525811293166134
    sa_std = 0.8335207024513095
    cycle_mean = 0.0485696876403053
    cycle_std = 0.2860212110245455

    plogp = []
    for mol in pred:
        cycles = nx.cycle_basis(nx.Graph(mol.edge_list[:, :2].tolist()))
        if cycles:
            len_cycle = [len(cycle) for cycle in cycles]
            max_cycle = max(len_cycle)
            cycle = max(0, max_cycle - 6)
        else:
            cycle = 0
        mol = mol.to_molecule()
        try:
            with util.no_rdkit_log():
                mol.UpdatePropertyCache()
                Chem.GetSymmSSSR(mol)
                logp_val = Descriptors.MolLogP(mol)
                sa_val = sascorer.calc_score(mol)
            logp_val = (logp_val - logp_mean) / logp_std
            sa_val = (sa_val - sa_mean) / sa_std
            cycle = (cycle - cycle_mean) / cycle_std
            score = logp_val - sa_val - cycle
        except Chem.AtomValenceException:
            score = -30
        plogp.append(score)

    return np.array(plogp)


@R.register('sa')
def sa(pred):
    """
    Synthetic accessibility score.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    sas = []
    for mol in pred:
        with util.no_rdkit_log():
            score = sascorer.calc_score(mol.to_molecule())
        sas.append(score)

    return np.array(sas)


@R.register('qed')
def qed(pred):
    """
    Quantitative estimation of drug-likeness.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    qeds = []
    for mol in pred:
        try:
            with util.no_rdkit_log():
                score = Descriptors.qed(mol.to_molecule())
        except Chem.AtomValenceException:
            score = -1
        qeds.append(score)

    return np.array(qeds)


@R.register('validity')
def validity(pred):
    """
    Chemical validity of molecules.

    Parameters:
        pred (PackedMolecule): molecules to evaluate
    """
    validitis = []
    for mol in pred:
        with util.no_rdkit_log():
            smiles = mol.to_smiles()
            mol = Chem.MolFromSmiles(smiles)
        validitis.append(1 if mol else 0)

    return np.array(validitis)


@R.register('accuracy')
def accuracy(pred, target):
    """
    Compute classification accuracy over sets with equal size.

    Suppose there are :math:`N` sets and :math:`C` categories.

    Parameters:
        pred (Tensor): prediction of shape :math:`(N, C)`
        target (Tensor): target of shape :math:`(N,)`
    """
    return (pred.argmax(axis=-1) == target).float().mean()


@R.register('mcc')
def mcc(pred, target, eps=1e-6):
    """
    Matthews correlation coefficient between target and prediction.

    Definition follows matthews_corrcoef for K classes in sklearn.
    For details, see: 'https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef'

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    num_class = pred.size(-1)
    pred = pred.argmax(-1)
    ones = np.ones(len(target))
    confusion_matrix = scatter_add(ones, target * num_class + pred, axis=0, n_axis=num_class ** 2)
    confusion_matrix = confusion_matrix.view(num_class, num_class)
    t = confusion_matrix.sum(axis=1)
    p = confusion_matrix.sum(axis=0)
    c = confusion_matrix.trace()
    s = confusion_matrix.sum()
    return np.sqrt(c * s - np.matmul(t, p)) / ((s * s - np.matmul(p, p)) * (s * s - np.matmul(t, t)) + eps)


@R.register('spearmanr')
def spearmanr(pred, target, eps=1e-6):
    """
    Spearman correlation between target and prediction.
    Implement in Numpy, but non-diffierentiable. (validation metric only)

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(inputs):
        input_set, input_inverse = inputs.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = np.zeros(len(input_inverse))
        ranking[order] = np.arange(1, len(inputs) + 1)

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(ranking, input_inverse, axis=0, n_axis=len(input_set))
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr_val = covariance / (pred_std * target_std + eps)
    return spearmanr_val


@R.register('vacc')
def variadic_accuracy(inputs, target, size):
    """
    Compute classification accuracy over variadic sizes of categories.

    Suppose there are :math:`N` samples, and the number of categories in all samples is summed to :math:`B`.

    Parameters:
        input (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(N,)`. Each target is a relative index in a sample.
        size (Tensor): number of categories of shape :math:`(N,)`
    """
    index2graph = functional.size_to_index(size)

    input_class = scatter_max(inputs, index2graph)[1]
    target_index = target + size.cumsum(0) - size
    acc_val = (input_class == target_index).float()
    return acc_val


@R.register('metric.pearsonr')
class PearsonR(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self._abs_sum = 0
        self._total_num = 0

    def clear(self):
        self._abs_sum = 0
        self._total_num = 0

    def update(self, preds, targets):
        shifted_x = preds - preds.mean(axis=0)
        shifted_y = targets - targets.mean(axis=0)
        sigma_x = ops.sqrt((shifted_x ** 2).sum(axis=0))
        sigma_y = ops.sqrt((shifted_y ** 2).sum(axis=0))

        pearson = (shifted_x * shifted_y).sum(axis=0) / (sigma_x * sigma_y + 1e-8)
        pearson = util.clip(pearson, xmin=-1, xmax=1)
        self._abs_sum += pearson.sum()
        self._total_num += pearson.shape[0]

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError("The 'Pearson Coefficient' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        return self._abs_sum / self._total_num


@R.register('metric.l1_loss')
class MAE(Metric):
    """_summary_
    Args:
        Metric (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.clear()

    def update(self, preds, targets):
        loss = ops.l1_loss(preds, targets)
        return loss

    def eval(self):
        pass

    def clear(self):
        pass


@R.register('metric.rsquared')
class Rsquared(Metric):
    """
        Coefficient of determination/ R squared measure tells us the goodness of fit of our model.
        Rsquared = 1 means that the regression predictions perfectly fit the data.
        If Rsquared is less than 0 then our model is worse than the mean predictor.
        https://en.wikipedia.org/wiki/Coefficient_of_determination
    """

    def __init__(self):
        super().__init__()
        self.clear()

    def update(self, preds, targets):
        total_ss = ((targets - targets.mean()) ** 2).sum()
        residual_ss = ((targets - preds) ** 2).sum()
        return 1 - residual_ss / total_ss

    def eval(self):
        pass

    def clear(self):
        pass


@R.register('metric.rmsd')
class RMSD(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def update(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(ops.sqrt(((lig_coords_pred - lig_coords) ** 2).sum(axis=1).mean()))
        return ops.concat(rmsds).mean()

    def eval(self):
        pass

    def clear(self):
        pass


class KabschRMSD(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def update(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        """_summary_

        Args:
            ligs_coords_pred (List[Tensor]): _description_
            ligs_coords (List[Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            lig_coords_pred_mean = lig_coords_pred.mean(axis=0, keep_dims=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(axis=0, keep_dims=True)  # (1,3)

            mat_a = (lig_coords_pred - lig_coords_pred_mean).transpose(1, 0) @ (lig_coords - lig_coords_mean)

            _, ut, vt = ops.svd(mat_a)

            corr_mat = ops.diag(Tensor([1, 1, int(mat_a.det().sign())]))
            rotation = (ut @ corr_mat) @ vt
            translation = lig_coords_pred_mean - (lig_coords_mean @ rotation.T)  # (1,3)

            lig_coords = lig_coords @ rotation.t() + translation
            rmsds.append(((lig_coords_pred - lig_coords) ** 2).sum(axis=1).mean().sqrt())
        return ops.concat(rmsds).mean()

    def eval(self):
        pass

    def clear(self):
        pass


class RMSDmedian(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def update(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(((lig_coords_pred - lig_coords) ** 2).sum(axis=1).mean().sqrt())
        return ops.median(ops.concat(rmsds))

    def eval(self):
        pass

    def clear(self):
        pass


class RMSDfraction(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self, distance) -> None:
        super().__init__()
        self.distance = distance
        self.clear()

    def update(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        """_summary_

        Args:
            ligs_coords_pred (List[Tensor]): _description_
            ligs_coords (List[Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(((lig_coords_pred - lig_coords) ** 2).sum(axis=1).mean().sqrt())
        count = ops.concat(rmsds) < self.distance
        return 100 * count.sum() / len(count)

    def eval(self):
        pass

    def clear(self):
        pass


class CentroidDist(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def update(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        """_summary_

        Args:
            ligs_coords_pred (List[Tensor]): _description_
            ligs_coords (List[Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            distances.append(ops.norm(lig_coords_pred.mean(axis=0)-lig_coords.mean(axis=0)))
        return ops.concat(distances).mean()

    def clear(self):
        pass

    def eval(self):
        pass


class CentroidDistMedian(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def update(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        """_summary_

        Args:
            ligs_coords_pred (List[Tensor]): _description_
            ligs_coords (List[Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            distances.append(ops.norm(lig_coords_pred.mean(axis=0)-lig_coords.mean(axis=0)))
        return ops.median(ops.concat(distances))

    def eval(self):
        pass

    def clear(self):
        pass


class CentroidDistFraction(Metric):
    """_summary_

    Args:
        Metric (_type_): _description_
    """

    def __init__(self, distance) -> None:
        super().__init__()
        self.distance = distance

    def update(self, preds: List[Tensor], targets: List[Tensor]) -> Tensor:
        """_summary_

        Args:
            preds (List[Tensor]): _description_
            targets (List[Tensor]): _description_

        Returns:
            Tensor: _description_
        """
        distances = []
        for lig_coords_pred, lig_coords in zip(preds, targets):
            distances.append(ops.norm(lig_coords_pred.mean(axis=0)-lig_coords.mean(axis=0)))
        count = ops.concat(distances) < self.distance
        return 100 * count.sum() / len(count)

    def eval(self):
        pass

    def clear(self):
        pass

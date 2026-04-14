# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
from typing import List, Tuple

import numpy as np
from numpy import ndarray
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.train import Metric

from ..utils import functional
from ..configs import Registry as R
from ..utils import scatter_add, scatter_mean, scatter_max

from ..configs import Config


@R.register('metric.accuracy')
class Accuracy(Metric):
    """Accuracy metric

    Args:
        eval_type (str, optional): The type of evaluation. Defaults to 'binary'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        super().__init__()
        self._target = []

    def update(self, pred, target):
        """
        Args:
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

    Args:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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

    Args:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): targets of shape :math:`(n,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    total = np.var(target, unbiased=False)
    residual = ops.mse_loss(pred, target)
    return 1 - residual / total


@R.register('mcc')
def mcc(pred, target, eps=1e-6):
    """
    Matthews correlation coefficient between target and prediction.

    Definition follows matthews_corrcoef for K classes in sklearn.
    For details, see: 'https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef'

    Args:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    num_class = pred.shape(-1)
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

    Args:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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

    Args:
        inputs (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(N,)`. Each target is a relative index in a sample.
        size (Tensor): number of categories of shape :math:`(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    index2graph = functional.size_to_index(size)

    input_class = scatter_max(inputs, index2graph)[1]
    target_index = target + size.cumsum(0) - size
    acc_val = (input_class == target_index).float()
    return acc_val


@R.register('metric.pearsonr')
class PearsonR(Metric):
    """
    Pearson correlation coefficient

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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
        pearson = pearson.clip(min=-1, max=1)
        self._abs_sum += pearson.sum()
        self._total_num += pearson.shape[0]

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError("The 'Pearson Coefficient' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        return self._abs_sum / self._total_num


@R.register('metric.rsquared')
class Rsquared(Metric):
    """
        Coefficient of determination/ R squared measure tells us the goodness of fit of our model.
        Rsquared = 1 means that the regression predictions perfectly fit the data.
        If Rsquared is less than 0 then our model is worse than the mean predictor.
        https://en.wikipedia.org/wiki/Coefficient_of_determination

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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
    """
    Root Mean Square Deviation

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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
    """
    Kabsch Root Mean Square Deviation

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self) -> None:
        super().__init__()
        self.clear()

    def update(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        """_summary_

        Args:
            ligs_coords_pred (List[Tensor]): predicted coordinates of ligands
            ligs_coords (List[Tensor]): real coordinates of ligands

        Returns:
            Tensor: output value
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
    """
    Median of RMSD

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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
    """Root Mean Square Fluctuation

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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
    """
    Centroid distance

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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
    """
    Median of centroid distance

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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
    """Fraction of centroid distance

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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


class GridCentroidDist(Metric):
    """Centroid Distance between grids data.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        super().__init__()
        self._centroids = []

    def clear(self):
        self._centroids = []

    def update(self, pred, target):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_
        """
        position = np.arange(36)
        # target = target.squeeze(-1)
        # pred = pred.squeeze(-1)
        if isinstance(target, ms.Tensor):
            target = target.asnumpy()
        if isinstance(pred, ms.Tensor):
            pred = pred.asnumpy()
        center_target_x = (target.sum(axis=(2, 3)) * position).mean(1) / 2
        center_target_y = (target.sum(axis=(1, 3)) * position).mean(1) / 2
        center_target_z = (target.sum(axis=(1, 2)) * position).mean(1) / 2

        max_predict_x = pred.max(axis=(2, 3)).argmax(axis=1) / 2
        max_predict_y = pred.max(axis=(1, 3)).argmax(axis=1) / 2
        max_predict_z = pred.max(axis=(1, 2)).argmax(axis=1) / 2

        center_p_t = ((max_predict_x-center_target_x) ** 2 +
                      (max_predict_y - center_target_y) ** 2 +
                      (max_predict_z-center_target_z) ** 2) ** 0.5
        self._centroids.append(center_p_t)

    def eval(self):
        centroids = np.concatenate(self._centroids)
        return centroids.mean()


class MaxError(Metric):
    r"""Metric to calcaulte the max error.

    Args:
        indexes (tuple):        Indexes for label and predicted data. Default: (1, 2)

        reduce_dims (bool): Whether to summation the data of all atoms in molecule. Default: ``True``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, index: int = 0, **kwargs):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)
        self._indexes = int(index)
        self._max_error = 0

    def clear(self):
        self._max_error = 0

    def update(self,
               loss: Tensor,
               predicts: Tuple[Tensor],
               labels: List[Tensor],
               num_atoms: Tensor,
               *args):
        """update metric"""
        # pylint: disable=unused-argument

        predicts: ndarray = self._convert_data(predicts)
        label: ndarray = self._convert_data(labels)
        diff = label.reshape(predicts.shape) - predicts
        max_error = diff.max() - diff.min()
        if max_error > self._max_error:
            self._max_error = max_error

    def eval(self):
        return self._max_error


class Error(Metric):
    r"""Metric to calcaulte the error.

    Args:
        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: ``False``.

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__()
        self._kwargs = kwargs

        if not isinstance(index, int):
            raise TypeError(f'The type of index should be int but got: {type(index)}')

        self._index = int(index)

        self._reduction = reduction
        if reduction is not None:
            if not isinstance(reduction, str):
                raise TypeError(f'The type of "reduction" must be str, but got: {type(reduction)}')
            if reduction.lower() not in ('mean', 'sum', 'none'):
                raise ValueError(f"For '{self.__class__.__name__}', the 'reduction' must be in "
                                 f" ['mean', 'sum', 'none'], but got {reduction}.")
            self._reduction = reduction.lower()
            if self._reduction == 'none':
                self._reduction = None

        self._aggregate = aggregate
        if reduction is not None:
            if not isinstance(aggregate, str):
                raise TypeError(f'The type of "aggregate" must be str, but got: {type(aggregate)}')
            if aggregate.lower() not in ('mean', 'sum', 'none'):
                raise ValueError(f"For '{self.__class__.__name__}', the 'reduction' must be in "
                                 f" ['mean', 'sum', 'none'], but got {aggregate}.")
            self._aggregate = aggregate.lower()
            if self._aggregate == 'none':
                self._aggregate = None

        self._by_atoms = per_atom

        self._error_sum = 0
        self._samples_num = 0

        self.clear()

    def clear(self):
        self._error_sum = 0
        self._samples_num = 0

    # pylint: disable=unused-argument
    def update(self,
               loss: Tensor,
               predicts: Tuple[Tensor],
               labels: List[Tensor],
               atom_mask: Tensor,
               ):
        """update metric"""
        # The shape looks like (B, ...)
        predict = self._convert_data(predicts[self._index])
        # The shape looks like (B, ...)
        label = self._convert_data(labels[self._index])

        error: ndarray = self._calc_error(predict, label)
        batch_size = error.shape[0]

        if len(error.shape) > 2 and self._aggregate is not None:
            axis = tuple(range(2, len(error.shape)))
            # The shape is changed to (B, A) from (B, A, ...)
            if self._aggregate == 'mean':
                error = np.mean(error, axis=axis)
            else:
                error = np.sum(error, axis=axis)

        num_atoms = 1
        total_num = batch_size
        if atom_mask is not None:
            atom_mask = self._convert_data(atom_mask)
            # pylint: disable=unexpected-keyword-arg
            # The shape changes like (B, 1) <- (B, A) OR (1, 1) <- (1, A)
            num_atoms = np.count_nonzero(atom_mask, -1, keepdims=True)
            total_num = np.sum(num_atoms)
            if num_atoms.shape[0] == 1:
                total_num *= batch_size

        atomic = False
        if atom_mask is not None and error.shape[1] == atom_mask.shape[1]:
            atomic = True
            atom_mask_ = atom_mask
            if error.ndim != atom_mask.ndim:
                # The shape is changed to (B, A, ...) from (B, A)
                newshape = atom_mask.shape + (1,) * (error.ndim - atom_mask.ndim)
                atom_mask_ = np.reshape(atom_mask, newshape)
            # The shape looks like (B, A) * (B, A)
            error *= atom_mask_

        weight = batch_size
        if self._reduction is not None:
            error_shape1 = error.shape[1]
            # The shape is changed to (B,) from (B, ...)
            axis = tuple(range(1, len(error.shape)))
            error = np.sum(error, axis=axis)
            if self._reduction == 'mean':
                weight = batch_size * error_shape1
                if atomic or self._by_atoms:
                    weight = total_num

        self._error_sum += np.sum(error, axis=0)
        self._samples_num += weight

    def eval(self) -> float:
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._error_sum / self._samples_num

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        """calculate error"""
        raise NotImplementedError


class MAE(Error):
    r"""Metric to calcaulte the mean absolute error.

    Args:
        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: ``False``.

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
            aggregate=aggregate,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        return np.abs(label.reshape(predict.shape) - predict)


class MSE(Error):
    r"""Metric to calcaulte the mean square error.

    Args:
        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: ``False``.

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
            aggregate=aggregate,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        return np.square(label.reshape(predict.shape) - predict)


class MNE(Error):
    r"""Metric to calcaulte the mean norm error.

    Args:
        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: ``False``.

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
            aggregate=aggregate,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        diff = label.reshape(predict.shape) - predict
        return np.linalg.norm(diff, axis=-1)


class RMSE(Error):
    r"""Metric to calcaulte the root mean square error.

    Args:
        index (int): Index of the output for which error to be calculated. Default: 0

        per_atom (bool): Calculate the error per atom. Default: ``False``.

        reduction (str): The way to reduce the shape of the output tensor from `(B, Y)` to `(B, 1)`.
            The optional values are "mean", "sum", and "none". Default: "mean".

        aggregate (str): The way of aggregating the extra dimensions of the output tensor,
            i.e. from `(B, A, ...)` to `(B, A)`. The optional values are "mean", "sum", and "none".
            Default: "sum".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 index: int = 0,
                 per_atom: bool = False,
                 reduction: str = 'mean',
                 aggregate: str = 'sum',
                 **kwargs
                 ):

        super().__init__(
            index=index,
            per_atom=per_atom,
            reduction=reduction,
            aggregate=aggregate,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return np.sqrt(self._error_sum / self._samples_num)

    def _calc_error(self, predict: ndarray, label: ndarray) -> ndarray:
        return np.square(label.reshape(predict.shape) - predict)


class Loss(Metric):
    r"""Metric to calcaulte the loss function.

    Args:
        indexes (int):            Index for loss function. Default: 0

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.clear()

    def clear(self):
        self._sum_loss = 0
        self._total_num = 0

    def update(self,
               loss: Tensor,
               predicts: Tuple[Tensor],
               labels: List[Tensor],
               num_atoms: Tensor,
               *args):
        # pylint: disable=unused-argument
        """update metric"""
        loss = self._convert_data(loss)

        if loss.ndim == 0:
            loss = loss.reshape(1)

        if loss.ndim != 1:
            raise ValueError(
                "Dimensions of loss must be 1, but got {}".format(loss.ndim))

        loss = loss.mean(-1)
        self._sum_loss += loss
        self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Total number can not be 0.')
        return self._sum_loss / self._total_num

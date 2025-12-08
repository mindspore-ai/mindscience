# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Utils."""

from typing import NamedTuple, List, Optional, Type

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, mint, context

MSINT = [ms.int64, ms.int32, ms.int16, ms.int8, ms.uint8]


def aggregate_nodes(tensor: Tensor, n_node: Tensor, reduction: str = "mean", deterministic: bool = False) -> Tensor:
    """Aggregates over a tensor based on graph sizes."""
    count = len(n_node)
    if deterministic:
        ms.set_seed(1)
    segments = ops.arange(count).repeat_interleave(n_node).astype(ms.int32)
    if reduction == "sum":
        return scatter_sum(tensor, segments, dim=0)
    if reduction == "mean":
        return scatter_mean(tensor, segments, dim=0)
    if reduction == "max":
        return scatter_max(tensor, segments, dim=0)
    raise ValueError("Invalid reduction argument. Use sum, mean or max.")


def segment_sum(data: Tensor, segment_ids: Tensor, num_segments: int):
    """Computes index based sum over segments of a tensor."""
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments)


def segment_max(data: Tensor, segment_ids: Tensor, num_segments: int):
    """Computes index based max over segments of a tensor."""
    assert segment_ids is not None, "segment_ids must not be None"
    assert num_segments > 0, "num_segments must be greater than 0"
    max_op = ops.ArgMaxWithValue(axis=0)
    _, max_values = max_op(data)
    return max_values


def segment_mean(data: Tensor, segment_ids: Tensor, num_segments: int):
    """Computes index based mean over segments of a tensor."""
    sum_v = segment_sum(data, segment_ids, num_segments)
    count = ops.scatter_add(ops.zeros(
        (num_segments,), dtype=ms.int32), segment_ids, ops.ones_like(segment_ids))
    return sum_v / count.astype(sum_v.dtype)


def segment_softmax(data: Tensor, segment_ids: Tensor, num_segments: int, weights: Optional[Tensor] = None):
    """Computes a softmax over segments of the tensor."""
    data_max = segment_max(data, segment_ids, num_segments)
    data = data - data_max[segment_ids]

    unnormalised_probs = ops.exp(data)
    if weights is not None:
        unnormalised_probs = unnormalised_probs * weights
    denominator = segment_sum(unnormalised_probs, segment_ids, num_segments)

    return safe_division(unnormalised_probs, denominator, segment_ids)


def safe_division(numerator: Tensor, denominator: Tensor, segment_ids: Tensor):
    """Divides logits by denominator, setting 0 where the denominator is zero."""
    result = ops.where(denominator[segment_ids] ==
                       0, 0, numerator / denominator[segment_ids])
    return result


def _broadcast(src: Tensor, other: Tensor, dim: int):
    """Broadcasts the source tensor to match the shape of the other tensor along the specified dimension."""
    if dim < 0:
        dim = other.ndim + dim
    if src.ndim == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.ndim, other.ndim):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None, reduce: str = "sum"
) -> Tensor:
    """Applies a sum reduction of the orb_models tensor along the specified dimension."""
    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = ops.zeros(size, dtype=src.dtype)
        return mint.scatter_add(out, dim, index, src)
    return mint.scatter_add(out, dim, index, src)


def scatter_std(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None, unbiased: bool = True
) -> Tensor:
    """Computes the standard deviation of the orb_models tensor along the specified dimension."""
    if out is not None:
        dim_size = out.shape[dim]

    if dim < 0:
        dim = src.ndim + dim

    count_dim = dim
    if index.ndim <= dim:
        count_dim = index.ndim - 1

    ones = ops.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = _broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = _broadcast(count, tmp, dim).clip(1)
    mean = tmp / count

    var = src - mean.gather(dim, index)
    var = var * var
    out = scatter_sum(var, index, dim, out=out, dim_size=dim_size)

    if unbiased:
        count = count - 1
        count = count.clip(1)
    out = out / (count + 1e-6)
    out = ops.sqrt(out)
    return out


def scatter_mean(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None
) -> Tensor:
    """Computes the mean of the orb_models tensor along the specified dimension."""
    out = scatter_sum(src, index, dim, out=out, dim_size=dim_size)
    dim_size = out.shape[dim]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.ndim
    if index.ndim <= index_dim:
        index_dim = index.ndim - 1

    ones = ops.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, index_dim, dim_size=dim_size)
    count = count.clip(1)
    count = _broadcast(count, out, dim)
    out = out / count
    return out


def scatter_max(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None
) -> Tensor:
    """Computes the maximum of the orb_models tensor for each group defined by index along the specified dimension."""
    if out is not None:
        raise NotImplementedError(
            "The 'out' argument is not supported for scatter_max")

    if src.dtype in MSINT:
        init_value = np.iinfo(src.dtype).min
    else:
        init_value = np.finfo(src.dtype).min

    if dim < 0:
        dim = src.ndim + dim

    if dim_size is None:
        dim_size = int(index.max()) + 1

    result = ops.ones(
        (dim_size, *src.shape[:dim], *src.shape[dim + 1:]), dtype=src.dtype)
    result = init_value * result
    broadcasted_index = _broadcast(index, src, dim)

    scatter_result = ops.ZerosLike()(result)
    index = ops.expand_dims(broadcasted_index, dim)
    scatter_result = scatter_result.scatter_update(index, src)
    result = ops.Maximum()(result, scatter_result)
    return result


class SSP(nn.Cell):
    """Shifted Softplus activation function.

    This activation is twice differentiable so can be used when regressing
    gradients for conservative force fields.
    """

    def __init__(self, beta: int = 1, threshold: int = 20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def construct(self, input_x: Tensor) -> Tensor:
        sp0 = ops.softplus(ops.zeros(1), self.beta, self.threshold)
        return ops.softplus(input_x, self.beta, self.threshold) - sp0


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: Optional[int] = None,
        output_activation: Type[nn.Cell] = nn.Identity,
        activation: Type[nn.Cell] = SSP,
        dropout: Optional[float] = None,
) -> nn.Cell:
    """Build a MultiLayer Perceptron.

    Args:
      input_size: Size of input layer.
      hidden_layer_sizes: An array of input size for each hidden layer.
      output_size: Size of the output layer.
      output_activation: Activation function for the output layer.
      activation: Activation function for the hidden layers.
      dropout: Dropout rate for hidden layers.
      checkpoint: Whether to use checkpointing.

    Returns:
      mlp: An MLP sequential container.
    """
    # Size of each layer
    layer_sizes = [input_size] + hidden_layer_sizes
    if output_size:
        layer_sizes.append(output_size)

    # Number of layers
    nlayers = len(layer_sizes) - 1

    # Create a list of activation functions and
    # set the last element to output activation function
    act = [activation for _ in range(nlayers)]
    act[-1] = output_activation

    # Create a list to hold layers
    layers = []
    for i in range(nlayers):
        if dropout is not None:
            layers.append(nn.Dropout(keep_prob=1 - dropout))
        layers.append(nn.Dense(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(act[i]())

    # Create a sequential container
    mlp = nn.SequentialCell(layers)
    return mlp


class CheckpointedSequential(nn.Cell):
    """Sequential container with checkpointing."""

    def __init__(self, *args, n_layers: int = 1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.CellList(list(args))

    def construct(self, input_x: Tensor) -> Tensor:
        """Forward pass with checkpointing enabled in training mode."""
        if context.get_context("mode") == context.GRAPH_MODE:
            # In graph mode, checkpointing is handled by MindSpore's graph optimization
            for layer in self.layers:
                input_x = layer(input_x)
        else:
            # In PyNative mode, we can manually checkpoint each layer
            for i in range(self.n_layers):
                input_x = self.layers[i](input_x)
        return input_x


class ReferenceEnergies(NamedTuple):
    """
    Reference energies for an atomic system.

    Our vasp reference energies are computed by running vasp
    optimisations on a single atom of each atom-type.

    Other reference energies are fitted using least-squares.

    Doing so with mp-traj-d3 gives the following:

    ---------- LSTQ ----------
    Reference MAE:  13.35608855004781
    (energy - ref) mean: 1.3931169304958624
    (energy - ref) std: 22.45615276341948
    (energy - ref)/natoms mean: 0.16737045963056316
    (energy - ref)/natoms std: 0.8189314920219992
    CO2: Predicted vs DFT: -23.154158610392408 vs -22.97
    H2O: Predicted vs DFT: -11.020918107591324 vs - 14.23
    ---------- VASP ----------
    Reference MAE:  152.4722089438871
    (energy - ref) mean: -152.47090833346033
    (energy - ref) std: 153.89049784836962
    (energy - ref)/natoms mean: -4.734136414817941
    (energy - ref)/natoms std: 1.3603868419157275
    CO2: Predicted vs DFT: -4.35888857 vs -22.97
    H2O: Predicted vs DFT: -2.66521147 vs - 14.23
    ---------- Shifted VASP ----------
    Reference MAE:  28.95948216608197
    (energy - ref) mean: 0.7083632520428979
    (energy - ref) std: 48.61861182844561
    (energy - ref)/natoms mean: 0.17320099403091083
    (energy - ref)/natoms std: 1.3603868419157275
    CO2: Predicted vs DFT: -19.080900796546562 vs -22.97
    H2O: Predicted vs DFT: -12.479886287697706 vs - 14.23

    Args:
        coefficients: Coefficients for each atom in the periodic table.
            Must be of length 118 with first entry equal to 0.
        residual_mean: Mean of (pred - target)
        residual_std: Standard deviation of (pred - target)
        residual_mean_per_atom: Mean of (pred - target)/n_atoms.
        residual_std_per_atom: Standard deviation of (pred - target)/n_atoms.
    """

    coefficients: np.ndarray
    residual_mean: float
    residual_std: float
    residual_mean_per_atom: float
    residual_std_per_atom: float


# We have only computed these for the first
# 88 elements, and padded the remainder with 0.
vasp_reference_energies = ReferenceEnergies(
    coefficients=np.array(
        [
            0.0,  # padding
            -1.11725225e00,
            7.69290000e-04,
            -3.22788480e-01,
            -4.47021900e-02,
            -2.90627280e-01,
            -1.26297013e00,
            -3.12415058e00,
            -1.54795922e00,
            -4.39757050e-01,
            -1.25673900e-02,
            -2.63927430e-01,
            -1.92670300e-02,
            -2.11267040e-01,
            -8.24799500e-01,
            -1.88734631e00,
            -8.91048980e-01,
            -2.58371430e-01,
            -2.50008000e-02,
            -2.71936150e-01,
            -7.11147600e-02,
            -2.06076796e00,
            -2.42753196e00,
            -3.57144559e00,
            -5.45540047e00,
            -5.15708214e00,
            -3.31393675e00,
            -1.84639284e00,
            -6.32812480e-01,
            -2.38017450e-01,
            -1.41047600e-02,
            -2.06349980e-01,
            -7.77477960e-01,
            -1.70160351e00,
            -7.84231510e-01,
            -2.27541260e-01,
            -2.26104900e-02,
            -2.79760570e-01,
            -9.92851900e-02,
            -2.18560872e00,
            -2.26603086e00,
            -3.14842282e00,
            -4.61199158e00,
            -3.34329762e00,
            -2.48233722e00,
            -1.27872811e00,
            -1.47784242e00,
            -2.04068960e-01,
            -1.89639300e-02,
            -1.88520140e-01,
            -6.76700640e-01,
            -1.42966694e00,
            -6.57608340e-01,
            -1.89308030e-01,
            -1.20491300e-02,
            -3.07991050e-01,
            -1.58601400e-01,
            -4.89728600e-01,
            -1.35031403e00,
            -3.31509450e-01,
            -3.23660410e-01,
            -3.15316610e-01,
            -3.11184530e-01,
            -8.44684689e00,
            -1.04408371e01,
            -2.30922790e-01,
            -2.26295040e-01,
            -2.92747580e-01,
            -2.92191740e-01,
            -2.91465170e-01,
            -3.80611000e-02,
            -2.87691040e-01,
            -3.51528971e00,
            -3.51343142e00,
            -4.64232388e00,
            -2.88816624e00,
            -1.46089612e00,
            -5.36042350e-01,
            -1.87182020e-01,
            -1.33549100e-02,
            -1.68142250e-01,
            -6.25378750e-01,
            -1.32291753e00,
            -3.26246040e-01,
            -1.10239294e00,
            -2.30839543e00,
            -4.61968511e00,
            -7.30638139e00,
            -1.04613411e01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    ),
    residual_mean=-152.47090833346033,
    residual_std=153.89049784836962,
    residual_mean_per_atom=-4.734136414817941,
    residual_std_per_atom=1.3603868419157275,
)

vasp_shifted_reference_energies = ReferenceEnergies(
    coefficients=np.array(
        [
            0.0,  # padding
            -6.0245896588488534,
            -4.9065681188488535,
            -5.230125888848853,
            -4.952039598848853,
            -5.197964688848853,
            -6.170307538848854,
            -8.031487988848854,
            -6.455296628848854,
            -5.347094458848853,
            -4.919904798848854,
            -5.171264838848853,
            -4.9266044388488535,
            -5.118604448848854,
            -5.732136908848854,
            -6.794683718848853,
            -5.798386388848853,
            -5.165708838848854,
            -4.932338208848853,
            -5.179273558848854,
            -4.978452168848854,
            -6.968105368848853,
            -7.334869368848853,
            -8.478782998848853,
            -10.362737878848854,
            -10.064419548848853,
            -8.221274158848853,
            -6.7537302488488535,
            -5.540149888848854,
            -5.145354858848854,
            -4.921442168848854,
            -5.113687388848853,
            -5.684815368848853,
            -6.6089409188488535,
            -5.691568918848853,
            -5.134878668848853,
            -4.929947898848853,
            -5.187097978848853,
            -5.006622598848853,
            -7.092946128848853,
            -7.173368268848853,
            -8.055760228848854,
            -9.519328988848853,
            -8.250635028848853,
            -7.389674628848853,
            -6.186065518848854,
            -6.3851798288488535,
            -5.111406368848853,
            -4.9263013388488535,
            -5.095857548848853,
            -5.5840380488488535,
            -6.337004348848853,
            -5.564945748848854,
            -5.096645438848854,
            -4.919386538848854,
            -5.2153284588488535,
            -5.065938808848854,
            -5.397066008848854,
            -6.257651438848853,
            -5.238846858848854,
            -5.230997818848854,
            -5.2226540188488535,
            -5.218521938848854,
            -13.354184298848853,
            -15.348174508848853,
            -5.138260198848854,
            -5.133632448848854,
            -5.200084988848854,
            -5.199529148848853,
            -5.198802578848854,
            -4.945398508848854,
            -5.195028448848854,
            -8.422627118848853,
            -8.420768828848853,
            -9.549661288848853,
            -7.795503648848854,
            -6.368233528848854,
            -5.443379758848853,
            -5.094519428848853,
            -4.920692318848854,
            -5.075479658848853,
            -5.532716158848854,
            -6.230254938848853,
            -5.2335834488488535,
            -6.009730348848853,
            -7.2157328388488535,
            -9.527022518848852,
            -12.213718798848854,
            -15.368678508848854,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
            -4.9073374088488535,
        ]
    ),
    residual_mean=0.7083632520428979,
    residual_std=48.61861182844561,
    residual_mean_per_atom=0.17320099403091083,
    residual_std_per_atom=1.3603868419157275,
)

mp_traj_d3_reference_energies = ReferenceEnergies(
    coefficients=np.array(
        [
            0.0,  # padding
            -3.6818229500085327,
            -1.3199148098871394,
            -3.688797198716366,
            -4.938608191337134,
            -7.901604711660046,
            -8.475968295226822,
            -7.42601366967988,
            -7.339095157582792,
            -4.9239197309790725,
            -0.061236726924086424,
            -3.0526401941340806,
            -3.0836199809602105,
            -5.055909838526647,
            -7.875649504560413,
            -7.175538036602013,
            -4.814514763424572,
            -2.9198,
            -0.13127266880110078,
            -2.8792125576832865,
            -5.635016298424046,
            -8.164720105254204,
            -10.712143655281858,
            -9.00292017736733,
            -9.619640942931085,
            -8.610981088341331,
            -7.3506162257219385,
            -5.943664565392655,
            -5.592846831852426,
            -3.6868017794232077,
            -1.579885044321145,
            -3.744040760877656,
            -4.945137332817033,
            -4.2021571924020655,
            -4.045303645442562,
            -2.652667661940346,
            6.497305115069106,
            -2.806819346028444,
            -5.164089337915934,
            -10.493037547114369,
            -12.256967896681578,
            -12.642602087796805,
            -9.20874164629371,
            -9.292405362859506,
            -8.304141715175632,
            -7.49355696426791,
            -5.44150554776011,
            -2.5621691409635474,
            -0.9687174918829102,
            -3.055905969721681,
            -4.02975498585447,
            -3.847125028451477,
            -3.1016305514702203,
            -1.8001556831915142,
            9.742275211909387,
            -3.045410331644577,
            -5.204088972093178,
            -9.267561428901118,
            -9.031458669303145,
            -8.345252241333469,
            -8.584977779192705,
            -7.955970517402418,
            -8.519743221802353,
            -13.927799873369949,
            -19.12242499580686,
            -8.156787154342183,
            -8.505944162624234,
            -8.015433843487497,
            -7.129355408977684,
            -8.166165621829014,
            -3.9995952334750644,
            -7.884852034766514,
            -13.281575162667238,
            -14.598283494757041,
            -9.729591400065184,
            -11.798570715867179,
            -9.878207068760076,
            -7.891075131963705,
            -5.964524120587406,
            -2.9665634245721275,
            -0.10530075207060852,
            -2.649420791761001,
            -4.00193074336809,
            -3.7403644338639785,
            -1.5543122344752192e-15,
            -8.881784197001252e-16,
            -8.881784197001252e-16,
            0.0,
            0.0,
            -5.480602125607218,
            -11.9439263006771,
            -12.974770001312883,
            -14.376719109855834,
            -15.49262474740642,
            -16.02533150334938,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ),
    residual_mean=1.3931169304958624,
    residual_std=22.45615276341948,
    residual_mean_per_atom=0.16737045963056316,
    residual_std_per_atom=0.8189314920219992,
)

REFERENCE_ENERGIES = {
    "vasp": vasp_reference_energies,
    "vasp-shifted": vasp_shifted_reference_energies,
    "mp-traj-d3": mp_traj_d3_reference_energies,
}

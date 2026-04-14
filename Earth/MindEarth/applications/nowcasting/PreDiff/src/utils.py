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
# ==============================================================================
"all util"
import math
import os
import shutil
import re
from copy import deepcopy
from inspect import isfunction
from typing import Dict, Any, Callable, Optional, Sequence
import numpy as np
import cv2
from einops import repeat, rearrange

import mindspore as ms
from mindspore import ops, mint, nn, Parameter, Tensor
from mindspore.train.metrics.metric import Metric
from mindspore.common.initializer import (
    initializer,
    One,
    Zero,
    HeNormal,
    Uniform,
    TruncatedNormal,
)
from mindearth.utils import create_logger


PREPROCESS_SCALE_01 = {
    "vis": 1,
    "ir069": 1,
    "ir107": 1,
    "vil": 1 / 255,
    "lght": 1,
}
PREPROCESS_OFFSET_01 = {
    "vis": 0,
    "ir069": 0,
    "ir107": 0,
    "vil": 0,
    "lght": 0,
}


class DiagonalGaussianDistribution(nn.Cell):
    """Diagonal Gaussian distribution layer for variational autoencoders.

    This class represents a diagonal Gaussian distribution parameterized by mean and log-variance,
    supporting sampling, KL divergence computation, and negative log-likelihood evaluation.

    Attributes:
        mean (Tensor): Mean values of the distribution
        logvar (Tensor): Clamped log-variance values
        std (Tensor): Standard deviation derived from logvar
        var (Tensor): Variance derived from logvar
        deterministic (bool): Flag indicating deterministic sampling mode
    """
    def __init__(self, parameters, deterministic=False):
        super().__init__()
        self.parameters = parameters
        self.mean, self.logvar = ops.chunk(parameters, 2, axis=1)
        self.logvar = ops.clamp(self.logvar, -30.0, 20.0)

        self.deterministic = deterministic
        self.std = ops.exp(0.5 * self.logvar)

        self.var = ops.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = ops.zeros_like(self.mean)

    def sample(self):
        """Generate a sample from the distribution.

        Returns:
            Tensor: Sampled tensor with same shape as mean

        Notes:
            - If deterministic=True, returns mean directly without noise
            - Uses reparameterization trick for differentiable sampling
        """
        sample = mint.randn(self.mean.shape)

        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        """Compute KL divergence between this distribution and another or standard normal."""
        if self.deterministic:
            return ms.Tensor([0.0])
        if other is None:
            return 0.5 * ops.sum(
                ops.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3]
            )
        return 0.5 * ops.sum(
            ops.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def mode(self):
        """Return the mode of the distribution (mean value)."""
        return self.mean


def _threshold(target, pred, t_input):
    """Apply thresholding to target and prediction tensors."""
    t = (target >= t_input).float()
    p = (pred >= t_input).float()
    is_nan = ops.logical_or(ops.isnan(target), ops.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p


@staticmethod
def process_data_dict_back(data_dict, data_types=None):
    """Rescale and offset data in dictionary using predefined parameters.

    Applies normalization using scale and offset values from global dictionaries.

    Args:
        data_dict (dict): Dictionary containing data tensors
        data_types (list, optional): Keys to process. Defaults to all keys in data_dict.
        rescale (str, optional): Rescaling mode identifier. Defaults to "01".

    Returns:
        dict: Processed data dictionary with normalized values
    """
    scale_dict = PREPROCESS_SCALE_01
    offset_dict = PREPROCESS_OFFSET_01
    if data_types is None:
        data_types = data_dict.keys()
    for key in data_types:
        data = data_dict[key]
        data = data.float() / scale_dict[key] - offset_dict[key]
        data_dict[key] = data
    return data_dict


class SEVIRSkillScore(Metric):
    """
    Class for calculating meteorological skill scores using threshold-based metrics.
    This metric class computes performance metrics like CSI, POD, etc.,
    across multiple thresholds for weather prediction evaluation.
    Args:
        layout (str): Data dimension layout specification (default "NHWT")
        mode (str): Operation mode affecting dimension handling ("0", "1", or "2")
        seq_in (Optional[int]): Input sequence length (required for modes 1/2)
        preprocess_type (str): Data preprocessing method ("sevir" or "sevir_pool*")
        threshold_list (Sequence[int]): List of thresholds for binary classification
        metrics_list (Sequence[str]): List of metrics to compute (csi, bias, sucr, pod)
        eps (float): Small value to prevent division by zero
    """
    def __init__(
            self,
            layout: str = "NHWT",
            mode: str = "0",
            seq_in: Optional[int] = None,
            preprocess_type: str = "sevir",
            threshold_list: Sequence[int] = (16, 74, 133, 160, 181, 219),
            metrics_list: Sequence[str] = ("csi", "bias", "sucr", "pod"),
            eps: float = 1e-4,
    ):
        super().__init__()
        self.layout = layout
        assert preprocess_type == "sevir" or preprocess_type.startswith("sevir_pool")
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_in = seq_in
        if mode in ("0",):
            self.keep_seq_in_dim = False
            state_shape = (len(self.threshold_list),)
        elif mode in ("1", "2"):
            self.keep_seq_in_dim = True
            assert isinstance(
                self.seq_in, int
            ), "seq_in must be provided when we need to keep seq_in dim."
            state_shape = (len(self.threshold_list), self.seq_in)

        else:
            raise NotImplementedError(f"mode {mode} not supported!")

        self.hits = Parameter(ops.zeros(state_shape), name="hits")
        self.misses = Parameter(ops.zeros(state_shape), name="misses")
        self.fas = Parameter(ops.zeros(state_shape), name="fas")

    @property
    def hits_misses_fas_reduce_dims(self):
        """Dimensions to reduce when calculating metric statistics.

        Returns:
            list[int]: List of dimensions to collapse during metric computation
        """
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find("T")
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_in_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def clear(self):
        """Clear the internal states."""
        self.hits.set_data(mint.zeros_like(self.hits))
        self.misses.set_data(mint.zeros_like(self.misses))
        self.fas.set_data(mint.zeros_like(self.fas))

    @staticmethod
    def pod(hits, misses, _, eps):
        """Probability of Detection"""
        return hits / (hits + misses + eps)

    @staticmethod
    def sucr(hits, _, fas, eps):
        """Probability of hits"""
        return hits / (hits + fas + eps)

    @staticmethod
    def csi(hits, misses, fas, eps):
        """critical success index"""
        return hits / (hits + misses + fas + eps)

    @staticmethod
    def bias(hits, misses, fas, eps):
        """Bias score"""
        bias = (hits + fas) / (hits + misses + eps)
        logbias = ops.pow(bias / ops.log(Tensor(2.0)), 2.0)
        return logbias

    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        """Calculate contingency table statistics for given threshold.

        Args:
            pred (Tensor): Model prediction tensor
            target (Tensor): Ground truth tensor
            threshold (int): Threshold value for binarization

        Returns:
            tuple[Tensor, Tensor, Tensor]: Hits, misses, false alarms
        """
        t, p = _threshold(target, pred, threshold)
        hits = ops.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
        misses = ops.sum(t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
        fas = ops.sum((1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas

    def preprocess(self, pred, target):
        """Apply data preprocessing based on configuration.

        Handles SEVIR-specific normalization and optional spatial pooling.

        Args:
            pred (Tensor): Raw model predictions
            target (Tensor): Raw ground truth data

        Returns:
            tuple[Tensor, Tensor]: Processed prediction and target tensors
        """
        if self.preprocess_type == "sevir":
            pred = process_data_dict_back(data_dict={"vil": pred.float()})["vil"]
            target = process_data_dict_back(data_dict={"vil": target.float()})["vil"]
        elif self.preprocess_type.startswith("sevir_pool"):
            pred = process_data_dict_back(data_dict={"vil": pred.float()})["vil"]
            target = process_data_dict_back(data_dict={"vil": target.float()})["vil"]
            self.pool_scale = int(re.search(r"\d+", self.preprocess_type).group())
            batch_size = target.shape[0]
            pred = rearrange(
                pred, f"{self.einops_layout} -> {self.einops_spatial_layout}"
            )
            target = rearrange(
                target, f"{self.einops_layout} -> {self.einops_spatial_layout}"
            )
            max_pool = nn.MaxPool2d(
                kernel_size=self.pool_scale, stride=self.pool_scale, pad_mode="pad"
            )
            pred = max_pool(pred)
            target = max_pool(target)
            pred = rearrange(
                pred,
                f"{self.einops_spatial_layout} -> {self.einops_layout}",
                N=batch_size,
            )
            target = rearrange(
                target,
                f"{self.einops_spatial_layout} -> {self.einops_layout}",
                N=batch_size,
            )
        else:
            raise NotImplementedError
        return pred, target

    def update(self, pred: Tensor, target: Tensor):
        """Update metric statistics with new batch of predictions."""
        pred, target = self.preprocess(pred, target)
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas = self.calc_seq_hits_misses_fas(pred, target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas

    def eval(self):
        """Compute final metric scores across all thresholds."""
        metrics_dict = {
            "pod": self.pod,
            "csi": self.csi,
            "sucr": self.sucr,
            "bias": self.bias,
            }
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        for metrics in self.metrics_list:
            if self.keep_seq_in_dim:
                score_avg = np.zeros((self.seq_in,))
            else:
                score_avg = 0
            scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            scores = scores.asnumpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_in_dim:
                    score = scores[i]
                else:
                    score = scores[i].item()
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2",):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret


def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    """Generate beta schedule for diffusion models.

    Supports linear, cosine, sqrt_linear and sqrt schedules.

    Args:
        schedule (str): Schedule type ("linear", "cosine", etc.)
        n_timestep (int): Number of time steps
        linear_start (float): Linear schedule start value
        linear_end (float): Linear schedule end value
        cosine_s (float): Cosine schedule shift parameter

    Returns:
        Tensor: Beta values for each time step
    """
    if schedule == "linear":
        betas = (
            mint.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=ms.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = ops.arange(n_timestep + 1, dtype=ms.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = ops.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = ops.linspace(linear_start, linear_end, n_timestep)
    elif schedule == "sqrt":
        betas = ops.linspace(linear_start, linear_end, n_timestep) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.asnumpy()


def extract_into_tensor(a, t, x_shape, batch_axis=0):
    """Extract tensor elements and reshape to match target dimensions."""
    batch_size = t.shape[0]
    out = a.gather_elements(-1, t)
    out_shape = [
        1,
    ] * len(x_shape)
    out_shape[batch_axis] = batch_size
    return out.reshape(out_shape)


def noise_like(shape):
    """Generate random noise tensor matching given shape."""
    return ops.randn(shape)


def default(val, d):
    """Return val if present, otherwise resolve default value."""
    if val is not None:
        return val
    return d() if isfunction(d) else d


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = ops.exp(
            -math.log(max_period)
            * ops.arange(start=0, end=half, dtype=ms.float32)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
        if dim % 2:
            embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for param in module.trainable_params():
        param.set_data(Zero()(shape=param.shape, dtype=param.dtype))
    return module


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    num_groups = min(32, channels)
    return nn.GroupNorm(num_groups, channels)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs, pad_mode="pad", has_bias=True)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs, pad_mode="pad", has_bias=True)
    return mint.nn.Conv3d(*args, **kwargs)



def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return mint.nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def round_to(dat, c):
    """round to"""
    return dat + (dat - dat % c) % c


def get_activation(act, inplace=False, **kwargs):
    """

    Parameters
    ----------
    act
        Name of the activation
    inplace
        Whether to perform inplace activation

    Returns
    -------
    activation_layer
        The activation
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == "leaky":
            negative_slope = kwargs.get("negative_slope", 0.1)
            return nn.LeakyReLU(negative_slope, inplace=inplace)
        if act == "identity":
            return nn.Identity()
        if act == "elu":
            return nn.ELU(inplace=inplace)
        if act == "gelu":
            return nn.GELU(approximate=False)
        if act == "relu":
            return nn.ReLU()
        if act == "sigmoid":
            return nn.Sigmoid()
        if act == "tanh":
            return nn.Tanh()
        if act in ('softrelu', 'softplus'):
            return ops.Softplus()
        if act == "softsign":
            return nn.Softsign()
        raise NotImplementedError('act="{}" is not supported. ')
    return act


def get_norm_layer(
        norm_type: str = "layer_norm",
        axis: int = -1,
        epsilon: float = 1e-5,
        in_channels: int = 0,
        **kwargs,
):
    """Get the normalization layer based on the provided type

    Parameters
    ----------
    norm_type
        The type of the layer normalization from ['layer_norm']
    axis
        The axis to normalize the
    epsilon
        The epsilon of the normalization layer
    in_channels
        Input channel

    Returns
    -------
    norm_layer
        The layer normalization layer
    """
    if isinstance(norm_type, str):
        if norm_type == "layer_norm":
            assert in_channels > 0
            assert axis == -1
            norm_layer = nn.LayerNorm(
                normalized_shape=[in_channels], epsilon=epsilon, **kwargs
            )
        else:
            raise NotImplementedError("norm_type={} is not supported".format(norm_type))
        return norm_layer
    if norm_type is None:
        return nn.Identity()
    raise NotImplementedError("The type of normalization must be str")


def generalize_padding(x, pad_t, pad_h, pad_w, padding_type, t_pad_left=False):
    """

    Parameters
    ----------
    x
        Shape (B, T, H, W, C)
    pad_t
    pad_h
    pad_w
    padding_type
    t_pad_left

    Returns
    -------
    out
        The result after padding the x. Shape will be (B, T + pad_t, H + pad_h, W + pad_w, C)
    """
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    assert padding_type in ["zeros", "ignore", "nearest"]
    _, t, h, w, _ = x.shape

    if padding_type == "nearest":
        return ops.interpolate(
            x.permute(0, 4, 1, 2, 3), size=(t + pad_t, h + pad_h, w + pad_w)
        ).permute(0, 2, 3, 4, 1)
    if t_pad_left:
        return ops.pad(x, (0, 0, 0, pad_w, 0, pad_h, pad_t, 0))
    return ops.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))


def generalize_unpadding(x, pad_t, pad_h, pad_w, padding_type):
    """Removes padding from a 5D tensor based on specified padding type and dimensions.

    Args:
        x (Tensor): Input tensor with shape (batch, time, height, width, channels).
        pad_t (int): Number of time steps to remove from the end.
        pad_h (int): Number of height units to remove from the end.
        pad_w (int): Number of width units to remove from the end.
        padding_type (str): Type of padding removal method ("zeros", "ignore", "nearest").

    Returns:
        Tensor: Processed tensor with padding removed according to specified method.

    Raises:
        AssertionError: If invalid padding_type is provided.
    """
    assert padding_type in ["zeros", "ignore", "nearest"]
    _, t, h, w, _ = x.shape
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type == "nearest":
        return ops.interpolate(
            x.permute(0, 4, 1, 2, 3), size=(t - pad_t, h - pad_h, w - pad_w)
        ).permute(0, 2, 3, 4, 1)
    return x[:, : (t - pad_t), : (h - pad_h), : (w - pad_w), :]


def _calculate_fan_in_and_fan_out(parameter):
    """Calculates fan_in and fan_out values for neural network weight initialization."""
    dimensions = parameter.ndim
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for parameter with fewer than 2 dimensions"
        )
    num_input_fmaps = parameter.shape[1]
    num_output_fmaps = parameter.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for s in parameter.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def apply_initialization(
        cell, linear_mode="0", conv_mode="0", norm_mode="0", embed_mode="0"
):
    """Applies parameter initialization strategies to neural network layers.

    Args:
        cell (nn.Cell): Neural network layer to initialize.
        linear_mode (str): Initialization mode for dense layers ("0", "1", "2").
        conv_mode (str): Initialization mode for convolutional layers ("0", "1", "2").
        norm_mode (str): Initialization mode for normalization layers ("0").
        embed_mode (str): Initialization mode for embedding layers ("0").

    Raises:
        NotImplementedError: If unsupported initialization mode is requested.
    """
    if isinstance(cell, nn.Dense):
        if linear_mode in ("0",):
            cell.weight.set_data(
                initializer(
                    HeNormal(mode="fan_in", nonlinearity="linear"),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
        elif linear_mode in ("1",):
            cell.weight.set_data(
                initializer.initializer(
                    HeNormal(mode="fan_out", nonlinearity="leaky_relu"),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
        elif linear_mode in ("2",):
            zeros_tensor = ops.zeros(cell.weight.shape, cell.weight.dtype)
            cell.weight.set_data(zeros_tensor)
        else:
            raise NotImplementedError
        if hasattr(cell, "bias") and cell.bias is not None:
            zeros_tensor = ops.zeros(cell.bias.shape, cell.bias.dtype)
            cell.bias.set_data(zeros_tensor)

    elif isinstance(
            cell, (nn.Conv2d, nn.Conv3d, nn.Conv2dTranspose, nn.Conv3dTranspose)
    ):
        if conv_mode in ("0",):
            cell.weight.set_data(
                initializer(
                    HeNormal(
                        negative_slope=math.sqrt(5), mode="fan_out", nonlinearity="relu"
                    ),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.has_bias:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    cell.bias.set_data(
                        initializer(Uniform(bound), cell.bias.shape, cell.bias.dtype)
                    )
        elif conv_mode in ("1",):
            cell.weight.set_data(
                initializer(
                    HeNormal(
                        mode="fan_out", nonlinearity="leaky_relu", negative_slope=0.1
                    ),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if hasattr(mcell, "bias") and mcell.bias is not None:
                cell.bias.set_data(
                    initializer(Zero(), cell.bias.shape, cell.bias.dtype)
                )
        elif conv_mode in ("2",):
            cell.weight.set_data(
                initializer(Zero(), cell.weight.shape, cell.weight.dtype)
            )
            if hasattr(m, "bias") and m.bias is not None:
                cell.bias.set_data(
                    initializer(Zero(), cell.bias.shape, cell.bias.dtype)
                )
        else:
            raise NotImplementedError

    elif isinstance(cell, nn.GroupNorm):
        if norm_mode in ("0",):
            if cell.gamma is not None:
                cell.gamma.set_data(
                    initializer(One(), cell.gamma.shape, cell.gamma.dtype)
                )
            if cell.beta is not None:
                cell.beta.set_data(
                    initializer(Zero(), cell.beta.shape, cell.beta.dtype)
                )
        else:
            raise NotImplementedError("Normalization mode not supported")
    elif isinstance(cell, nn.Embedding):
        if embed_mode == "0":
            cell.embedding_table.set_data(
                initializer(
                    TruncatedNormal(sigma=0.02),
                    cell.embedding_table.shape,
                    cell.embedding_table.dtype,
                )
            )
        else:
            raise NotImplementedError
    else:
        pass


def prepare_output_directory(base_config, device_id):
    """Creates/updates output directory for experiment results.

    Args:
        base_config (dict): Configuration dictionary containing directory paths.
        device_id (int): Device identifier for directory naming.

    Returns:
        str: Path to the created/updated output directory.

    Raises:
        OSError: If directory operations fail unexpectedly.
    """
    output_path = os.path.join(
        base_config["summary"]["summary_dir"], f"single_device{device_id}"
    )

    try:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            print(f"Cleared previous output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        print(f"Directory operation failed: {e}", exc_info=True)
        raise
    base_config["summary"]["summary_dir"] = output_path
    return output_path


def configure_logging_system(output_dir, config):
    """Sets up logging system for the application.

    Args:
        output_dir (str): Directory where logs should be stored.
        config (dict): Configuration dictionary containing experiment parameters.

    Returns:
        Logger: Configured logger instance.
    """
    logger = create_logger(path=os.path.join(output_dir, "results.log"))
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(config["summary"])
    return logger


def prepare_dataset(config, module):
    """Initializes and prepares the dataset for training/evaluation.

    Args:
        config (dict): Configuration dictionary with dataset parameters.
        SEVIRPLModule (Module): Data module class for dataset handling.

    Returns:
        tuple: (DataModule, total_num_steps) containing initialized data module and total training steps.

    Raises:
        ValueError: If configuration is not provided.
    """
    if config is not None:
        dataset_cfg = config["data"]
        total_batch_size = config["optim"]["total_batch_size"]
        micro_batch_size = config["optim"]["micro_batch_size"]
        max_epochs = config["optim"]["max_epochs"]
    else:
        raise ValueError("config is required but not provided")
    dm = module.get_sevir_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8,
    )
    dm.setup()
    total_num_steps = module.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    return dm, total_num_steps


def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    """Creates a learning rate warmup schedule as a lambda function.

    Args:
        warmup_steps (int): Number of steps for the warmup phase.
        min_lr_ratio (float): Minimum learning rate ratio at the start of training.

    Returns:
        function: Lambda function that calculates the warmup multiplier based on current step.
    """
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        return 1.0

    return ret_lambda


def get_loss_fn(loss: str = "l2") -> Callable:
    """
    Returns a loss function based on the provided loss type.

    Args:
    loss (str): Type of loss function. Default is "l2".

    Returns:
    Callable: A loss function corresponding to the provided loss type.
    """
    if loss in ("l2", "mse"):
        return nn.MSELoss()
    return nn.L1Loss()


def disabled_train(self):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def disable_train(model: nn.Cell):
    """
    Disable training to avoid error when used in pl.LightningModule
    """
    model.set_train(False)
    model.train = disabled_train
    return model


def layout_to_in_out_slice(layout, t_in, t_out=None):
    """layout_to_in_out_slice"""
    t_axis = layout.find("T")
    num_axes = len(layout)
    in_slice = [
        slice(None, None),
    ] * num_axes
    out_slice = deepcopy(in_slice)
    in_slice[t_axis] = slice(None, t_in)
    if t_out is None:
        out_slice[t_axis] = slice(t_in, None)
    else:
        out_slice[t_axis] = slice(t_in, t_in + t_out)
    return in_slice, out_slice


def parse_layout_shape(layout: str) -> Dict[str, Any]:
    r"""

    Parameters
    ----------
    layout: str
            e.g., "NTHWC", "NHWC".

    Returns
    -------
    ret:    Dict
    """
    batch_axis = layout.find("N")
    t_axis = layout.find("T")
    h_axis = layout.find("H")
    w_axis = layout.find("W")
    c_axis = layout.find("C")
    return {
        "batch_axis": batch_axis,
        "t_axis": t_axis,
        "h_axis": h_axis,
        "w_axis": w_axis,
        "c_axis": c_axis,
    }


def ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (np.ndarray): First input image (grayscale or single-channel), shape (H, W)
        img2 (np.ndarray): Second input image with identical shape to img1

    Returns:
        float: SSIM value between 0 (completely dissimilar) and 1 (perfect similarity)

    Notes:
        - Uses 11x11 Gaussian window with σ=1.5 for weighted filtering
        - Follows the standard SSIM formulation with constants c1=0.0001, c2=0.0009
        - Computes valid convolution regions (edges truncated by kernel size)
    """
    c1 = 0.01**2
    c2 = 0.03**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_ssim_function(img1, img2):
    """calculate ssim function"""
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        if img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    raise ValueError("Wrong input image dimensions.")




def calculate_ssim(videos1, videos2):
    """Calculate Structural Similarity Index (SSIM) between two video sequences across all timestamps.

    Args:
        videos1 (Tensor or np.ndarray): First video sequence with shape (batch_size, time_steps,
          height, width, channels)
        videos2 (Tensor or np.ndarray): Second video sequence with identical shape to videos1

    Returns:
        dict[int, float]: Dictionary where keys are timestamp indices and values are the mean SSIM values
                         across all batches for that timestamp

    Raises:
        AssertionError: If input video tensors have different shapes
    """
    ssim_results = []
    for video_num in range(videos1.shape[0]):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        ssim_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            img1 = video1[clip_timestamp]
            img2 = video2[clip_timestamp]
            ssim_results_of_a_video.append(calculate_ssim_function(img1, img2))
        ssim_results.append(ssim_results_of_a_video)
    ssim_results = np.array(ssim_results)
    ssim_score = {}
    for clip_timestamp in range(len(video1)):
        ssim_score[clip_timestamp] = np.mean(ssim_results[:, clip_timestamp])

    return ssim_score


def init_model(module, config, mode):
    """Initialize model with ckpt"""
    summary_params = config.get("summary")
    module.main_model.set_train(True)
    if mode != "train":
        summary_params["load_ckpt"] = "True"
        module.main_model.set_train(False)
    if summary_params["load_ckpt"]:
        params = ms.load_checkpoint(summary_params.get("ckpt_path"))
        ms.load_param_into_net(
            module.main_model, params
        )
    return module

def self_axial(input_shape):
    """Axial attention implementation from "Axial-Deeplab:
      Efficient Convolutional Neural Networks for Semantic Segmentation"
    Args:
        input_shape (tuple): Input tensor shape (T, H, W, C).
    Returns:
        tuple: Axial attention parameters with separate temporal/spatial cuboids.
    """
    t, h, w, _ = input_shape
    cuboid_size = [(t, 1, 1), (1, h, 1), (1, 1, w)]
    strategy = [("l", "l", "l"), ("l", "l", "l"), ("l", "l", "l")]
    shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size

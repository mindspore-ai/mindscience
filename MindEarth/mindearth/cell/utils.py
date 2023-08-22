# Copyright 2022 Huawei Technologies Co., Ltd
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
'''utils'''
import numpy as np
import mindspore.ops.operations as P
from mindspore import nn, ops, Parameter, numpy as msnp
from mindspore.common.initializer import initializer, Normal

__all__ = ['to_2tuple', 'unpatchify', 'patchify', 'get_2d_sin_cos_pos_embed',
           'pixel_shuffle', 'pixel_unshuffle', 'PixelShuffle', 'PixelUnshuffle',
           'SpectralNorm']


def to_2tuple(t):
    """
    Args:
        t (Union[int, tuple(int)]): The grid height and width.

    Returns:
        Same as input or a tuple as (t,t).

    """
    return t if isinstance(t, tuple) else (t, t)


def get_2d_sin_cos_pos_embed(embed_dim, grid_size):
    """
    Args:
        embed_dim (int): The output dimension for each position.
        grid_size (tuple(int)): The grid height and width.

    Returns:
        The numpy array with shape of (1, grid_height*grid_width, embed_dim)

    """
    grid_size = to_2tuple(grid_size)
    grid_height = np.arange(grid_size[0], dtype=np.float32)
    grid_width = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_width, grid_height)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = np.expand_dims(pos_embed, 0)
    return pos_embed


def get_2d_sin_cos_pos_embed_from_grid(embed_dim, grid):
    """
    use half of dimensions to encode grid_height

    Args:
        embed_dim (int): output dimension for each position.
        grid (int): a numpy array of positions to be encoded: size (M,).

    Returns:
        The numpy array with shape of (M/2, embed_dim)
    """
    emb_height = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_width = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_height, emb_width], axis=1)  # (H*W, D)
    return emb


def get_1d_sin_cos_pos_embed_from_grid(embed_dim, pos):
    """
    Args:
        embed_dim (int): output dimension for each position.
        pos (int): a numpy array of positions to be encoded: size (M,).

    Returns:
        The numpy array with shape of (M, embed_dim)
    """
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def patchify(label, patch_size=16):
    """
    Args:
        label (Union[int, float]): output dimension for each position.
        patch_size (int): The patch size of image. Default: 16.

    Returns:
        The numpy array with new shape of (H, W).
    """
    label_shape = label.shape
    label = np.reshape(label, (label_shape[0] // patch_size,
                               patch_size,
                               label_shape[1] // patch_size,
                               patch_size,
                               label_shape[2]))
    label = np.transpose(label, (0, 2, 1, 3, 4))
    label_new_shape = label.shape
    label = np.reshape(label, (label_new_shape[0] * label_new_shape[1],
                               label_new_shape[2] * label_new_shape[3] * label_new_shape[4]))
    return label


def unpatchify(labels, img_size=(192, 384), patch_size=16, nchw=False):
    """
    Args:
        labels (Union[int, float]): output dimension for each position.
        img_size (tuple(int)): Input image size. Default (192, 384).
        patch_size (int): The patch size of image. Default: 16.
        nchw (bool): If True, the unpatchify shape contains N, C, H, W.

    Returns:
        The tensor with shape of (N, H, W, C).
    """
    label_shape = labels.shape
    output_dim = label_shape[-1] // (patch_size * patch_size)
    labels = P.Reshape()(labels, (label_shape[0],
                                  img_size[0] // patch_size,
                                  img_size[1] // patch_size,
                                  patch_size,
                                  patch_size,
                                  output_dim))

    labels = P.Transpose()(labels, (0, 1, 3, 2, 4, 5))
    labels = P.Reshape()(labels, (label_shape[0],
                                  img_size[0],
                                  img_size[1],
                                  output_dim))
    if nchw:
        labels = P.Transpose()(labels, (0, 3, 1, 2))
    return labels


class SpectralNorm(nn.Cell):
    """Applies spectral normalization to a parameter in the given module.

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm.

    Args:
        module (nn.Cell): containing module.
        n_power_iterations (int): number of power iterations to calculate spectral norm.
        dim (int): dimension corresponding to number of outputs.
        eps (float): epsilon for numerical stability in calculating norms.

    Inputs:
        - **input** - The positional parameter of containing module.
        - **kwargs** - The keyword parameter of containing module.

    Outputs:
        The forward propagation of containing module.
    """
    def __init__(
            self,
            module,
            n_power_iterations: int = 1,
            dim: int = 0,
            eps: float = 1e-12
    ) -> None:
        super(SpectralNorm, self).__init__()
        self.parametrizations = module
        self.weight = module.weight
        self.use_weight_norm = True
        ndim = self.weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[-{ndim}, {ndim - 1}] but got {dim})")

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        self.l2_normalize = ops.L2Normalize(epsilon=self.eps)
        self.expand_dims = ops.ExpandDims()
        self.assign = P.Assign()
        if ndim > 1:
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix()

            h, w = weight_mat.shape
            u = initializer(Normal(1.0, 0), [h]).init_data()
            v = initializer(Normal(1.0, 0), [w]).init_data()
            self._u = Parameter(self.l2_normalize(u), requires_grad=False)
            self._v = Parameter(self.l2_normalize(v), requires_grad=False)
            self._u, self._v = self._power_method(weight_mat, 15)

    def construct(self, *inputs, **kwargs):
        """SpectralNorm forward function"""
        if self.weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            self.l2_normalize(self.weight)
            self.assign(self.parametrizations.weight, self.weight)
        else:
            weight_mat = self._reshape_weight_to_matrix()
            if self.use_weight_norm:
                self._u, self._v = self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.copy()
            v = self._v.copy()

            sigma = ops.tensor_dot(u, msnp.multi_dot([weight_mat, self.expand_dims(v, -1)]), 1)

            self.assign(self.parametrizations.weight, self.weight / sigma)

        return self.parametrizations(*inputs, **kwargs)

    def remove_weight_norm(self):
        self.use_weight_norm = False

    def _power_method(self, weight_mat, n_power_iterations):
        for _ in range(n_power_iterations):
            self._u = self.l2_normalize(msnp.multi_dot([weight_mat, self.expand_dims(self._v, -1)]).flatten())
            self._v = self.l2_normalize(msnp.multi_dot([weight_mat.T, self.expand_dims(self._u, -1)]).flatten())
        return self._u, self._v

    def _reshape_weight_to_matrix(self):
        # Precondition
        if self.dim != 0:
            # permute dim to front
            input_perm = [d for d in range(self.weight.dim()) if d != self.dim]
            input_perm.insert(0, self.dim)

            self.weight = ops.transpose(self.weight, input_perm)

        return self.weight.reshape(self.weight.shape[0], -1)


def pixel_shuffle(x, upscale_factor):
    r"""
    Applies a pixel_shuffle operation over an input signal composed of several input planes. This is useful for
    implementiong efficient sub-pixel convolution with a stride of :math:`1/r`. For more details, refer to
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the `x` is of shape :math:`(*, C \times r^2, H, W)` , and the output is of shape
    :math:`(*, C, H \times r, W \times r)`, where `r` is an upscale factor and `*` is zero or more batch dimensions.

    Args:
        x (Tensor): Tensor of shape :math:`(*, C \times r^2, H, W)` . The dimension of `x` is larger than 2, and the
            length of third to last dimension can be divisible by `upscale_factor` squared.
        upscale_factor (int):  factor to increase spatial resolution by, and is a positive integer.

    Returns:
        - **output** (Tensor) - Tensor of shape :math:`(*, C, H \times r, W \times r)` .

    Raises:
        ValueError: If `upscale_factor` is not a positive integer.
        ValueError: If the length of third to last dimension is not divisible by `upscale_factor` squared.
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    idx = x.shape
    length = len(idx)
    if length < 3:
        raise TypeError(f"For pixel_shuffle, the dimension of `x` should be larger than 2, but got {length}.")
    pre = idx[:-3]
    c, h, w = idx[-3:]
    if c % upscale_factor ** 2 != 0:
        raise ValueError("For 'pixel_shuffle', the length of third to last dimension is not divisible"
                         "by `upscale_factor` squared.")
    c = c // upscale_factor ** 2
    input_perm = (pre + (c, upscale_factor, upscale_factor, h, w))
    reshape = ops.Reshape()
    x = reshape(x, input_perm)
    input_perm = [i for i in range(length - 2)]
    input_perm = input_perm + [length, length - 2, length + 1, length - 1]
    input_perm = tuple(input_perm)
    transpose = ops.Transpose()
    x = transpose(x, input_perm)
    x = reshape(x, (pre + (c, upscale_factor * h, upscale_factor * w)))
    return x


class PixelShuffle(nn.Cell):
    r"""
    Applies a pixelshuffle operation over an input signal composed of several input planes. This is useful for
    implementiong efficient sub-pixel convolution with a stride of :math:`1/r`. For more details, refer to
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the input is of shape :math:`(*, C \times r^2, H, W)` , and the output is of shape
    :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor and * is zero or more batch dimensions.

    Args:
        upscale_factor (int):  factor to increase spatial resolution by, and is a positive integer.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, C \times r^2, H, W)` . The dimension of `x` is larger than 2, and
          the length of third to last dimension can be divisible by `upscale_factor` squared.

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(*, C, H \times r, W \times r)` .

    Raises:
        ValueError: If `upscale_factor` is not a positive integer.
        ValueError: If the length of third to last dimension of `x` is not divisible by `upscale_factor` squared.
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def construct(self, x):
        return pixel_shuffle(x, self.upscale_factor)


def pixel_unshuffle(x, downscale_factor):
    r"""
    Applies a pixel_unshuffle operation over an input signal composed of several input planes. For more details, refer
    to `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the input is of shape :math:`(*, C, H \times r, W \times r)` , and the output is of shape
    :math:`(*, C \times r^2, H, W)` , where `r` is a downscale factor and `*` is zero or more batch dimensions.

    Args:
        x (Tensor): Tensor of shape :math:`(*, C, H \times r, W \times r)` . The dimension of `x` is larger than 2,
            and the length of second to last dimension or last dimension can be divisible by `downscale_factor` .
        downscale_factor (int): factor to decrease spatial resolution by, and is a positive integer.

    Returns:
        - **output** (Tensor) - Tensor of shape :math:`(*, C \times r^2, H, W)` .

    Raises:
        ValueError: If `downscale_factor` is not a positive integer.
        ValueError: If the length of second to last dimension or last dimension is not divisible by `downscale_factor` .
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    idx = x.shape
    length = len(idx)
    if length < 3:
        raise TypeError(f"For pixel_unshuffle, the dimension of `x` should be larger than 2, but got {length}.")
    pre = idx[:-3]
    c, h, w = idx[-3:]
    if h % downscale_factor != 0 or w % downscale_factor != 0:
        raise ValueError("For 'pixel_unshuffle', the length of second to last 2 dimension should be divisible "
                         "by downscale_factor.")
    h = h // downscale_factor
    w = w // downscale_factor
    input_perm = (pre + (c, h, downscale_factor, w, downscale_factor))
    reshape = ops.Reshape()
    x = reshape(x, input_perm)
    input_perm = [i for i in range(length - 2)]
    input_perm = input_perm + [length - 1, length + 1, length - 2, length]
    input_perm = tuple(input_perm)
    transpose = ops.Transpose()
    x = transpose(x, input_perm)
    x = reshape(x, (pre + (c * downscale_factor * downscale_factor, h, w)))
    return x


class PixelUnshuffle(nn.Cell):
    r"""
    Applies a pixelunshuffle operation over an input signal composed of several input planes. For more details, refer to
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    <https://arxiv.org/abs/1609.05158>`_ .

    Typically, the input is of shape :math:`(*, C, H \times r, W \times r)` , and the output is of shape
    :math:`(*, C \times r^2, H, W)` , where r is a downscale factor and * is zero or more batch dimensions.

    Args:
        downscale_factor (int): factor to decrease spatial resolution by, and is a positive integer.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, C, H \times r, W \times r)` . The dimension of `x` is larger than
          2, and the length of second to last dimension or last dimension can be divisible by `downscale_factor` .

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(*, C \times r^2, H, W)` .

    Raises:
        ValueError: If `downscale_factor` is not a positive integer.
        ValueError: If the length of second to last dimension or last dimension is not divisible by `downscale_factor` .
        TypeError: If the dimension of `x` is less than 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def construct(self, x):
        return pixel_unshuffle(x, self.downscale_factor)

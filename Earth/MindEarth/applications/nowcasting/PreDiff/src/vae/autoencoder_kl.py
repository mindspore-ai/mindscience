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
"vae base class"
from typing import Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint

from src.utils import DiagonalGaussianDistribution
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


class Encoder(nn.Cell):
    """
    A class representing an encoder network for image encoding.

    Args:
        in_channels (int): Number of input image channels (default: 3)
        out_channels (int): Number of output image channels (default: 3)
        down_block_types (tuple): Types of downsampling blocks (default: ("DownEncoderBlock2D",))
        block_out_channels (tuple): Output channels for each downsampling block (default: (64,))
        layers_per_block (int): Number of layers per downsampling block (default: 2)
        norm_num_groups (int): Number of groups for group normalization (default: 32)
        act_fn (str): Activation function type (default: "silu")
        double_z (bool): Whether to double output channels (default: True)

    Returns:
        None
    """
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
            double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            has_bias=True,
            pad_mode="pad",
        )

        self.mid_block = None
        self.down_blocks = nn.CellList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            conv_out_channels,
            3,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

    def construct(self, x):
        """
        Forward pass through the encoder network.

        Args:
            x (Tensor): Input image tensor

        Returns:
            Tensor: Encoded output tensor
        """

        sample = self.conv_in(x)

        for _, down_block in enumerate(self.down_blocks):
            sample = down_block(sample)

        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder(nn.Cell):
    """
    Decoder class for the decoding process in image generation tasks.

    Args:
        in_channels (int): Number of input channels, defaults to 3.
        out_channels (int): Number of output channels, defaults to 3.
        up_block_types (tuple): Types of upsample blocks, defaults to ("UpDecoderBlock2D",).
        block_out_channels (tuple): Output channels for each block, defaults to (64,).
        layers_per_block (int): Number of layers per block, defaults to 2.
        norm_num_groups (int): Number of groups for normalization, defaults to 32.
        act_fn (str): Activation function type, defaults to "silu".

    Attributes:
        layers_per_block (int): Number of layers per block.
        conv_in (nn.Conv2d): Input convolution layer.
        mid_block (UNetMidBlock2D): Middle block.
        up_blocks (nn.CellList): List of upsample blocks.
        conv_norm_out (nn.GroupNorm): Output normalization layer.
        conv_act (nn.SiLU): Output activation layer.
        conv_out (nn.Conv2d): Output convolution layer.
    """
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
        )

        self.mid_block = None
        self.up_blocks = nn.CellList([])
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            3,
            padding=1,
            pad_mode="pad",
            has_bias=True,
        )

    def construct(self, z):
        """
        Builds the decoder computation graph.

        Args:
            z (Tensor): Input tensor.

        Returns:
            Tensor: Decoded output tensor.
        """
        sample = z
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class AutoencoderKL(nn.Cell):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
    """

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
            up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
            block_out_channels: Tuple[int] = (64,),
            layers_per_block: int = 1,
            act_fn: str = "silu",
            latent_channels: int = 4,
            norm_num_groups: int = 32,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = mint.nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = mint.nn.Conv2d(latent_channels, latent_channels, 1)
        self.use_slicing = False

    def encode(self, x: ms.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def _decode(self, z: ms.Tensor) -> ms.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously invoked, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def decode(self, z: ms.Tensor) -> ms.Tensor:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = ops.cat(decoded_slices)
        else:
            decoded = self._decode(z)
        return decoded

    def construct(
            self,
            sample: ms.Tensor,
            sample_posterior: bool = False,
            return_posterior: bool = False,
    ) -> ms.Tensor:
        r"""
        Args:
            sample (`ms.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_posterior (`bool`, *optional*, defaults to `False`):
                Whether or not to return `posterior` along with `dec` for calculating the training loss.
        """

        posterior = self.encode(sample)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if return_posterior:
            return dec, posterior
        return dec

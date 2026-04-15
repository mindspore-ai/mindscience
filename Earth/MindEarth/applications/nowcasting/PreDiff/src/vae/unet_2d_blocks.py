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
"UNetMidBlock2D"
import math
from typing import Optional

import mindspore as ms
from mindspore import nn, ops

from .resnet import Downsample2D, ResnetBlock2D, Upsample2D


def get_down_block(
        down_block_type,
        num_layers,
        in_channels,
        out_channels,
        temb_channels,
        add_downsample,
        resnet_eps,
        resnet_act_fn,
        attn_num_head_channels,
        resnet_groups=None,
        cross_attention_dim=None,
        downsample_padding=None,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        resnet_time_scale_shift="default",
):
    """set down_block"""
    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
        up_block_type,
        num_layers,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels,
        add_upsample,
        resnet_eps,
        resnet_act_fn,
        attn_num_head_channels,
        resnet_groups=None,
        cross_attention_dim=None,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        resnet_time_scale_shift="default",
):
    """set up_block"""
    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class AttentionBlock(nn.Cell):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case. Uses three q, k, v linear layers to compute attention.

    Args:
        channels (int): The number of channels in the input and output.
        num_head_channels (int, optional): The number of channels in each attention head. If None, uses a single head.
        norm_num_groups (int, optional): Number of groups for group normalization (default: 32).
        rescale_output_factor (float, optional): Factor to rescale the output (default: 1.0).
        eps (float, optional): Epsilon value for group normalization (default: 1e-5).

    Attributes:
        num_heads (int): Calculated number of attention heads based on `num_head_channels`.
        group_norm (nn.GroupNorm): Group normalization layer.
        query/key/value (nn.Dense): Linear layers for query, key, and value projections.
        proj_attn (nn.Dense): Final projection layer after attention computation.
    """

    def __init__(
            self,
            channels: int,
            num_head_channels: Optional[int] = None,
            norm_num_groups: int = 32,
            rescale_output_factor: float = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = (
            channels // num_head_channels if num_head_channels is not None else 1
        )
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(
            num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True
        )

        # define q,k,v as linear layers
        self.query = nn.Dense(channels, channels)
        self.key = nn.Dense(channels, channels)
        self.value = nn.Dense(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Dense(channels, channels, 1)

        self._use_memory_efficient_attention_xformers = False
        self._attention_op = None
        self.softmax_op = ops.Softmax(axis=-1)

    def reshape_heads_to_batch_dim(self, tensor):
        """
        Reshape tensor to split attention heads into batch dimension for efficient computation."
        """
        batch_size, seq_in, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size, seq_in, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_in, dim // head_size
        )
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        """
        Reverse reshape_heads_to_batch_dim to merge batch dimension back into heads."
        """
        batch_size, seq_in, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_in, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_in, dim * head_size
        )
        return tensor

    def construct(self, hidden_states):
        """Compute multi-head self-attention."""
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).swapaxes(
            1, 2
        )
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)

        shape = (query_proj.shape[0], query_proj.shape[1], key_proj.shape[1])
        uninitialized_tensor = ms.numpy.empty(shape=shape, dtype=query_proj.dtype)
        attention_scores = ops.baddbmm(
            uninitialized_tensor,
            query_proj,
            key_proj.swapaxes(-1, -2),
            beta=0,
            alpha=scale,
        )
        attention_probs = self.softmax_op(attention_scores.astype(ms.float32)).type(
            attention_scores.dtype
        )

        hidden_states = ops.bmm(attention_probs, value_proj)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = hidden_states.swapaxes(-1, -2).reshape(
            batch, channel, height, width
        )
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class UNetMidBlock2D(nn.Cell):
    """
        UNet middle block for 2D architectures.
    """
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            add_attention: bool = True,
            attn_num_head_channels=1,
            output_scale_factor=1.0,
    ):
        """
        UNet middle block for 2D architectures. Contains residual blocks and optional attention layers.

        Args:
            in_channels (int): Number of input channels.
            temb_channels (int): Number of time embedding channels.
            dropout (float): Dropout probability (default: 0.0).
            num_layers (int): Number of residual blocks (default: 1).
            resnet_eps (float): Epsilon for ResNet normalization (default: 1e-6).
            resnet_time_scale_shift (str): Time scale shift method for ResNet ("default" or "scale_shift").
            resnet_act_fn (str): Activation function for ResNet layers (default: "swish").
            resnet_groups (int): Number of groups for group normalization in ResNet.
            resnet_pre_norm (bool): Whether to use pre-normalization in ResNet.
            add_attention (bool): Whether to include attention blocks (default: True).
            attn_num_head_channels (int): Number of channels per attention head.
            output_scale_factor (float): Scaling factor for output (default: 1.0).

        Attributes:
            resnets (nn.CellList): List of ResNet blocks.
            attentions (nn.CellList): List of attention blocks (or None if disabled).
        """
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention
        self.num_layers = num_layers
        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(self.num_layers):
            if self.add_attention:
                attentions.append(
                    AttentionBlock(
                        in_channels,
                        num_head_channels=attn_num_head_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)

    def construct(self, hidden_states, temb=None):
        """
        Forward pass through the middle block.

        Args:
            hidden_states (Tensor): Input tensor.
            temb (Tensor, optional): Time embedding tensor.

        Returns:
            Tensor: Output tensor after processing through all blocks.
        """

        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DownEncoderBlock2D(nn.Cell):
    """
    Downsample block for encoder part of UNet.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_downsample=True,
            downsample_padding=1,
    ):
        """
        Downsample block for encoder part of UNet. Contains residual blocks and optional downsampling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float): Dropout probability (default: 0.0).
            num_layers (int): Number of residual blocks (default: 1).
            resnet_eps (float): Epsilon for ResNet normalization (default: 1e-6).
            resnet_time_scale_shift (str): Time scale shift method for ResNet ("default" or "scale_shift").
            resnet_act_fn (str): Activation function for ResNet layers (default: "swish").
            resnet_groups (int): Number of groups for group normalization in ResNet.
            resnet_pre_norm (bool): Whether to use pre-normalization in ResNet.
            output_scale_factor (float): Scaling factor for output (default: 1.0).
            add_downsample (bool): Whether to include downsampling layer (default: True).
            downsample_padding (int): Padding for downsampling convolution (default: 1).

        Attributes:
            resnets (nn.CellList): List of ResNet blocks.
            downsamplers (nn.CellList or None): Downsampling layer if enabled.
        """
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.CellList(resnets)

        if add_downsample:
            self.downsamplers = nn.CellList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

    def construct(self, hidden_states):
        """
        Forward pass through the downsample block.

        Args:
            hidden_states (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after residual blocks and optional downsampling.
        """
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Cell):
    """
    Upsample block for decoder part of UNet.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor=1.0,
            add_upsample=True,
    ):
        """
        Upsample block for decoder part of UNet. Contains residual blocks and optional upsampling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float): Dropout probability (default: 0.0).
            num_layers (int): Number of residual blocks (default: 1).
            resnet_eps (float): Epsilon for ResNet normalization (default: 1e-6).
            resnet_time_scale_shift (str): Time scale shift method for ResNet ("default" or "scale_shift").
            resnet_act_fn (str): Activation function for ResNet layers (default: "swish").
            resnet_groups (int): Number of groups for group normalization in ResNet.
            resnet_pre_norm (bool): Whether to use pre-normalization in ResNet.
            output_scale_factor (float): Scaling factor for output (default: 1.0).
            add_upsample (bool): Whether to include upsampling layer (default: True).

        Attributes:
            resnets (nn.CellList): List of ResNet blocks.
            upsamplers (nn.CellList or None): Upsampling layer if enabled.
        """
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.CellList(resnets)

        if add_upsample:
            self.upsamplers = nn.CellList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def construct(self, hidden_states):
        """Forward pass through the upsample block."""
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

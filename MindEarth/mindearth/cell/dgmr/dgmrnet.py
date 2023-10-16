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
"""Dgmr base class"""
from mindspore import ops
from mindspore.nn import Cell

from .dgmr import ContextConditioningStack, LatentConditioningStack, Sampler, SpatialDiscriminator, TemporalDiscriminator


class DgmrDiscriminator(Cell):
    r"""
    The Dgmr Discriminator is based on Temporal Discriminator and Spatial Discriminator,
    which contains deep residual block. The details can be found in `Skilful precipitation
    nowcasting using deep generative models of radar <https://arxiv.org/abs/2104.00954>`_.

    Args:
        in_channels (int): The channels of input frame.
        num_spatial_frames (int): 8 Random frames out of lead times.
        conv_type (str): convolutional layer's type.

    Inputs:
         - **x** (Tensor) - Tensor of shape :math:`(2, frames\_size, channels, height\_size, width\_size)`.

    Outputs:
        Tensor, the output of the DgmrDiscriminator.

         - **output** (Tensor) - Tensor of shape :math:`(2, 2, 1)`

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import ops, Tensor
        >>> from mindspore.nn import Cell
        >>> from mindearth.cell.dgmr.dgmrnet import DgmrDiscriminator
        >>> real_and_generator = np.random.rand(2, 22, 1, 256, 256).astype(np.float32)
        >>> net = DgmrDiscriminator(in_channels=1, num_spatial_frames=8, con_type="standard")
        >>> out = net(Tensor(real_and_generator, ms.float32))
        >>> print(out.shape)
        (2, 2, 1)
    """
    def __init__(
            self,
            in_channels=1,
            num_spatial_frames=8,
            conv_type="standard"
    ):
        super().__init__()

        self.spatial_discriminator = SpatialDiscriminator(
            in_channels=in_channels, num_timesteps=num_spatial_frames, conv_type=conv_type
        )

        self.temporal_discriminator = TemporalDiscriminator(
            in_channels=in_channels, conv_type=conv_type
        )

    def construct(self, x):
        """Dgmr discriminator forward function."""
        spatial_loss = self.spatial_discriminator(x)
        temporal_loss = self.temporal_discriminator(x)

        concat_op = ops.Concat(axis=1)

        output = concat_op((spatial_loss, temporal_loss))
        return output


class DgmrGenerator(Cell):
    r"""
    The Dgmr Generator is based on Conditional_Stack, Latent_Stack, Upsample_Stack and ConvGRU,
    which contain deep residual block. The details can be found in `Skilful precipitation
    nowcasting using deep generative models of radar <https://arxiv.org/abs/2104.00954>`_.

    Args:
        forecast_steps (int): The steps of forecast frames.
        in_channels (int): The channels of input frame.
        out_channels (int): Shape of the output predictions, generally should be same as the input shape.
        conv_type (str): The convolution type.
        latent_channels (int): Latent channels according to network.
        context_channels (int): Context channels according to network.
        generation_steps (int): Number of generation steps to use in forward pass,
                                in paper is 6 and the best is chosen for the loss,
                                this results in huge amounts of GPU memory though,
                                so less might work better for training.

    Inputs:
         - **x** (Tensor) - Tensor of shape :math:`(batch\_size, input\_frames,
           out_channels, height\_size, width\_size)`.

    Outputs:
        Tensor，the output of Dgmr Generator。

         - **output** (Tensor) - Tensor of shape :math:`(batch\_size, output\_frames,
           out_channels, height\_size, width\_size)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import ops, Tensor
        >>> from mindspore.nn import Cell
        >>> from mindearth.cell.dgmr.dgmrnet import DgmrGenerator
        >>> input_frames = np.random.rand(1, 4, 1, 256, 256).astype(np.float32)
        >>> net = DgmrGenerator(
        >>>         forecast_steps = 18,
        >>>         in_channels = 1,
        >>>         out_channels = 256,
        >>>         conv_type = "standard",
        >>>         latent_channels = 768,
        >>>         context_channels = 384,
        >>>         generation_steps = 1
        >>>     )
        >>> out = net(Tensor(input_frames, ms.float32))
        >>> print(out.shape)
        (1, 18, 1, 256, 256)
    """
    def __init__(
            self,
            forecast_steps=18,
            in_channels=1,
            out_channels=256,
            conv_type="standard",
            latent_channels=768,
            context_channels=384,
            generation_steps=1
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.in_channels = in_channels
        self.generation_steps = generation_steps

        self.conditioning_stack = ContextConditioningStack(
            in_channels=in_channels,
            conv_type=conv_type,
            out_channels=self.context_channels,
        )

        self.latent_stack = LatentConditioningStack(
            shape=(8 * self.in_channels, out_channels // 32, out_channels // 32, 1),
            out_channels=self.latent_channels,
        )
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )

    def construct(self, x):
        """Dgmr generator forward function."""
        conditioning_states = self.conditioning_stack(x)
        latent_dim = self.latent_stack(x)
        output = self.sampler(conditioning_states, latent_dim)
        return output

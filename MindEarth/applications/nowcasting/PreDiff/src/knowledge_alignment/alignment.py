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
"NoisyCuboidTransformerEncoder"
import math
import numpy as np

import mindspore as ms
from mindspore import nn, ops, mint
from mindspore.common.initializer import TruncatedNormal

from src.utils import (
    conv_nd,
    zero_module,
    timestep_embedding,
    apply_initialization,
    round_to,
    self_axial
)
from src.diffusion import (
    PatchMerging3D,
    PosEmbed,
    StackCuboidSelfAttentionBlock,
    TimeEmbedLayer,
    TimeEmbedResBlock,
)


class QKVAttention(nn.Cell):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def construct(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        q_transposed = ops.transpose(
            (q * scale).view(bs * self.n_heads, ch, length), (0, 2, 1)
        )
        k_reshaped = (k * scale).view(bs * self.n_heads, ch, length)
        weight = ops.BatchMatMul()(q_transposed, k_reshaped)
        weight = nn.Softmax(axis=-1)(weight.float()).type(weight.dtype)
        weight_transposed = ops.transpose(weight, (0, 2, 1))
        v_reshaped = v.reshape(bs * self.n_heads, ch, length)
        a = ops.BatchMatMul()(v_reshaped, weight_transposed)
        return a.reshape(bs, -1, length)


class AttentionPool3d(nn.Cell):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            data_dim: int,
            embed_dim: int,
            num_heads: int,
            output_dim: int = None,
            init_mode: str = "0",
    ):
        r"""
        Parameters
        ----------
        data_dim:   int
            e.g. T*H*W if data is 3D
        embed_dim:  int
            input data channels
        num_heads:  int
        output_dim: int
        """
        super().__init__()
        self.positional_embedding = ms.Parameter(
            ops.randn(embed_dim, data_dim + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = num_heads
        self.attention = QKVAttention(self.num_heads)
        self.init_mode = init_mode

    def construct(self, x):
        r"""

        Parameters
        ----------
        x:  ms.Tensor
            layout = "NCTHW"

        Returns
        -------
        ret:    ms.Tensor
            layout = "NC"
        """
        b, c, _ = x.shape
        x = x.reshape(b, c, -1)
        x = mint.cat([x.mean(axis=-1, keep_dims=True), x], dim=-1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

    def reset_parameters(self):
        '''set parameters'''
        apply_initialization(self.qkv_proj, conv_mode="0")
        apply_initialization(self.c_proj, conv_mode=self.init_mode)


class NoisyCuboidTransformerEncoder(nn.Cell):
    r"""
    Half U-Net style CuboidTransformerEncoder that parametrizes `U(z_t, t, ...)`.
    It takes `x_t`, `t` as input.
    The conditioning can be concatenated to the input like the U-Net in FVD paper.

    For each block, we apply the StackCuboidSelfAttention. The final block state is read out by a pooling layer.

        x --> attn --> downscale --> ... --> poll --> out

    Besides, we insert the embeddings of the timesteps `t` before each cuboid attention blocks.
    """

    def __init__(
            self,
            input_shape=None,
            out_channels=1,
            base_units=128,
            block_units=None,
            scale_alpha=1.0,
            depth=None,
            downsample=2,
            downsample_type="patch_merge",
            use_attn_pattern=None,
            block_cuboid_size=None,
            block_cuboid_strategy=None,
            block_cuboid_shift_size=None,
            num_heads=4,
            attn_drop=0.0,
            proj_drop=0.0,
            ffn_drop=0.0,
            ffn_activation="gelu",
            gated_ffn=False,
            norm_layer="layer_norm",
            use_inter_ffn=True,
            hierarchical_pos_embed=False,
            padding_type="zeros",
            use_relative_pos=True,
            self_attn_use_final_proj=True,
            # global vectors
            num_global_vectors=0,
            use_global_vector_ffn=True,
            use_global_self_attn=False,
            separate_global_qkv=False,
            global_dim_ratio=1,
            # initialization
            attn_linear_init_mode="0",
            ffn_linear_init_mode="0",
            ffn2_linear_init_mode="2",
            attn_proj_linear_init_mode="2",
            conv_init_mode="0",
            down_linear_init_mode="0",
            global_proj_linear_init_mode="2",
            norm_init_mode="0",
            # timestep embedding for diffusion
            time_embed_channels_mult=4,
            time_embed_use_scale_shift_norm=False,
            time_embed_dropout=0.0,
            # readout
            pool: str = "attention",
            readout_seq: bool = True,
            t_out: int = None,
    ):
        super().__init__()
        # initialization mode
        self.attn_linear_init_mode = attn_linear_init_mode
        self.ffn_linear_init_mode = ffn_linear_init_mode
        self.ffn2_linear_init_mode = ffn2_linear_init_mode
        self.attn_proj_linear_init_mode = attn_proj_linear_init_mode
        self.conv_init_mode = conv_init_mode
        self.down_linear_init_mode = down_linear_init_mode
        self.global_proj_linear_init_mode = global_proj_linear_init_mode
        self.norm_init_mode = norm_init_mode

        self.input_shape = input_shape
        self.out_channels = out_channels
        self.num_blocks = len(depth)
        self.depth = depth
        self.base_units = base_units
        self.scale_alpha = scale_alpha
        self.downsample = downsample
        self.downsample_type = downsample_type
        if not isinstance(downsample, (tuple, list)):
            downsample = (1, downsample, downsample)
        if block_units is None:
            block_units = [
                round_to(base_units * int((max(downsample) ** scale_alpha) ** i), 4)
                for i in range(self.num_blocks)
            ]
        else:
            assert len(block_units) == self.num_blocks and block_units[0] == base_units
        self.block_units = block_units
        self.hierarchical_pos_embed = hierarchical_pos_embed
        self.num_global_vectors = num_global_vectors
        use_global_vector = num_global_vectors > 0
        self.use_global_vector = use_global_vector
        self.global_dim_ratio = global_dim_ratio
        self.use_global_vector_ffn = use_global_vector_ffn

        self.time_embed_channels_mult = time_embed_channels_mult
        self.time_embed_channels = self.block_units[0] * time_embed_channels_mult
        self.time_embed_use_scale_shift_norm = time_embed_use_scale_shift_norm
        self.time_embed_dropout = time_embed_dropout
        self.pool = pool
        self.readout_seq = readout_seq
        self.t_out = t_out

        if self.use_global_vector:
            self.init_global_vectors = ms.Parameter(
                mint.zeros((self.num_global_vectors, global_dim_ratio * base_units))
            )

        _, h_in, w_in, _ = input_shape
        self.first_proj = TimeEmbedResBlock(
            channels=input_shape[-1],
            emb_channels=None,
            dropout=proj_drop,
            out_channels=self.base_units,
            use_conv=False,
            use_embed=False,
            use_scale_shift_norm=False,
            dims=3,
            up=False,
            down=False,
        )
        self.pos_embed = PosEmbed(
            embed_dim=base_units,
            max_t=input_shape[0],
            max_h=h_in,
            max_w=w_in,
        )

        # diffusion time embed
        self.time_embed = TimeEmbedLayer(
            base_channels=self.block_units[0],
            time_embed_channels=self.time_embed_channels,
        )
        if self.num_blocks > 1:
            if downsample_type == "patch_merge":
                self.downsample_layers = nn.CellList(
                    [
                        PatchMerging3D(
                            dim=self.block_units[i],
                            downsample=downsample,
                            padding_type=padding_type,
                            out_dim=self.block_units[i + 1],
                            linear_init_mode=down_linear_init_mode,
                            norm_init_mode=norm_init_mode,
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            else:
                raise NotImplementedError
            if self.use_global_vector:
                self.down_layer_global_proj = nn.CellList(
                    [
                        mint.nn.Linear(
                            in_features=global_dim_ratio * self.block_units[i],
                            out_features=global_dim_ratio * self.block_units[i + 1],
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            if self.hierarchical_pos_embed:
                self.down_hierarchical_pos_embed_l = nn.CellList(
                    [
                        PosEmbed(
                            embed_dim=self.block_units[i],
                            max_t=self.mem_shapes[i][0],
                            max_h=self.mem_shapes[i][1],
                            max_w=self.mem_shapes[i][2],
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )

        if use_attn_pattern:
            block_attn_patterns = self.depth
            block_cuboid_size = []
            block_cuboid_strategy = []
            block_cuboid_shift_size = []
            for idx, _ in enumerate(block_attn_patterns):
                cuboid_size, strategy, shift_size = self_axial(self.mem_shapes[idx])
                block_cuboid_size.append(cuboid_size)
                block_cuboid_strategy.append(strategy)
                block_cuboid_shift_size.append(shift_size)
        else:
            if not isinstance(block_cuboid_size[0][0], (list, tuple)):
                block_cuboid_size = [block_cuboid_size for _ in range(self.num_blocks)]
            else:
                assert (
                    len(block_cuboid_size) == self.num_blocks
                ), f"Incorrect input format! Received block_cuboid_size={block_cuboid_size}"

            if not isinstance(block_cuboid_strategy[0][0], (list, tuple)):
                block_cuboid_strategy = [
                    block_cuboid_strategy for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_cuboid_strategy) == self.num_blocks
                ), f"Incorrect input format! Received block_strategy={block_cuboid_strategy}"

            if not isinstance(block_cuboid_shift_size[0][0], (list, tuple)):
                block_cuboid_shift_size = [
                    block_cuboid_shift_size for _ in range(self.num_blocks)
                ]
            else:
                assert (
                    len(block_cuboid_shift_size) == self.num_blocks
                ), f"Incorrect input format! Received block_shift_size={block_cuboid_shift_size}"
        self.block_cuboid_size = block_cuboid_size
        self.block_cuboid_strategy = block_cuboid_strategy
        self.block_cuboid_shift_size = block_cuboid_shift_size

        # cuboid self attention blocks
        down_self_blocks = []
        # ResBlocks that incorporate `time_embed`
        down_time_embed_blocks = []
        for i in range(self.num_blocks):
            down_time_embed_blocks.append(
                TimeEmbedResBlock(
                    channels=self.mem_shapes[i][-1],
                    emb_channels=self.time_embed_channels,
                    dropout=self.time_embed_dropout,
                    out_channels=self.mem_shapes[i][-1],
                    use_conv=False,
                    use_embed=True,
                    use_scale_shift_norm=self.time_embed_use_scale_shift_norm,
                    dims=3,
                    up=False,
                    down=False,
                )
            )

            ele_depth = depth[i]

            stack_cuboid_blocks = [
                StackCuboidSelfAttentionBlock(
                    dim=self.mem_shapes[i][-1],
                    num_heads=num_heads,
                    block_cuboid_size=block_cuboid_size[i],
                    block_strategy=block_cuboid_strategy[i],
                    block_shift_size=block_cuboid_shift_size[i],
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    ffn_drop=ffn_drop,
                    activation=ffn_activation,
                    gated_ffn=gated_ffn,
                    norm_layer=norm_layer,
                    use_inter_ffn=use_inter_ffn,
                    padding_type=padding_type,
                    use_global_vector=use_global_vector,
                    use_global_vector_ffn=use_global_vector_ffn,
                    use_global_self_attn=use_global_self_attn,
                    separate_global_qkv=separate_global_qkv,
                    global_dim_ratio=global_dim_ratio,
                    use_relative_pos=use_relative_pos,
                    use_final_proj=self_attn_use_final_proj,
                    # initialization
                    attn_linear_init_mode=attn_linear_init_mode,
                    ffn_linear_init_mode=ffn_linear_init_mode,
                    ffn2_linear_init_mode=ffn2_linear_init_mode,
                    attn_proj_linear_init_mode=attn_proj_linear_init_mode,
                    norm_init_mode=norm_init_mode,
                )
                for _ in range(ele_depth)
            ]
            down_self_blocks.append(nn.CellList(stack_cuboid_blocks))

        self.down_self_blocks = nn.CellList(down_self_blocks)
        self.down_time_embed_blocks = nn.CellList(down_time_embed_blocks)

        out_shape = self.mem_shapes[-1]
        cuboid_out_channels = out_shape[-1]
        if pool == "adaptive":
            self.out = nn.SequentialCell(
                nn.GroupNorm(min(cuboid_out_channels, 32), cuboid_out_channels),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(2, cuboid_out_channels, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            if readout_seq:
                data_dim = np.prod(out_shape[1:-1]).item() + num_global_vectors
            else:
                data_dim = np.prod(out_shape[:-1]).item() + num_global_vectors
            self.out = nn.SequentialCell(
                nn.GroupNorm(min(cuboid_out_channels, 32), cuboid_out_channels),
                nn.SiLU(),
                AttentionPool3d(
                    data_dim,
                    cuboid_out_channels,
                    num_heads,
                    out_channels,
                    init_mode="0",
                ),
            )
        elif pool == "spatial":
            self.out = nn.SequentialCell(
                mint.nn.Linear(self._feature_size, 2048),
                mint.nn.ReLU(),
                mint.nn.Linear(2048, out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.SequentialCell(
                mint.nn.Linear(self._feature_size, 2048),
                mint.nn.GroupNorm(2048, 2048),
                nn.SiLU(),
                mint.nn.Linear(2048, out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

        self.reset_parameters()

    def reset_parameters(self):
        """set parameters"""
        if self.num_global_vectors > 0:
            TruncatedNormal(self.init_global_vectors, sigma=0.02)
        self.first_proj.reset_parameters()
        self.pos_embed.reset_parameters()
        # inner U-Net
        for block in self.down_self_blocks:
            for m in block:
                m.reset_parameters()
        for m in self.down_time_embed_blocks:
            m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.downsample_layers:
                m.reset_parameters()
            if self.use_global_vector:
                apply_initialization(
                    self.down_layer_global_proj,
                    linear_mode=self.global_proj_linear_init_mode,
                )
        if self.hierarchical_pos_embed:
            for m in self.down_hierarchical_pos_embed_l:
                m.reset_parameters()
        if self.pool == "attention":
            apply_initialization(self.out[0], norm_mode=self.norm_init_mode)
            self.out[2].reset_parameters()
        else:
            raise NotImplementedError

    def transpose_and_first_proj(self, x, batch_size):
        """transpose and first_proj"""
        x = x.transpose(0, 4, 1, 2, 3)
        x = self.first_proj(x)
        x = x.transpose(0, 2, 3, 4, 1)
        if self.use_global_vector:
            global_vectors = self.init_global_vectors.broadcast_to(
                batch_size,
                self.num_global_vectors,
                self.global_dim_ratio * self.base_units,
            )
            return x, global_vectors
        return x, None

    @property
    def mem_shapes(self):
        """Get the shape of the output memory based on the input shape. This can be used for constructing the decoder.

        Returns
        -------
        mem_shapes
            A list of shapes of the output memory
        """
        inner_data_shape = tuple(self.input_shape)[:3] + (self.base_units,)
        if self.num_blocks == 1:
            return [inner_data_shape]
        mem_shapes = [inner_data_shape]
        curr_shape = inner_data_shape
        for down_layer in self.downsample_layers:
            curr_shape = down_layer.get_out_shape(curr_shape)
            mem_shapes.append(curr_shape)
        return mem_shapes

    def construct(self, x, t):
        """
        Forward pass through the NoisyCuboidTransformerEncoder.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, seq_in, H, W, C).
        - t (Tensor): Timestep tensor.

        Returns:
        - Tensor: Output tensor after processing through the encoder.
        """
        batch_size, seq_in, _, _, _ = x.shape
        x, global_vectors = self.transpose_and_first_proj(x, batch_size)
        x = self.pos_embed(x)
        t_emb = self.time_embed(timestep_embedding(t, self.block_units[0]))
        for i in range(self.num_blocks):
            if i > 0:
                x = self.downsample_layers[i - 1](x)
                if self.hierarchical_pos_embed:
                    x = self.down_hierarchical_pos_embed_l[i - 1](x)
                if self.use_global_vector:
                    global_vectors = self.down_layer_global_proj[i - 1](global_vectors)
            for idx in range(self.depth[i]):
                x = x.transpose(0, 4, 1, 2, 3)
                x = self.down_time_embed_blocks[i](x, t_emb)
                x = x.transpose(0, 2, 3, 4, 1)
                if self.use_global_vector:
                    x, global_vectors = self.down_self_blocks[i][idx](x, global_vectors)
                else:
                    x = self.down_self_blocks[i][idx](x)

        if self.readout_seq:
            if self.t_out is not None:
                seq_in = self.t_out
                start_idx = x.shape[1] - self.t_out
                x = x[:, start_idx:, ...]
            out = x.transpose(0, 1, 4, 2, 3)
            b_out, t_out, c_out, h_out, w_out = out.shape
            out = out.reshape(b_out * t_out, c_out, h_out * w_out)
            if self.num_global_vectors > 0:
                out_global = global_vectors.tile((seq_in, 1, 1))
                out_global = out_global.transpose(0, 2, 1)
                out = mint.cat([out, out_global], dim=2)
            out = self.out(out)
            out = out.reshape(batch_size, seq_in, -1)
        else:
            out = x.transpose(0, 4, 1, 2, 3)
            b_out, c_out, t_out, h_out, w_out = out.shape
            out = out.reshape(b_out, c_out, t_out * h_out * w_out)
            if self.num_global_vectors > 0:
                out_global = global_vectors.transpose(0, 2, 1)
                out = mint.cat([out, out_global], dim=2)
            out = self.out(out)
        return out

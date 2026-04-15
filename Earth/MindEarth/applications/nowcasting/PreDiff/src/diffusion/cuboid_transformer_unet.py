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
"CuboidTransformerUNet base class"
from mindspore import ops, nn, Parameter
import mindspore.common.initializer as initializer

from src.utils import timestep_embedding, apply_initialization, round_to, self_axial
from .time_embed import TimeEmbedLayer, TimeEmbedResBlock
from .cuboid_transformer import (
    PosEmbed,
    Upsample3DLayer,
    PatchMerging3D,
    StackCuboidSelfAttentionBlock,
)


class CuboidTransformerUNet(nn.Cell):
    r"""
    U-Net style CuboidTransformer that parametrizes `p(x_{t-1}|x_t)`.
    It takes `x_t`, `t` as input.
    The conditioning can be concatenated to the input like the U-Net in FVD paper.

    For each block, we apply the StackCuboidSelfAttention in U-Net style

        x --> attn --> downscale --> ... --> z --> attn --> upscale --> ... --> out

    Besides, we insert the embeddings of the timesteps `t` before each cuboid attention blocks.
    """

    def __init__(
            self,
            input_shape=None,
            target_shape=None,
            base_units=256,
            block_units=None,
            scale_alpha=1.0,
            depth=None,
            downsample=2,
            downsample_type="patch_merge",
            upsample_type="upsample",
            upsample_kernel_size=3,
            use_attn_pattern=True,
            block_cuboid_size=None,
            block_cuboid_strategy=None,
            block_cuboid_shift_size=None,
            num_heads=4,
            attn_drop=0.0,
            proj_drop=0.0,
            ffn_drop=0.0,
            ffn_activation="leaky",
            gated_ffn=False,
            norm_layer="layer_norm",
            use_inter_ffn=True,
            hierarchical_pos_embed=False,
            padding_type="ignore",
            use_relative_pos=True,
            self_attn_use_final_proj=True,
            # global vectors
            num_global_vectors=False,
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
            unet_res_connect=True,
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
        self.target_shape = target_shape
        self.num_blocks = len(depth)
        self.depth = depth
        self.base_units = base_units
        self.scale_alpha = scale_alpha
        self.downsample = downsample
        self.downsample_type = downsample_type
        self.upsample_type = upsample_type
        self.upsample_kernel_size = upsample_kernel_size
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
        if global_dim_ratio != 1:
            assert (
                separate_global_qkv is True
            ), f"Setting global_dim_ratio != 1 requires separate_global_qkv == True."
        self.global_dim_ratio = global_dim_ratio
        self.use_global_vector_ffn = use_global_vector_ffn

        self.time_embed_channels_mult = time_embed_channels_mult
        self.time_embed_channels = self.block_units[0] * time_embed_channels_mult
        self.time_embed_use_scale_shift_norm = time_embed_use_scale_shift_norm
        self.time_embed_dropout = time_embed_dropout
        self.unet_res_connect = unet_res_connect

        if self.use_global_vector:
            self.init_global_vectors = Parameter(
                ops.zeros((self.num_global_vectors, global_dim_ratio * base_units))
            )

        t_in, h_in, w_in, c_in = input_shape
        t_out, h_out, w_out, c_out = target_shape
        assert h_in == h_out and w_in == w_out and c_in == c_out
        self.t_in = t_in
        self.t_out = t_out
        self.first_proj = TimeEmbedResBlock(
            channels=self.data_shape[-1],
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
            max_t=self.data_shape[0],
            max_h=h_in,
            max_w=w_in,
        )

        # diffusion time embed
        self.time_embed = TimeEmbedLayer(
            base_channels=self.block_units[0],
            time_embed_channels=self.time_embed_channels,
        )
        # # inner U-Net
        if self.num_blocks > 1:
            # Construct downsampling layers
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
                        nn.Dense(
                            in_channels=global_dim_ratio * self.block_units[i],
                            out_channels=global_dim_ratio * self.block_units[i + 1],
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            # Construct upsampling layers
            if self.upsample_type == "upsample":
                self.upsample_layers = nn.CellList(
                    [
                        Upsample3DLayer(
                            dim=self.mem_shapes[i + 1][-1],
                            out_dim=self.mem_shapes[i][-1],
                            target_size=self.mem_shapes[i][:3],
                            kernel_size=upsample_kernel_size,
                            conv_init_mode=conv_init_mode,
                        )
                        for i in range(self.num_blocks - 1)
                    ]
                )
            else:
                raise NotImplementedError
            if self.use_global_vector:
                self.up_layer_global_proj = nn.CellList(
                    [
                        nn.Dense(
                            in_channels=global_dim_ratio * self.block_units[i + 1],
                            out_channels=global_dim_ratio * self.block_units[i],
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
                self.up_hierarchical_pos_embed_l = nn.CellList(
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
        up_self_blocks = []
        # ResBlocks that incorporate `time_embed`
        down_time_embed_blocks = []
        up_time_embed_blocks = []
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

            up_time_embed_blocks.append(
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
            up_self_blocks.append(nn.CellList(stack_cuboid_blocks))
        self.down_self_blocks = nn.CellList(down_self_blocks)
        self.up_self_blocks = nn.CellList(up_self_blocks)
        self.down_time_embed_blocks = nn.CellList(down_time_embed_blocks)
        self.up_time_embed_blocks = nn.CellList(up_time_embed_blocks)
        self.final_proj = nn.Dense(self.base_units, c_out)

        self.reset_parameters()

    def reset_parameters(self):
        '''init parameters'''
        if self.num_global_vectors > 0:
            initializer.TruncatedNormal(self.init_global_vectors, sigma=0.02)
        self.first_proj.reset_parameters()
        apply_initialization(self.final_proj, linear_mode="2")
        self.pos_embed.reset_parameters()
        for block in self.down_self_blocks:
            for m in block:
                m.reset_parameters()
        for m in self.down_time_embed_blocks:
            m.reset_parameters()
        for block in self.up_self_blocks:
            for m in block:
                m.reset_parameters()
        for m in self.up_time_embed_blocks:
            m.reset_parameters()
        if self.num_blocks > 1:
            for m in self.downsample_layers:
                m.reset_parameters()
            for m in self.upsample_layers:
                m.reset_parameters()
            if self.use_global_vector:
                apply_initialization(
                    self.down_layer_global_proj,
                    linear_mode=self.global_proj_linear_init_mode,
                )
                apply_initialization(
                    self.up_layer_global_proj,
                    linear_mode=self.global_proj_linear_init_mode,
                )
        if self.hierarchical_pos_embed:
            for m in self.down_hierarchical_pos_embed_l:
                m.reset_parameters()
            for m in self.up_hierarchical_pos_embed_l:
                m.reset_parameters()

    @property
    def data_shape(self):
        '''set datashape'''
        if not hasattr(self, "_data_shape"):
            t_in, h_in, w_in, c_in = self.input_shape
            t_out, h_out, w_out, c_out = self.target_shape
            assert h_in == h_out and w_in == w_out and c_in == c_out
            self._data_shape = (
                t_in + t_out,
                h_in,
                w_in,
                c_in + 1,
            )
        return self._data_shape

    @property
    def mem_shapes(self):
        """Get the shape of the output memory based on the input shape. This can be used for constructing the decoder.

        Returns
        -------
        mem_shapes
            A list of shapes of the output memory
        """
        inner_data_shape = tuple(self.data_shape)[:3] + (self.base_units,)
        if self.num_blocks == 1:
            return [inner_data_shape]
        mem_shapes = [inner_data_shape]
        curr_shape = inner_data_shape
        for down_layer in self.downsample_layers:
            curr_shape = down_layer.get_out_shape(curr_shape)
            mem_shapes.append(curr_shape)
        return mem_shapes

    def construct(self, x, t, cond):
        """

        Parameters
        ----------
        x:  mindspore.Tensor
            Shape (B, t_out, H, W, C)
        t:  mindspore.Tensor
            Shape (B, )
        cond:   mindspore.Tensor
            Shape (B, t_in, H, W, C)
        verbose:    bool

        Returns
        -------
        out:    mindspore.Tensor
            Shape (B, T, H, W, C)
        """

        x = ops.cat([cond, x], axis=1)
        obs_indicator = ops.ones_like(x[..., :1])
        obs_indicator[:, self.t_in :, ...] = 0.0
        x = ops.cat([x, obs_indicator], axis=-1)
        x = x.transpose((0, 4, 1, 2, 3))
        x = self.first_proj(x)
        x = x.transpose((0, 2, 3, 4, 1))
        x = self.pos_embed(x)
        # inner U-Net
        t_emb = self.time_embed(timestep_embedding(t, self.block_units[0]))
        if self.unet_res_connect:
            res_connect_l = []
        for i in range(self.num_blocks):
            # Downample
            if i > 0:
                x = self.downsample_layers[i - 1](x)
            for idx in range(self.depth[i]):
                x = x.transpose((0, 4, 1, 2, 3))
                x = self.down_time_embed_blocks[i](x, t_emb)
                x = x.transpose((0, 2, 3, 4, 1))
                x = self.down_self_blocks[i][idx](x)
            if self.unet_res_connect and i < self.num_blocks - 1:
                res_connect_l.append(x)

        for i in range(self.num_blocks - 1, -1, -1):
            if self.unet_res_connect and i < self.num_blocks - 1:
                x = x + res_connect_l[i]
            for idx in range(self.depth[i]):
                x = x.transpose((0, 4, 1, 2, 3))
                x = self.up_time_embed_blocks[i](x, t_emb)
                x = x.transpose((0, 2, 3, 4, 1))
                x = self.up_self_blocks[i][idx](x)
            if i > 0:
                x = self.upsample_layers[i - 1](x)
        x = self.final_proj(x[:, self.t_in :, ...])
        return x

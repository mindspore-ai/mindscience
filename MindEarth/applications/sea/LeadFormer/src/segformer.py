# right 2020-2022 Huawei Technologies Co., Ltd
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
"""net"""
from functools import partial

import mindspore.nn as nn
import mindspore.ops as ops

from .backbone import MixVisionTransformer, init_weights


class MLP(nn.Cell):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Dense(input_dim, embed_dim)
        self.apply(init_weights)

    def construct(self, x):
        x = x.flatten(start_dim=2).swapaxes(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Cell):
    """
    A convolutional module including convolution, batch normalization, and activation.

    Args:
        c1 (int): Number of input channels
        c2 (int): Number of output channels
        k (int): Kernel size, default=1
        s (int): Stride, default=1
        p (int): Padding size, default=0
        g (int): Number of groups for grouped convolution, default=1
        act (Union[bool, nn.Cell]): Activation function. True for ReLU, False/None for Identity,
                                   or directly provide a activation module
    """
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            pad_mode="pad",
            padding=p,
            has_bias=False,
            group=g,
        )
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.97)
        self.act = (
            nn.ReLU()
            if act is True
            else (act if isinstance(act, nn.Cell) else nn.Identity())
        )

    def construct(self, x):
        """Standard forward pass: Conv -> BN -> Activation"""
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Cell):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(
            self,
            num_classes=20,
            in_channels=None,
            embedding_dim=768,
            dropout_ratio=0.1,
    ):
        super().__init__()
        in_channels = in_channels if in_channels is not None else [64, 128, 320, 512]
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(
            embedding_dim, num_classes, kernel_size=1, has_bias=True
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_v4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_v3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_v2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_v1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse_v = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred_v = nn.Conv2d(embedding_dim, 2, kernel_size=1, has_bias=True)
        self.dropout_v = nn.Dropout2d(dropout_ratio)

    def _upsample(self, inputs, linears, fuse, pred, dropout):
        """upsample"""
        c1, c2, c3, c4 = inputs
        b = c4.shape[0]
        proj4 = linears[3](c4).permute(0, 2, 1).reshape(b, -1, c4.shape[2], c4.shape[3])
        proj3 = linears[2](c3).permute(0, 2, 1).reshape(b, -1, c3.shape[2], c3.shape[3])
        proj2 = linears[1](c2).permute(0, 2, 1).reshape(b, -1, c2.shape[2], c2.shape[3])
        proj1 = linears[0](c1).permute(0, 2, 1).reshape(b, -1, c1.shape[2], c1.shape[3])

        up4 = ops.interpolate(proj4, size=c1.shape[2:], mode="bilinear", align_corners=False)
        up3 = ops.interpolate(proj3, size=c1.shape[2:], mode="bilinear", align_corners=False)
        up2 = ops.interpolate(proj2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        fused = fuse(ops.cat((up4, up3, up2, proj1), axis=1))
        out = pred(dropout(fused))
        return out

    def construct(self, inputs):
        """construct"""
        context_output = self._upsample(
            inputs,
            [self.linear_c1, self.linear_c2, self.linear_c3, self.linear_c4],
            self.linear_fuse,
            self.linear_pred,
            self.dropout,
        )
        vision_output = self._upsample(
            inputs,
            [self.linear_v1, self.linear_v2, self.linear_v3, self.linear_v4],
            self.linear_fuse_v,
            self.linear_pred_v,
            self.dropout_v,
        )
        return context_output, vision_output


class SegFormer(nn.Cell):
    """Simple and Efficient Semantic Segmentation Framework"""
    def __init__(
            self,
            in_channels=None,
            num_classes=21,
            embedding_dim=256,
            num_heads=None,
            mlp_ratios=None,
            depths=None,
            sr_ratios=None,
    ):
        self.in_channels = in_channels if in_channels is not None else [64, 128, 320, 512]
        self.num_heads = num_heads if num_heads is not None else [1, 2, 5, 8]
        self.mlp_ratios = mlp_ratios if mlp_ratios is not None else [4, 4, 4, 4]
        self.depths = depths if depths is not None else [2, 2, 2, 2]
        self.sr_ratios = sr_ratios if sr_ratios is not None else [8, 4, 2, 1]
        super().__init__()
        self.backbone = MixVisionTransformer(
            embed_dims=self.in_channels,
            num_heads=self.num_heads,
            mlp_ratios=self.mlp_ratios,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            depths=self.depths,
            sr_ratios=self.sr_ratios,
            drop_rate=0.0,
            drop_path_rate=0.1,
        )
        self.decode_head = SegFormerHead(num_classes, self.in_channels, embedding_dim)

    def construct(self, inputs):
        h, w = inputs.shape[2], inputs.shape[3]

        x = self.backbone(inputs)
        x, v = self.decode_head(x)

        x = ops.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        v = ops.interpolate(v, size=(h, w), mode="bilinear", align_corners=True)
        return x, v

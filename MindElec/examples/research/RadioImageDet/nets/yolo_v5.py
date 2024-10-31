# Copyright 2024 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""Construct a YOLOv5 model"""
import mindspore as ms
import mindspore.nn as nn

from nets.cspdarknet_v5 import C3, Conv, CSPDarknet


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Cell):
    """Construct a YOLOv5 model"""
    def __init__(self, anchors_mask, num_classes, phi):
        super(YoloBody, self).__init__()
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33,}
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25,}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.backbone = CSPDarknet(base_channels, base_depth)

        self.upsample = nn.Upsample(
            scale_factor=2.0,
            mode="nearest",
            recompute_scale_factor=True)

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3(
            base_channels * 16,
            base_channels * 8,
            base_depth,
            shortcut=False)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3(
            base_channels * 8,
            base_channels * 4,
            base_depth,
            shortcut=False)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3(
            base_channels * 8,
            base_channels * 8,
            base_depth,
            shortcut=False)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3(
            base_channels * 16,
            base_channels * 16,
            base_depth,
            shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 +
        # num_classes)
        self.yolo_head_p3 = nn.Conv2d(
            base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 +
        # num_classes)
        self.yolo_head_p4 = nn.Conv2d(
            base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 +
        # num_classes)
        self.yolo_head_p5 = nn.Conv2d(
            base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def construct(self, x):
        """Construct"""
        feat1, feat2, feat3 = self.backbone(x)
        p5 = self.conv_for_feat3(feat3)
        p5_upsample = self.upsample(p5)
        p4 = ms.ops.cat([p5_upsample, feat2], 1)
        p4 = self.conv3_for_upsample1(p4)
        p4 = self.conv_for_feat2(p4)
        p4_upsample = self.upsample(p4)
        p3 = ms.ops.cat([p4_upsample, feat1], 1)
        p3 = self.conv3_for_upsample2(p3)
        p3_downsample = self.down_sample1(p3)
        p4 = ms.ops.cat([p3_downsample, p4], 1)
        p4 = self.conv3_for_downsample1(p4)
        p4_downsample = self.down_sample2(p4)
        p5 = ms.ops.cat([p4_downsample, p5], 1)
        p5 = self.conv3_for_downsample2(p5)

        out2 = self.yolo_head_p3(p3)
        out1 = self.yolo_head_p4(p4)
        out0 = self.yolo_head_p5(p5)
        return out0, out1, out2

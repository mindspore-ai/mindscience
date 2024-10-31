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
"""Construct a YOLOv2 model"""
import mindspore.nn as nn

from nets.cspdarknet_v2 import CSPDarknet


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Cell):
    """Construct a YOLOv2 model"""
    def __init__(self, anchors_mask, num_classes, phi):
        super(YoloBody, self).__init__()
        depth_dict = {'s': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33,}
        width_dict = {'s': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25,}
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.backbone = CSPDarknet(base_channels, base_depth)

        self.yolo_head = nn.Conv2d(
            base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def construct(self, x):
        #  backbone
        feat = self.backbone(x)
        out = self.yolo_head(feat)
        return out

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
"""Dataloader for YOLOv2"""
import cv2
import numpy as np
from PIL import Image

from utils.utils import cvt_color, preprocess_input


class YoloDataset():
    """Dataloader for YOLOv2"""
    def __init__(self, annotation_lines, input_shape,
                 num_classes, anchors, anchors_mask, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.train = train
        self.length = len(self.annotation_lines)

        self.bbox_attrs = 5 + num_classes
        self.threshold = 4

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image, box = self.get_random_data(
            self.annotation_lines[index], self.input_shape, random=self.train)
        image = np.transpose(
            preprocess_input(
                np.array(
                    image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # ---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        y_true = self.get_target(box)
        return image, box, y_true

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape,
                        jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """get random data"""
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvt_color(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(','))))
                       for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # discard invalid box
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        new_ar = iw / ih * self.rand(1 - jitter,
                                     1 + jitter) / self.rand(1 - jitter,
                                                             1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge(
            (cv2.LUT(
                hue, lut_hue), cv2.LUT(
                sat, lut_sat), cv2.LUT(
                val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        """merge bboxes"""
        merge_bbox = []
        conditions = [
            lambda x1, y1, x2, y2: y1 <= cuty and x1 <= cutx,
            lambda x1, y1, x2, y2: y2 >= cuty and x1 <= cutx,
            lambda x1, y1, x2, y2: y2 >= cuty and x2 >= cutx,
            lambda x1, y1, x2, y2: y1 <= cuty and x2 >= cutx
        ]

        adjust_coords = [
            lambda x1, y1, x2, y2: (x1, y1, min(x2, cutx), min(y2, cuty)),
            lambda x1, y1, x2, y2: (x1, max(y1, cuty), min(x2, cutx), y2),
            lambda x1, y1, x2, y2: (max(x1, cutx), max(y1, cuty), x2, y2),
            lambda x1, y1, x2, y2: (max(x1, cutx), y1, x2, min(y2, cuty))
        ]

        for i, boxes in enumerate(bboxes):
            condition = conditions[i]
            adjust = adjust_coords[i]
            for box in boxes:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if not condition(x1, y1, x2, y2):
                    continue

                x1, y1, x2, y2 = adjust(x1, y1, x2, y2)
                merge_bbox.append([x1, y1, x2, y2, box[-1]])

        return merge_bbox

    def get_near_points(self, x, y, i, j):
        """get near points"""
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, targets):
        """get target"""
        num_layers = len(self.anchors_mask)

        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape //
                       {0: 32, 1: 16, 2: 8, 3: 4}[l] for l in range(num_layers)]
        y_true = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs),
                           dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32')
                          for l in range(num_layers)]

        if len(targets) == 0:
            return y_true

        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]
            anchors = np.array(self.anchors) / {0: 32, 1: 16, 2: 8, 3: 4}[l]

            batch_target = np.zeros_like(targets)
            batch_target[:, [0, 2]] = targets[:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[:, [1, 3]] * in_h
            batch_target[:, 4] = targets[:, 4]
            ratios_of_gt_anchors = np.expand_dims(
                batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(
                anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios = np.concatenate(
                [ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
            max_ratios = np.max(ratios, axis=-1)

            for t, ratio in enumerate(max_ratios):
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))

                    offsets = self.get_near_points(
                        batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j,
                                                 local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue

                        c = int(batch_target[t, 4])

                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]

        return y_true[0]


# DataLoader中collate_fn使用
def yolo_dataset_collate(image, boxes, y_true, _):
    images = {"images": image}
    bboxes = {"bboxes": boxes}
    y_trues = {"y_trues": y_true}

    return images, bboxes, y_trues

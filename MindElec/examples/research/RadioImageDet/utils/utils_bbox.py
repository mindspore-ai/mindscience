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
"""utils for bbox detection"""
import mindspore as ms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class DecodeBox():
    """Decode the box"""
    def __init__(self, anchors_list, num_classes, input_shape, anchors_mask_list=None):
        super(DecodeBox, self).__init__()
        if anchors_mask_list is None:
            anchors_mask_list = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = anchors_list
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask_list

    def decode_box(self, inputs):
        """Decode the box"""
        outputs = []
        for i, input_list in enumerate(inputs):
            batch_size = input_list.shape[0]
            input_height = input_list.shape[2]
            input_width = input_list.shape[3]

            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            scaled_anchors = [(anchor_width / stride_w,
                               anchor_height / stride_h) for anchor_width,
                              anchor_height in self.anchors[self.anchors_mask[i]]]

            prediction = input_list.view(batch_size, len(self.anchors_mask[i]),
                                         self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2)

            x = ms.ops.sigmoid(prediction[..., 0])
            y = ms.ops.sigmoid(prediction[..., 1])
            w = ms.ops.sigmoid(prediction[..., 2])
            h = ms.ops.sigmoid(prediction[..., 3])
            conf = ms.ops.sigmoid(prediction[..., 4])
            pred_cls = ms.ops.sigmoid(prediction[..., 5:])

            floattensor = ms.float32
            longtensor = ms.int64

            grid_x = ms.ops.linspace(0, input_width - 1, input_width).tile((input_height, 1)).tile(
                (batch_size * len(self.anchors_mask[i]), 1, 1)).view(x.shape).type(floattensor)
            grid_y = ms.ops.linspace(0, input_height - 1, input_height).tile((input_width, 1)).t().tile(
                (batch_size * len(self.anchors_mask[i]), 1, 1)).view(y.shape).type(floattensor)

            anchor_w = ms.Tensor(
                scaled_anchors,
                floattensor).index_select(
                    1,
                    ms.Tensor(
                        [0],
                        longtensor))
            anchor_h = ms.Tensor(
                scaled_anchors,
                floattensor).index_select(
                    1,
                    ms.Tensor(
                        [1],
                        longtensor))
            anchor_w = anchor_w.tile((batch_size, 1)).tile(
                (1, 1, input_height * input_width)).view(w.shape)
            anchor_h = anchor_h.tile((batch_size, 1)).tile(
                (1, 1, input_height * input_width)).view(h.shape)

            pred_boxes = ms.ops.zeros_like(prediction[..., :4])
            pred_boxes[..., 0] = ms.ops.add(
                ms.ops.subtract(ms.ops.mul(x, 2.), 0.5), grid_x)
            pred_boxes[..., 1] = ms.ops.add(
                ms.ops.subtract(ms.ops.mul(y, 2.), 0.5), grid_y)
            pred_boxes[..., 2] = ms.ops.mul(
                ms.ops.pow(ms.ops.mul(w, 2), 2), anchor_w)
            pred_boxes[..., 3] = ms.ops.mul(
                ms.ops.pow(ms.ops.mul(h, 2), 2), anchor_h)

            scales = ms.Tensor(
                [input_width, input_height, input_width, input_height]).type(floattensor)
            output = ms.ops.cat((pred_boxes.view(batch_size, -1, 4) / scales,
                                 conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh,
                           input_shape, image_shape, letterbox_image):
        """generate correction boxes"""
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            new_shape = np.round(
                image_shape *
                np.min(
                    input_shape /
                    image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                                box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape,
                            image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        """non maximum suppression"""
        box_corner = ms.ops.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            class_conf, class_pred = ms.ops.max(
                image_pred[:, 5:5 + num_classes], 1, keepdims=True)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf[:, 0]
                         >= conf_thres).squeeze()  # (25200,)

            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if image_pred.shape[0] == 0:
                continue
            detections = ms.ops.cat(
                (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            unique_labels, _ = ms.ops.unique(detections[:, -1])

            # for c in unique_labels:
            for j in range(unique_labels.shape[0]):
                c = unique_labels[j]
                detections_class = detections[detections[:, -1] == c]

                a = detections_class[:, :4]
                b = ms.ops.mul(detections_class[:, 4], detections_class[:, 5]).view(
                    detections_class.shape[0], 1)
                box_with_score_m = ms.ops.cat((a, b), axis=-1)
                _, _, mask = ms.ops.NMSWithMask(
                    nms_thres)(box_with_score_m)
                max_detections = detections_class[mask]

                output[i] = max_detections if output[i] is None else ms.ops.cat(
                    (output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].asnumpy()
                box_xy, box_wh = (
                    output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(
                    box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


if __name__ == "__main__":
    def get_anchors_and_decode(
            input_list, input_shape, anchors_list, anchors_mask_list, num_classes):
        """get anchors and decode"""
        # -----------------------------------------------#
        #   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
        # -----------------------------------------------#
        batch_size = input_list.size(0)
        input_height = input_list.size(2)
        input_width = input_list.size(3)

        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width
        scaled_anchors = [(anchor_width / stride_w,
                           anchor_height / stride_h) for anchor_width,
                          anchor_height in anchors_list[anchors_mask_list[2]]]

        prediction = input_list.view(batch_size, len(anchors_mask_list[2]),
                                     num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        x = ms.ops.sigmoid(prediction[..., 0])
        y = ms.ops.sigmoid(prediction[..., 1])
        w = ms.ops.sigmoid(prediction[..., 2])
        h = ms.ops.sigmoid(prediction[..., 3])

        floattensor = ms.float32
        longtensor = ms.int64

        grid_x = ms.ops.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * len(anchors_mask_list[2]), 1, 1).view(x.shape).type(floattensor)
        grid_y = ms.ops.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * len(anchors_mask_list[2]), 1, 1).view(y.shape).type(floattensor)

        anchor_w = floattensor(scaled_anchors).index_select(1, longtensor([0]))
        anchor_h = floattensor(scaled_anchors).index_select(1, longtensor([1]))
        anchor_w = anchor_w.repeat(
            batch_size,
            1).repeat(
                1,
                1,
                input_height *
                input_width).view(
                    w.shape)
        anchor_h = anchor_h.repeat(
            batch_size,
            1).repeat(
                1,
                1,
                input_height *
                input_width).view(
                    h.shape)

        pred_boxes = floattensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5

        box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32
        box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32
        grid_x = grid_x.cpu().numpy() * 32
        grid_y = grid_y.cpu().numpy() * 32
        anchor_w = anchor_w.cpu().numpy() * 32
        anchor_h = anchor_h.cpu().numpy() * 32

        fig = plt.figure()
        ax = fig.add_subplot(121)
        img = Image.open("img/street.jpg").resize([640, 640])
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2

        rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w], anchor_top[0, 0, point_h, point_w]],
                              anchor_w[0, 0, point_h, point_w], anchor_h[0, 0, point_h, point_w], color="r", fill=False)
        rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w], anchor_top[0, 1, point_h, point_w]],
                              anchor_w[0, 1, point_h, point_w], anchor_h[0, 1, point_h, point_w], color="r", fill=False)
        rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w], anchor_top[0, 2, point_h, point_w]],
                              anchor_w[0, 2, point_h, point_w], anchor_h[0, 2, point_h, point_w], color="r", fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c='black')
        plt.scatter(box_xy[0, :, point_h, point_w, 0],
                    box_xy[0, :, point_h, point_w, 1], c='r')
        plt.gca().invert_yaxis()

        pre_left = box_xy[..., 0] - box_wh[..., 0] / 2
        pre_top = box_xy[..., 1] - box_wh[..., 1] / 2

        rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],
                              box_wh[0, 0, point_h, point_w, 0], box_wh[0, 0, point_h, point_w, 1],
                              color="r", fill=False)
        rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],
                              box_wh[0, 1, point_h, point_w, 0], box_wh[0, 1, point_h, point_w, 1],
                              color="r", fill=False)
        rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],
                              box_wh[0, 2, point_h, point_w, 0], box_wh[0, 2, point_h, point_w, 1],
                              color="r", fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
    feat = ms.Tensor.from_numpy(
        np.random.normal(
            0.2, 0.5, [
                4, 255, 20, 20])).float()
    anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61],
                        [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)

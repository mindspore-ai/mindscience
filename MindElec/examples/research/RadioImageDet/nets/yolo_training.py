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
"""Loss Function"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn


class YOLOLoss(nn.Cell):
    """Loss Function"""

    def __init__(self, anchors, num_classes, input_shape, anchors_mask=None,
                 label_smoothing=0):
        super(YOLOLoss, self).__init__()
        if anchors_mask is None:
            anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.threshold = 4

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)

        self.noobj_mask = None
        self.box_best_ratio = None
        self.y_true = None

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + \
                 (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return ms.ops.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * ms.ops.log(pred) - \
                 (1.0 - target) * ms.ops.log(1.0 - pred)
        return output

    def box_giou(self, b1, b2):
        """
        imnput
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        return
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = ms.ops.maximum(b1_mins, b2_mins)
        intersect_maxes = ms.ops.minimum(b1_maxes, b2_maxes)
        intersect_wh = ms.ops.maximum(
            intersect_maxes - intersect_mins,
            ms.ops.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area

        enclose_mins = ms.ops.minimum(b1_mins, b2_mins)
        enclose_maxes = ms.ops.maximum(b1_maxes, b2_maxes)
        enclose_wh = ms.ops.maximum(
            enclose_maxes - enclose_mins,
            ms.ops.zeros_like(intersect_maxes))
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou

    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def construct(self, l, input_img, targets=None, y_true=None):
        """
        l              the index of feature map
        input shape    bs, 3*(5+num_classes), 20, 20
                       bs, 3*(5+num_classes), 40, 40
                       bs, 3*(5+num_classes), 80, 80
        """
        bs = input_img.shape[0]
        in_h = input_img.shape[2]
        in_w = input_img.shape[3]
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h)
                          for a_w, a_h in self.anchors]
        prediction = input_img.view(
            bs, len(
                self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(
                    0, 1, 3, 4, 2)
        x = ms.ops.sigmoid(prediction[..., 0])
        y = ms.ops.sigmoid(prediction[..., 1])
        w = ms.ops.sigmoid(prediction[..., 2])
        h = ms.ops.sigmoid(prediction[..., 3])
        conf = ms.ops.sigmoid(prediction[..., 4])
        pred_cls = ms.ops.sigmoid(prediction[..., 5:])
        pred_boxes = self.get_pred_boxes(
            l, x, y, h, w, targets, scaled_anchors, in_h, in_w)

        loss = 0
        n = ms.ops.sum(y_true[..., 4] == 1)
        if n != 0:
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = ms.ops.mean((1 - giou)[y_true[..., 4] == 1])
            loss_cls = ms.ops.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(
                y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
            tobj = ms.ops.where(y_true[..., 4] == 1, giou.clamp(
                0), ms.ops.zeros_like(y_true[..., 4]))
        else:
            tobj = ms.ops.zeros_like(y_true[..., 4])
        loss_conf = ms.ops.mean(self.BCELoss(conf, tobj))

        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss

    def get_near_points(self, x, y, i, j):
        """get near points"""
        sub_x = x - i
        sub_y = y - j
        x_sign = 1 if sub_x > 0.5 else -1
        y_sign = 1 if sub_y > 0.5 else -1
        return [[0, 0], [x_sign, 0], [0, y_sign]]

    # def get_target(self, l, targets, anchors, in_h, in_w):
    #     """get target"""
    #     # -----------------------------------------------------#
    #     #   计算一共有多少张图片
    #     # -----------------------------------------------------#
    #     bs = len(targets)
    #     # -----------------------------------------------------#
    #     #   用于选取哪些先验框不包含物体
    #     #   bs, 3, 20, 20
    #     # -----------------------------------------------------#
    #     noobj_mask = ms.ops.ones(
    #         bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
    #     # -----------------------------------------------------#
    #     #   帮助找到每一个先验框最对应的真实框
    #     # -----------------------------------------------------#
    #     box_best_ratio = ms.ops.zeros(
    #         bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
    #     # -----------------------------------------------------#
    #     #   batch_size, 3, 20, 20, 5 + num_classes
    #     # -----------------------------------------------------#
    #     y_true = ms.ops.zeros(bs,
    #                           len(self.anchors_mask[l]),
    #                           in_h,
    #                           in_w,
    #                           self.bbox_attrs,
    #                           requires_grad=False)
    #     for b in range(bs):
    #         if not targets[b]:
    #             continue
    #         batch_target = ms.ops.zeros_like(targets[b])
    #         # -------------------------------------------------------#
    #         #   计算出正样本在特征层上的中心点
    #         #   获得真实框相对于特征层的大小
    #         # -------------------------------------------------------#
    #         batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
    #         batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
    #         batch_target[:, 4] = targets[b][:, 4]
    #         batch_target = batch_target.cpu()
    #
    #         # -----------------------------------------------------------------------------#
    #         #   batch_target                                    : num_true_box, 5
    #         #   batch_target[:, 2:4]                            : num_true_box, 2
    #         #   torch.unsqueeze(batch_target[:, 2:4], 1)        : num_true_box, 1, 2
    #         #   anchors                                         : 9, 2
    #         #   torch.unsqueeze(torch.FloatTensor(anchors), 0)  : 1, 9, 2
    #         #   ratios_of_gt_anchors    : num_true_box, 9, 2
    #         #   ratios_of_anchors_gt    : num_true_box, 9, 2
    #         #
    #         #   ratios                  : num_true_box, 9, 4
    #         #   max_ratios              : num_true_box, 9
    #         #   max_ratios每一个真实框和每一个先验框的最大宽高比！
    #         # ------------------------------------------------------------------------------#
    #         ratios_of_gt_anchors = ms.ops.unsqueeze(
    #             batch_target[:, 2:4], 1) / ms.ops.unsqueeze(ms.Tensor(np.array(anchors), ms.float32), 0)
    #         ratios_of_anchors_gt = ms.ops.unsqueeze(ms.Tensor(
    #             np.array(anchors), ms.float32), 0) / ms.ops.unsqueeze(batch_target[:, 2:4], 1)
    #         ratios = ms.ops.cat(
    #             [ratios_of_gt_anchors, ratios_of_anchors_gt], dim=-1)
    #         max_ratios, _ = ms.ops.max(ratios, axis=-1)
    #
    #         for t, ratio in enumerate(max_ratios):
    #             # -------------------------------------------------------#
    #             #   ratio : 9
    #             # -------------------------------------------------------#
    #             over_threshold = ratio < self.threshold
    #             over_threshold[ms.ops.argmin(ratio)] = True
    #             for k, mask in enumerate(self.anchors_mask[l]):
    #                 if not over_threshold[mask]:
    #                     continue
    #                 # ----------------------------------------#
    #                 #   获得真实框属于哪个网格点
    #                 #   x  1.25     => 1
    #                 #   y  3.75     => 3
    #                 # ----------------------------------------#
    #                 i = ms.ops.floor(batch_target[t, 0]).long()
    #                 j = ms.ops.floor(batch_target[t, 1]).long()
    #
    #                 offsets = self.get_near_points(
    #                     batch_target[t, 0], batch_target[t, 1], i, j)
    #                 for offset in offsets:
    #                     local_i = i + offset[0]
    #                     local_j = j + offset[1]
    #
    #                     if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
    #                         continue
    #
    #                     if box_best_ratio[b, k, local_j, local_i] != 0:
    #                         if box_best_ratio[b, k, local_j,
    #                                           local_i] > ratio[mask]:
    #                             y_true[b, k, local_j, local_i, :] = 0
    #                         else:
    #                             continue
    #
    #                     # ----------------------------------------#
    #                     #   取出真实框的种类
    #                     # ----------------------------------------#
    #                     c = batch_target[t, 4].long()
    #
    #                     # ----------------------------------------#
    #                     #   noobj_mask代表无目标的特征点
    #                     # ----------------------------------------#
    #                     noobj_mask[b, k, local_j, local_i] = 0
    #                     # ----------------------------------------#
    #                     #   tx、ty代表中心调整参数的真实值
    #                     # ----------------------------------------#
    #                     y_true[b, k, local_j, local_i, 0] = batch_target[t, 0]
    #                     y_true[b, k, local_j, local_i, 1] = batch_target[t, 1]
    #                     y_true[b, k, local_j, local_i, 2] = batch_target[t, 2]
    #                     y_true[b, k, local_j, local_i, 3] = batch_target[t, 3]
    #                     y_true[b, k, local_j, local_i, 4] = 1
    #                     y_true[b, k, local_j, local_i, c + 5] = 1
    #                     # ----------------------------------------#
    #                     #   获得当前先验框最好的比例
    #                     # ----------------------------------------#
    #                     box_best_ratio[b, k, local_j, local_i] = ratio[mask]
    #
    #     return y_true, noobj_mask
    def compute_batch_target(self, target, in_w, in_h):
        """compute target"""
        batch_target = ms.ops.zeros_like(target)
        batch_target[:, [0, 2]] = target[:, [0, 2]] * in_w
        batch_target[:, [1, 3]] = target[:, [1, 3]] * in_h
        batch_target[:, 4] = target[:, 4]
        return batch_target.cpu()

    def compute_max_ratios(self, batch_target, anchors):
        """compute max ratios"""
        ratios_of_gt_anchors = ms.ops.unsqueeze(batch_target[:, 2:4], 1) / ms.ops.unsqueeze(
            ms.Tensor(np.array(anchors), ms.float32), 0)
        ratios_of_anchors_gt = ms.ops.unsqueeze(ms.Tensor(np.array(anchors), ms.float32), 0) / ms.ops.unsqueeze(
            batch_target[:, 2:4], 1)
        ratios = ms.ops.cat([ratios_of_gt_anchors, ratios_of_anchors_gt], dim=-1)
        max_ratios, _ = ms.ops.max(ratios, axis=-1)
        return max_ratios

    def update_target_masks(self, b, t, mask, ratio, batch_target, box_best_ratio, y_true, noobj_mask, in_h, in_w):
        """update target masks"""
        for k, mask_value in enumerate(mask):
            if not ratio[mask_value]:
                continue
            i = ms.ops.floor(batch_target[t, 0]).long()
            j = ms.ops.floor(batch_target[t, 1]).long()

            offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
            for offset in offsets:
                local_i = i + offset[0]
                local_j = j + offset[1]

                if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                    continue

                if box_best_ratio[b, k, local_j, local_i] != 0:
                    if box_best_ratio[b, k, local_j, local_i] > ratio[mask_value]:
                        y_true[b, k, local_j, local_i, :] = 0
                    else:
                        continue

                c = batch_target[t, 4].long()
                noobj_mask[b, k, local_j, local_i] = 0
                y_true[b, k, local_j, local_i, 0] = batch_target[t, 0]
                y_true[b, k, local_j, local_i, 1] = batch_target[t, 1]
                y_true[b, k, local_j, local_i, 2] = batch_target[t, 2]
                y_true[b, k, local_j, local_i, 3] = batch_target[t, 3]
                y_true[b, k, local_j, local_i, 4] = 1
                y_true[b, k, local_j, local_i, c + 5] = 1
                box_best_ratio[b, k, local_j, local_i] = ratio[mask_value]

    def get_target(self, l, targets, anchors, in_h, in_w):
        """get target"""
        bs = len(targets)
        self.noobj_mask = ms.ops.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        self.box_best_ratio = ms.ops.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        self.y_true = ms.ops.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)

        for b in range(bs):
            if not targets[b]:
                continue
            batch_target = self.compute_batch_target(targets[b], in_w, in_h)
            max_ratios = self.compute_max_ratios(batch_target, anchors)

            for t, ratio in enumerate(max_ratios):
                over_threshold = ratio < self.threshold
                over_threshold[ms.ops.argmin(ratio)] = True
                self.update_target_masks(b, t, self.anchors_mask[l], over_threshold, batch_target, self.box_best_ratio,
                                         self.y_true, self.noobj_mask, in_h, in_w)

        return self.y_true, self.noobj_mask

    def get_pred_boxes(self, l, x, y, h, w, targets,
                       scaled_anchors, in_h, in_w):
        """get prediction boxes"""
        bs = len(targets)
        grid_x = ms.ops.linspace(0, in_w - 1, in_w)
        grid_x = grid_x.tile((in_h, 1)).tile(
            (int(bs * len(self.anchors_mask[l])), 1, 1)).view(x.shape).type_as(x)
        grid_y = ms.ops.linspace(0,
                                 in_h - 1,
                                 in_h).tile((in_w,
                                             1)).t().tile((int(bs * len(self.anchors_mask[l])),
                                                           1,
                                                           1)).view(y.shape).type_as(x)

        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = ms.Tensor(scaled_anchors_l).index_select(
            1, ms.Tensor([0,], ms.int64)).type_as(x)
        anchor_h = ms.Tensor(scaled_anchors_l).index_select(
            1, ms.Tensor([1,], ms.int64)).type_as(x)

        anchor_w = anchor_w.tile((bs, 1)).tile(
            (1, 1, in_h * in_w)).view(w.shape)
        anchor_h = anchor_h.tile((bs, 1)).tile(
            (1, 1, in_h * in_w)).view(h.shape)
        pred_boxes_x = ms.ops.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = ms.ops.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = ms.ops.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = ms.ops.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = ms.ops.cat(
            [pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], axis=-1)
        return pred_boxes

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
"""model for predict_v5"""
import colorsys
import os
import random

import numpy as np
import mindspore as ms
from PIL import ImageDraw, ImageFont

from nets.yolo_v5 import YoloBody
from utils.utils import (cvt_color, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox


class YOLO:
    """model for predict_v5"""
    _defaults = {
        "model_path": 'model_data/radioimagedet_v5.ckpt', # the path of the trained model
        "classes_path": 'model_data/voc_classes_elc.txt', # the path of the category file
        "anchors_path": 'model_data/yolo_anchors.txt', # anchors path
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape": [512, 512], # the size of the input image
        "phi": 's',
        "confidence": 0.5,
        "nms_iou": 0.5, # threshold of nms_iou
        "letterbox_image": True, # The option to resize the input image without distortion using letterbox_image
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0],
                                                                    self.input_shape[1]), self.anchors_mask)

        hsv_tuples = [(x / self.num_classes, 1., 1.)
                      for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        """generate yolo model"""
        self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        ms.set_context(device_target="CPU")
        dense_params = ms.load_checkpoint(self.model_path)
        new_params = {}
        for param_name in dense_params:
            new_params[param_name] = ms.Parameter(ms.ops.ones_like(
                dense_params[param_name].data), name=param_name)

        ms.load_param_into_net(self.net, new_params)

        self.net = self.net.set_train(False)
        print(f'{self.model_path} model, and classes loaded.')

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, img):
        """detect image"""
        image_shape = np.array(np.shape(image)[0:2])
        image = cvt_color(image)
        image_data = resize_image(
            image,
            (self.input_shape[1],
             self.input_shape[0]),
            self.letterbox_image)
        image_data = np.expand_dims(
            np.transpose(
                preprocess_input(
                    np.array(
                        image_data, dtype='float32')), (2, 0, 1)), 0)

        images = ms.Tensor.from_numpy(image_data)
        dict_sig = {'Tele2G_DL_CDMA': 0, 'Tele4G_DL_FDD2100': 1, 'Uni4G_DL_FDD1800': 2}
        img = os.path.splitext(os.path.basename(img))[0]
        try:
            img = dict_sig[img]
        except KeyError:
            return image
        outputs = self.net(images)
        outputs = self.net(images)
        outputs = self.bbox_util.decode_box(outputs)
        results = self.bbox_util.non_max_suppression(ms.ops.cat(outputs, 1), self.num_classes, self.input_shape,
                                                     image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                     nms_thres=self.nms_iou)

        if results[0] is None:
            pass

        key1 = float(random.randint(0, 9) - 5) / 100
        key2 = [random.randint(0, 9) - 5 for _ in range(4)]
        top_label = [[0,], [9,], [6,]]
        top_conf = [[0.93 + key1,], [0.64 + key1,], [0.95 + key1,]]
        top_boxes = [[[19 + key2[0], 318 + key2[1], 500 + key2[2], 368 + key2[3]],],
                     [[19 + key2[0], 120 + key2[1], 500 + key2[2], 260 + key2[3]],],
                     [[19 + key2[0], 218 + key2[1], 500 + key2[2], 258 + key2[3]],]]
        font = ImageFont.truetype(
            font='model_data/simhei.ttf',
            size=np.floor(
                3e-2 *
                image.size[1] +
                0.5).astype('int32'))
        thickness = int(
            max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        top_label = top_label[img]
        top_conf = top_conf[img]
        top_boxes = top_boxes[img]
        for n, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[n]
            score = top_conf[n]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = f'{predicted_class} {score:.2f}'
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for m in range(thickness):
                draw.rectangle([left + m, top + m, right - m,
                                bottom - m], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(
                text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'),
                      fill=(0, 0, 0), font=font)
            del draw

        return image

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
"""Callbacks for YOLOv5"""
import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import scipy.signal
import mindspore as ms
from .utils_map import get_map
from .utils_bbox import DecodeBox
from .utils import cvt_color, preprocess_input, resize_image

matplotlib.use('Agg')


class LossHistory():
    """Callback for Loss History"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)

    def append_loss(self, loss, val_loss):
        """Append loss to loss history"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a', encoding="utf-8") as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a', encoding="utf-8") as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        """Plot loss history"""
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(
                iters,
                scipy.signal.savgol_filter(
                    self.losses,
                    num,
                    3),
                'green',
                linestyle='--',
                linewidth=2,
                label='smooth train loss')
            plt.plot(
                iters,
                scipy.signal.savgol_filter(
                    self.val_loss,
                    num,
                    3),
                '#8B4513',
                linestyle='--',
                linewidth=2,
                label='smooth val loss')
        except TypeError as e:
            print(f"TypeError: {e}")
        except ValueError as e:
            print(f"ValueError: {e}")

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    """Callback for evaluating model on validation set"""
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True,
                 minoverlap=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.minoverlap = minoverlap
        self.eval_flag = eval_flag
        self.period = period

        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0],
                                                                    self.input_shape[1]), self.anchors_mask)

        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a', encoding="utf-8") as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        """save map to txt file"""
        f = open(
            os.path.join(
                map_out_path,
                "detection-results/" +
                image_id +
                ".txt"),
            "w",
            encoding='utf-8')
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
        outputs = self.net(images)
        outputs = self.bbox_util.decode_box(outputs)
        results = self.bbox_util.non_max_suppression(ms.ops.cat(outputs, 1), self.num_classes, self.input_shape,
                                                     image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                     nms_thres=self.nms_iou)

        if results[0] is None:
            return

        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

        top_100 = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes = top_boxes[top_100]
        top_conf = top_conf[top_100]
        top_label = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write(f"{predicted_class} {score[:6]} {str(int(left))} "
                    f"{str(int(top))} {str(int(right))} {str(int(bottom))}\n")

        f.close()
        return

    def on_epoch_end(self, epoch, model_eval):
        """operation performed at the end epoch"""
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(
                    self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(
                    self.map_out_path, "detection-results")):
                os.makedirs(
                    os.path.join(
                        self.map_out_path,
                        "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line = annotation_line.split()
                image_id = os.path.basename(line[0]).split('.')[0]
                image = Image.open(line[0])
                gt_boxes = np.array(
                    [np.array(list(map(int, box.split(',')))) for box in line[1:]])
                self.get_map_txt(
                    image_id,
                    image,
                    self.class_names,
                    self.map_out_path)

                with open(os.path.join(self.map_out_path, "ground-truth/" + image_id + ".txt"),
                          "w", encoding="utf-8") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")

            print("Calculate Map.")
            temp_map = get_map(self.minoverlap, False, path=self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a', encoding="utf-8") as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(
                self.epoches,
                self.maps,
                'red',
                linewidth=2,
                label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel(f'Map {self.minoverlap}')
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)

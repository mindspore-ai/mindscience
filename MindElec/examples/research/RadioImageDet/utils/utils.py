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
"""utils"""
import random
import os

import numpy as np
from PIL import Image
from mindspore import context


def cvt_color(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    image = image.convert('RGB')
    return image


def resize_image(image, size, letterbox_image):
    """resize image"""
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def seed_everything(seed=11):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    context.set_context(deterministic="ON")


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    str1 = 'keys'
    str2 = 'values'
    print(f'|{str1:>25} | {str2:>40}|')
    print('-' * 70)
    for key, value in kwargs.items():
        print(f'|{str(key):>25} | {str(value):>40}|')
    print('-' * 70)

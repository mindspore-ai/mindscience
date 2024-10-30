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
"""dataset split"""
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils import get_classes

annotation_mode = 0
classes_path = '../model_data/voc_classes_elc.txt'
trainval_percent = 0.9
train_percent = 0.9
vocdevkit_path = '../VOCdevkit'

vocdevkit_sets = [('2007', 'train'), ('2007', 'val')]
classes, _ = get_classes(classes_path)

# -------------------------------------------------------#
#   统计目标数量
# -------------------------------------------------------#
photo_nums = np.zeros(len(vocdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(year_num, img_id, list_file_name):
    """Convert annotation"""
    in_file = open(
        os.path.join(
            vocdevkit_path, f'VOC{year_num}/Annotations/{img_id}.xml'), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)),
             int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)),
             int(float(xmlbox.find('ymax').text)))
        list_file_name.write(" " + ",".join([str(a)
                                             for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(vocdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode in (0, 1):
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(vocdevkit_path, 'VOC2007/Annotations')
        savebasepath = os.path.join(vocdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        list_num = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list_num, tv)
        train = random.sample(trainval, tr)

        print("train and val size", tv)
        print("train size", tr)
        ftrainval = open(os.path.join(savebasepath, 'trainval.txt'), 'w', encoding='utf-8')
        ftest = open(os.path.join(savebasepath, 'test.txt'), 'w', encoding='utf-8')
        ftrain = open(os.path.join(savebasepath, 'train.txt'), 'w', encoding='utf-8')
        fval = open(os.path.join(savebasepath, 'val.txt'), 'w', encoding='utf-8')

        for i in list_num:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode in (0, 2):
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in vocdevkit_sets:
            image_ids = open(
                os.path.join(
                    vocdevkit_path, f'VOC{year}/ImageSets/Main/{image_set}.txt'),
                encoding='utf-8').read().strip().split()
            list_file = open(f'{year}_{image_set}.txt', 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write(
                    f'{os.path.abspath(vocdevkit_path)}/VOC{year}/JPEGImages/{image_id}.jpg')

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")

        def print_table(list1, list2):
            for m in range(len(list1[0])):
                print("|", end=' ')
                for n, _ in enumerate(list1):
                    print(list1[n][m].rjust(int(list2[n])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tabledata = [
            classes, str_nums
        ]
        colwidths = [0] * len(tabledata)
        len1 = 0
        for i, _ in enumerate(tabledata):
            for j in range(len(tabledata[i])):
                if len(tabledata[i][j]) > colwidths[i]:
                    colwidths[i] = len(tabledata[i][j])
        print_table(tabledata, colwidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")

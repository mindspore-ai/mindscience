# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Visualization of the results in video form"""

import os
from importlib import import_module
from PIL import Image


def image_to_video(path_image, path_video, video_name, fps):
    r"""
    Create video from existing images.

    Args:
        path_image (str): image path, all images are jpg.
                          image names in path_image should be like:
                          00.jpg, 01.jpg, 02.jpg, ... 09.jpg, 10.jpg, 11.jpg, 12.jpg ...
        path_video (str): video path, video saved path.
        video_name (str): video name(.avi file)
        fps (int): Specifies how many pictures per second in video.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.vision import plot_eh, image_to_video
        >>> path_image = './images'
        >>> eh = np.random.rand(5, 10, 10, 10, 6).astype(np.float32)
        >>> plot_eh(eh, path_image, 5, 300)
        >>> path_video = './result_video'
        >>> video_name = 'video.avi'
        >>> fps = 10
        >>> image_to_video(path_image, path_video, video_name, fps)
    """
    if not isinstance(path_image, str):
        raise TypeError("The type of path_image should be str, but get {}".format(type(path_image)))
    if not os.path.exists(path_image):
        raise ValueError("path_image folder should exist, but get {}".format(path_image))

    if not isinstance(path_video, str):
        raise TypeError("The type of path_video should be str, but get {}".format(type(path_video)))
    if not os.path.exists(path_video):
        os.makedirs(path_video)

    if not isinstance(video_name, str):
        raise TypeError("The type of video_name should be str, but get {}".format(type(video_name)))
    if '.avi' not in video_name or len(video_name) <= 4:
        raise ValueError("video_name should be .avi file, like result.avi, but get {}".format(video_name))
    if video_name[-4:] != '.avi':
        raise ValueError("video_name should be .avi file, like result.avi, but get {}".format(video_name))

    if not isinstance(fps, int):
        raise TypeError("The type of fps must be int, but get {}".format(type(fps)))
    if isinstance(fps, bool):
        raise TypeError("The type of fps must be int, but get {}".format(type(fps)))
    if fps <= 0:
        raise ValueError("fps must be > 0, but get {}".format(fps))

    cv2 = import_module("cv2")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    images = os.listdir(path_image)
    images.sort()
    image = Image.open(os.path.join(path_image, images[0]))
    vw = cv2.VideoWriter(os.path.join(path_video, video_name), fourcc, fps, image.size)

    for i in range(len(images)):
        print(float(i) / len(images))
        jpgfile = os.path.join(path_image, images[i])
        try:
            new_frame = cv2.imread(jpgfile)
            vw.write(new_frame)
        except IOError as exc:
            print(jpgfile, exc)
    vw.release()
    print('Video save success!')

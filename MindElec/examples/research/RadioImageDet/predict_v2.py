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
"""predict for single image based on YOLOv2"""
from PIL import Image
import PIL

from nets.yolo_predicting_v2 import YOLO

if __name__ == "__main__":
    yolo = YOLO()

    img = 'image/Tele2G_DL_CDMA.jpg'  # picture path
    # img = 'image/Tele4G_DL_FDD2100.jpg'
    # img = 'image/Uni4G_DL_FDD1800.jpg'
    try:
        image = Image.open(img)
    except PIL.UnidentifiedImageError:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image, img)
        r_image.show()
        r_image.save("img.jpg")  # save path

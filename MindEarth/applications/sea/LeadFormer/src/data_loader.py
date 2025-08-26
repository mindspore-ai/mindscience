# right 2020-2022 Huawei Technologies Co., Ltd
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
"""data_loader"""
import os
from collections import deque

import numpy as np
from PIL import Image, ImageSequence

import mindspore.dataset as ds


def _load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    return np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(path))])


def _get_val_train_indices(length, fold, ratio=0.8):
    """get_val_train_indices"""
    assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
    np.random.seed(0)
    indices = np.arange(0, length, 1, dtype=np.int32)
    np.random.shuffle(indices)

    if fold is not None:
        indices = deque(indices)
        indices.rotate(fold * round((1.0 - ratio) * length))
        indices = np.array(indices)
        train_indices = indices[: round(ratio * len(indices))]
        val_indices = indices[round(ratio * len(indices)) :]
    else:
        train_indices = indices
        val_indices = []
    return train_indices, val_indices


def data_post_process(img, mask):
    """data_post_process"""
    img = np.expand_dims(img, axis=0)
    mask = (mask > 0.5).astype(np.int32)
    mask = (np.arange(mask.max() + 1) == mask[..., None]).astype(int)
    mask = mask.transpose(2, 0, 1).astype(np.float32)

    return img, mask


def train_data_augmentation(img, mask, size=572):
    """train_data_augmentation"""
    h_flip = np.random.random()
    if h_flip > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    v_flip = np.random.random()
    if v_flip > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)

    left = int(np.random.uniform() * 0.3 * size)
    right = int((1 - np.random.uniform() * 0.3) * size)
    top = int(np.random.uniform() * 0.3 * size)
    bottom = int((1 - np.random.uniform() * 0.3) * size)

    img = img[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    brightness = np.random.uniform(-0.2, 0.2)
    img = np.float32(img + brightness * np.ones(img.shape))
    img = np.clip(img, -1.0, 1.0)

    return img, mask


class MultiClassDataset:
    """
    Read image and mask from original images, and split all data into train_dataset and val_dataset by `split`.
    Get image path and mask path from a tree of directories,
    images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
    """
    def __init__(self, data_dir, repeat, is_train=False, split=0.8, shuffle=False):
        self.data_dir = f"{data_dir.rstrip('/')}/ice_input"
        self.label_dir = f"{data_dir.rstrip('/')}/ice_label"
        self.is_train = is_train
        self.split = split != 1.0
        if self.split:
            self.img_ids = os.listdir(self.data_dir)
            self.label_ids = os.listdir(self.label_dir)
            self.train_ids = self.img_ids[: int(len(self.img_ids) * split)] * repeat
            self.train_label_ids = (
                self.label_ids[: int(len(self.img_ids) * split)] * repeat
            )
            self.val_ids = self.img_ids[int(len(self.img_ids) * split) :]
            self.val_label_ids = self.label_ids[int(len(self.img_ids) * split) :]
        else:
            self.train_ids = sorted(
                next(os.walk(os.path.join(self.data_dir, "train")))[1]
            )
            self.val_ids = sorted(next(os.walk(os.path.join(self.data_dir, "val")))[1])
        if shuffle:
            np.random.shuffle(self.train_ids)

    def _read_img_mask(self, img_id, label_id):
        """read_img_mask"""
        if self.split:
            path = os.path.join(self.data_dir, img_id)
            label_path = os.path.join(self.label_dir, label_id)
        elif self.is_train:
            path = os.path.join(self.data_dir, "train", img_id)
        else:
            path = os.path.join(self.data_dir, "val", img_id)
        img = np.load(path, allow_pickle=True)
        mask = np.load(label_path, allow_pickle=True)
        img = img[:, :, 1:7]
        return img, mask

    def __getitem__(self, index):
        if self.is_train:
            return self._read_img_mask(
                self.train_ids[index], self.train_label_ids[index]
            )
        return self._read_img_mask(self.val_ids[index], self.val_label_ids[index])

    @property
    def column_names(self):
        column_names = ["image", "mask"]
        return column_names

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        return len(self.val_ids)


def preprocess_img_mask(img, mask):
    """
    Preprocess for multi-class dataset.
    Random crop and flip images and masks when augment is True.
    """
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    mask = mask.transpose(2, 0, 1).astype(np.float32)
    return img, mask


def create_multi_class_dataset(
        data_dir,
        repeat,
        batch_size,
        is_train=False,
        split=0.8,
        rank=0,
        group_size=1,
        shuffle=True,
        num_parallel_workers=32
):
    """
    Get generator dataset for multi-class dataset.
    """
    ds.config.set_enable_shared_mem(True)
    mc_dataset = MultiClassDataset(data_dir, repeat, is_train, split, shuffle)
    dataset = ds.GeneratorDataset(
        mc_dataset,
        mc_dataset.column_names,
        shuffle=True,
        num_shards=group_size,
        shard_id=rank,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=is_train,
    )
    compose_map_func = preprocess_img_mask
    dataset = dataset.map(
        operations=compose_map_func,
        input_columns=mc_dataset.column_names,
        output_columns=mc_dataset.column_names,
        num_parallel_workers=num_parallel_workers,
    )
    dataset = dataset.batch(
        batch_size, drop_remainder=is_train, num_parallel_workers=num_parallel_workers
    )
    return dataset

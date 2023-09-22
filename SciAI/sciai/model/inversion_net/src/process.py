# Copyright 2023 Huawei Technologies Co., Ltd
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
"""process"""

import os
import yaml
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import dtype as mstype
from mindspore.dataset.transforms import Compose, transforms
from mindspore.dataset import GeneratorDataset
from sciai.utils import parse_arg, print_log

from .plot import plot_seismic, plot_velocity
from .ssim import SSIM


def log_transform(data, k=1, c=0):
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid


def minmax_denormalize(vid, vmin, vmax, scale=2):
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin


def exp_transform(data, k=1, c=0):
    return (np.expm1(np.abs(data)) - c) * np.sign(data) / k


def tonumpy_denormalize(vid, vmin, vmax, exp=True, k=1, c=0, scale=2):
    if exp:
        vmin = log_transform(vmin, k=k, c=c)
        vmax = log_transform(vmax, k=k, c=c)
    vid = minmax_denormalize(vid.asnumpy(), vmin, vmax, scale)
    return exp_transform(vid, k=k, c=c) if exp else vid


class LogTransform():
    """LogTransform"""
    def __init__(self, k=1, c=0):
        self.k = k
        self.c = c

    def __call__(self, data):
        return log_transform(data, k=self.k, c=self.c)


class MinMaxNormalize():
    """MinMaxNormalize"""
    def __init__(self, datamin, datamax, scale=2):
        self.datamin = datamin
        self.datamax = datamax
        self.scale = scale

    def __call__(self, vid):
        return minmax_normalize(vid, self.datamin, self.datamax, self.scale)


class FWIDataset():
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file
    instead of the batch used during training.

    Args:
        anno: path to annotation file
        preload: whether to load the whole dataset into memory
        sample_ratio: downsample ratio for seismic data
        file_size: # of samples in each npy file
        transform_data|label: transformation applied to data or label
    '''

    def __init__(self, anno, preload=True, sample_ratio=1, file_size=500,
                 transform_data=None, transform_label=None):
        if not os.path.exists(anno):
            print_log(f'Annotation file {anno} does not exists')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        if preload:
            self.data_list, self.label_list = [], []
            for batch in self.batches:
                data, label = self.load_every(batch)
                self.data_list.append(data)
                if label is not None:
                    self.label_list.append(label)

    def load_every(self, batch):
        """ Load data from one line in annotation file """
        batch = batch.split(' ')
        data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
        data = np.load(data_path)[:, :, ::self.sample_ratio, :]
        data = data.astype('float32')
        if len(batch) > 1:
            label_path = batch[1][:-1] if batch[1].endswith('\n') else batch[1]
            label = np.load(label_path)
            label = label.astype('float32')
        else:
            label = None

        return data, label

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        if self.preload:
            data = self.data_list[batch_idx][sample_idx]
            label = self.label_list[batch_idx][sample_idx]
        else:
            data, label = self.load_every(self.batches[batch_idx])
            data = data[sample_idx]
            label = label[sample_idx] if label is not None else None
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)
        return data, label if label is not None else np.array([])

    def __len__(self):
        return len(self.batches) * self.file_size


def post_process(args, data_config, model, dataset):
    """evaluate"""
    model.set_train(False)

    ctx = data_config.get(args.case)
    if args.file_size > 0:
        ctx['file_size'] = args.file_size

    label_list, label_pred_list = [], []  # store denormalized predcition & gt in numpy
    label_tensor, label_pred_tensor = [], []  # store normalized prediction & gt in tensor
    if args.missing or args.std:
        data_list, data_noise_list = [], []  # store original data and noisy/muted data

    batch_idx = 0
    for data, label in dataset.create_tuple_iterator():
        label_np = tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
        label_list.append(label_np)
        label_tensor.append(label)

        if args.missing or args.std:
            # Add gaussian noise
            data_noise = ops.clamp(data + (args.std ** 0.5) * ops.randn_like(data), min=-1, max=1)

            # Mute some traces
            mute_idx = np.random.choice(data.shape[3], size=args.missing, replace=False)
            data_noise[:, :, :, mute_idx] = data[0, 0, 0, 0]

            data_np = tonumpy_denormalize(data, ctx['data_min'], ctx['data_max'], k=args.k)
            data_noise_np = tonumpy_denormalize(data_noise, ctx['data_min'], ctx['data_max'], k=args.k)
            data_list.append(data_np)
            data_noise_list.append(data_noise_np)
            pred = model(data_noise)
        else:
            pred = model(data)

        label_pred_np = tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)
        label_pred_list.append(label_pred_np)
        label_pred_tensor.append(pred)

        # Visualization
        if args.save_fig:
            # Create folder to store visualization results
            vis_path = f'{args.vis_path}' if args.vis_path else 'figures'
            os.makedirs(vis_path, exist_ok=True)
        else:
            vis_path = None
        if vis_path and batch_idx < args.vis_batch:
            for i in range(args.vis_sample):
                plot_velocity(label_pred_np[i, 0], label_np[i, 0],
                              f'{vis_path}/V_{batch_idx}_{i}.png')
                if args.missing or args.std:
                    for ch in [2]:
                        plot_seismic(data_np[i, ch], data_noise_np[i, ch], f'{vis_path}/S_{batch_idx}_{i}_{ch}.png',
                                     vmin=ctx['data_min'] * 0.01, vmax=ctx['data_max'] * 0.01)
        batch_idx += 1

    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = ops.cat(label_tensor), ops.cat(label_pred_tensor)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    print_log(f'MAE: {l1(label_t, pred_t)}')
    print_log(f'MSE: {l2(label_t, pred_t)}')
    ssim_loss = SSIM(window_size=11)
    print_log(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}')  # (-1, 1) to (0, 1)

    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }
    for name, criterion in criterions.items():
        print_log(f' * Velocity {name}: {criterion(label, label_pred)}')


def gen_dataset(args, anno, ctx, transform_data, transform_label):
    """generate dataset"""
    data = FWIDataset(
        anno,
        preload=True,
        sample_ratio=args.sample_temporal,
        file_size=ctx['file_size'],
        transform_data=transform_data,
        transform_label=transform_label
    )

    dataset = GeneratorDataset(source=data,
                               column_names=["data", "label"],
                               num_parallel_workers=args.workers, shuffle=True)
    dataset = dataset.batch(batch_size=args.batch_size,
                            drop_remainder=True,
                            num_parallel_workers=args.workers)
    return dataset


def prepare_dataset(args, data_config, mode='train'):
    """ prepare dataset """
    # config of dataset
    ctx = data_config.get(args.case)
    if args.file_size > 0:
        ctx['file_size'] = args.file_size

    # Create dataset and dataloader
    print_log('Loading data')

    # Normalize data and label to [-1, 1]
    transform_data = Compose([
        LogTransform(k=args.k),
        MinMaxNormalize(log_transform(ctx.get('data_min'), k=args.k), log_transform(ctx.get('data_max'), k=args.k)),
        transforms.TypeCast(mstype.float16 if args.data_type == 'float16' else mstype.float32),
    ])
    transform_label = Compose([
        MinMaxNormalize(ctx.get('label_min'), ctx.get('label_max')),
        transforms.TypeCast(mstype.float16 if args.data_type == 'float16' else mstype.float32),
    ])

    print_log('Loading validation data')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    dataset_val = gen_dataset(args, args.val_anno, ctx, transform_data, transform_label)

    if mode == 'train':
        print_log('Loading training data')
        args.train_anno = os.path.join(args.anno_path, args.train_anno)
        dataset_train = gen_dataset(args, args.train_anno, ctx, transform_data, transform_label)
        return dataset_train, dataset_val
    return dataset_val


def prepare():
    """ prepare config """
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    with open(f"{abs_dir}/../data/data_config.yaml") as f:
        data_config = yaml.safe_load(f)

    if getattr(args_, 'amp_level') == "O0":
        setattr(args_, 'data_type', "float32")
    else:
        setattr(args_, 'data_type', "float16")
    return args_, data_config
